from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np
import pdb

# code adopted from https://github.com/gengshan-y/high-res-stereo/blob/master/models/submodule.py

class sepConv3dBlock(nn.Module):
    '''
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1),pad=(0,1,1)):
        super(sepConv3dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes,stride)
        self.conv1 = sepConv3d(in_planes, out_planes, (3,3,3), stride, pad)
        self.conv2 = sepConv3d(out_planes, out_planes, (3,3,3), (1,1,1), pad)


    def forward(self,x):
        #x:torch.Size([60, 16, 2, 64, 64])
        out = F.relu(self.conv1(x),inplace=True)
        #out:torch.Size([60, 16, 3, 64, 64])
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out),inplace=True)
        return out
    
class sepConv2dBlock(nn.Module):
    '''
    Separable 3d convolution block as 2 separable convolutions and a projection
    layer
    '''
    def __init__(self, in_planes, out_planes, stride=(1,1,1)):
        super(sepConv2dBlock, self).__init__()
        if in_planes == out_planes and stride==(1,1,1):
            self.downsample = None
        else:
            self.downsample = projfeat3d(in_planes, out_planes,stride)
        self.conv1 = sepConv3d(in_planes, out_planes, 3, stride, 1)
        self.conv2 = sepConv3d(out_planes, out_planes, 3, (1,1,1), 1)


    def forward(self,x):
        out = F.relu(self.conv1(x),inplace=True)
        if self.downsample:
            x = self.downsample(x)
        out = F.relu(x + self.conv2(out),inplace=True)
        return out




class projfeat3d(nn.Module):
    '''
    Turn 3d projection into 2d projection
    '''
    def __init__(self, in_planes, out_planes, stride):
        super(projfeat3d, self).__init__()
        self.stride = stride
        self.conv1 = nn.Conv2d(in_planes, out_planes, (1,1), padding=(0,0), stride=stride[:2],bias=False)
        self.bn = nn.BatchNorm2d(out_planes)

    def forward(self,x):
        b,c,d,h,w = x.size()
        x = self.conv1(x.view(b,c,d,h*w))
        x = self.bn(x)
        x = x.view(b,-1,d//self.stride[0],h,w)
        return x
    
# model=projfeat3d(10,10,stride=(1,1,1))
# img=torch.rand(10,10,50,50)
# model(img)


# original conv3d block
def sepConv3d(in_planes, out_planes, kernel_size, stride, pad,bias=False):
    if bias:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                         nn.BatchNorm3d(out_planes))
    
def sepConv2d(in_planes, out_planes, kernel_size, stride, pad,bias=False):
    if bias:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias))
    else:
        return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=bias),
                         nn.BatchNorm2d(out_planes))



class disparityregression(nn.Module):
    def __init__(self, divisor):
        super(disparityregression, self).__init__()
        self.divisor = divisor

    def forward(self, x, focal_dist=None, uncertainty=False):
        disp = focal_dist.unsqueeze(-1).unsqueeze(-1)
        out = torch.sum(x*disp,1, keepdim=True) * self.divisor

        if uncertainty:
            with torch.no_grad():
                std = torch.sqrt(torch.sum(x * (out - disp)**2, 1, keepdim=True))
            return out, std.detach()
        else:
            return out

class distregression(nn.Module):
    def __init__(self):
        super(distregression, self).__init__()

    def forward(self, s1,blur):
        #for +""45 degrees lines
        s1est1=s1-blur
        s1est1=s1est1.unsqueeze(dim=1)
        s1est1=torch.repeat_interleave(s1est1,s1.shape[1]+1,dim=1)
        imgsize=s1.shape[-1]
        bs=s1.shape[0]
        n=s1.shape[1]
        #for -45 degrees lines
        s1est2=s1+blur
        s1est2=s1est2.unsqueeze(dim=1)
        s1est2=torch.repeat_interleave(s1est2,s1.shape[1]+1,dim=1)

        mask=torch.zeros(2,bs,n+1,n,imgsize,imgsize)
        for i in range(1,n+1):
            mask[0,:,i,:i,:,:]=1
        for i in range(0,n+1):
            mask[1,:,i,i:,:]=1
        mask=mask.cuda()

        s1est=s1est1*mask[1]+s1est2*mask[0]

        s1eststd=torch.std(s1est,dim=2)
        argmin=torch.argmin(s1eststd,dim=1)

        argmin=torch.unsqueeze(argmin,dim=1).unsqueeze(dim=2)
        argmin=torch.repeat_interleave(argmin,repeats=n,dim=2)

        sel=torch.gather(s1est,dim=1,index=argmin)
        s2_pred=torch.mean(sel,dim=2)[:,0,:,:]
        return s2_pred

# class decoderBlock(nn.Module):
#     def __init__(self, nconvs, inchannelF,channelF,stride=(1,1,1),up=False, nstride=1,pool=False):
#         super(decoderBlock, self).__init__()
#         self.pool=pool
#         stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)
#         self.convs = [sepConv3dBlock(inchannelF,channelF,stride=stride[0])]
#         for i in range(1,nconvs):
#             self.convs.append(sepConv3dBlock(channelF,channelF, stride=stride[i]))
#         self.convs = nn.Sequential(*self.convs)

#         self.classify = nn.Sequential(sepConv3d(channelF, channelF, 3, (1,1,1), 1),
#                                        nn.ReLU(inplace=True),
#                                        sepConv3d(channelF, 1, 3, (1,1,1),1,bias=True),
#                                        nn.ReLU(inplace=True))

#         self.up = False
#         if up:
#             self.up = True
#             self.up = nn.Sequential(nn.Upsample(scale_factor=(1,2,2),mode='trilinear'),
#                                  sepConv3d(channelF, channelF//2, 3, (1,1,1),1,bias=False),
#                                  nn.ReLU(inplace=True))

#         if pool:
#             self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF, channelF, 1, (1,1,1), 0),
#                                sepConv3d(channelF, channelF, 1, (1,1,1), 0),
#                                sepConv3d(channelF, channelF, 1, (1,1,1), 0),
#                                sepConv3d(channelF, channelF, 1, (1,1,1), 0)])
            
 

#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#                 if hasattr(m.bias,'data'):
#                     m.bias.data.zero_()


#     def forward(self,fvl):
#         # left
#         #beforre fvl:torch.Size([1, 16, 6, 64, 64])
#         fvl = self.convs(fvl)
#         #fvl:torch.Size([1, 16, 6, 64, 64])
#         # pooling
#         if self.pool:
#             fvl_out = fvl
#             _,_,d,h,w=fvl.shape
#             for i,pool_size in enumerate(np.linspace(1,min(d,h,w)//2,4,dtype=int)):
#                 kernel_size = (int(d/pool_size), int(h/pool_size), int(w/pool_size))
#                 out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)       
#                 out = self.pool_convs[i](out)
#                 out = F.upsample(out, size=(d,h,w), mode='trilinear')
#                 fvl_out = fvl_out + 0.25*out
#             fvl = F.relu(fvl_out/2.,inplace=True)


#         if self.training:
#             # classification
#             costl = self.classify(fvl)
#             if self.up:
#                 fvl = self.up(fvl)
#         else:
#             # classification
#             if self.up:
#                 fvl = self.up(fvl)
#                 costl=fvl
#             else:
#                 costl = self.classify(fvl)
#                 #after clssify:torch.Size([1, 1, 6, 64, 64])

#         return fvl,costl.squeeze(1)
    

class decoderBlock(nn.Module):
    def __init__(self, nconvs, inchannelF,channelF,stride=(1,1,1),up=False, nstride=1,pool=False):
        super(decoderBlock, self).__init__()
        self.pool=pool
        stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)
        self.convs = [sepConv3dBlock(inchannelF,channelF,stride=stride[0],pad=(1,1,1))]
        # for i in range(1,nconvs):
        #     self.convs.append(sepConv3dBlock(channelF,channelF, stride=stride[i]))
        self.convs = nn.Sequential(*self.convs)

        self.classify = nn.Sequential(sepConv3d(channelF, channelF, 3, (1,1,1), 1),
                                    nn.ReLU(inplace=True),
                                    sepConv3d(channelF, channelF, 3, (1,1,1), 1),
                                    nn.ReLU(inplace=True),
                                    sepConv3d(channelF, 1, (2,3,3), (1,1,1), (0,1,1)),
                                    nn.ReLU(inplace=True))

        self.aggregate = nn.Sequential(nn.Conv2d(2, 16, 3, (1,1), 1),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(16, 1, 3, (1,1),1,bias=True),
                                       nn.ReLU(inplace=True))

        self.up = False
        if up:
            self.up = True
            self.up = nn.Sequential(nn.Upsample(scale_factor=(1,2,2),mode='trilinear'),
                                 sepConv3d(channelF, channelF//2, 3, (1,1,1),1,bias=False),
                                 nn.ReLU(inplace=True))

        if pool:
            self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0)])
            
 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()


    def forward(self,fvl):
        # left
        #fvl:torch.Size([60, 1, 32, 64, 64])
        fvl = self.convs(fvl)
        #fvl after:torch.Size([5*bs, 16, 2, 64, 64])
        # pooling
        if self.pool:
            fvl_out = fvl
            _,_,d,h,w=fvl.shape
            for i,pool_size in enumerate(np.linspace(1,min(d,h,w)//2,4,dtype=int)):
                kernel_size = (int(d/pool_size), int(h/pool_size), int(w/pool_size))
                out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)       
                out = self.pool_convs[i](out)
                out = F.upsample(out, size=(d,h,w), mode='trilinear')
                fvl_out = fvl_out + 0.25*out
            fvl = F.relu(fvl_out/2.,inplace=True)

        if self.training:
            # classification
            costl = self.classify(fvl)
            #costl:torch.Size([60, 1, 1, 64, 64])
            costl=torch.squeeze(costl,dim=1)
            #costtl:torch.Size([60, 1, 64, 64])
            if self.up:
                fvl = self.up(fvl)
        else:
            # classification
            if self.up:
                fvl = self.up(fvl)
                costl=fvl
            else:
                costl = self.classify(fvl)
                costl=torch.squeeze(costl,dim=1)
                #after classify:torch.Size([5, 2, 64, 64])
                #after ag:torch.Size([5, 1, 64, 64])
        return fvl,costl
    

class decoderBlock2D(nn.Module):
    def __init__(self, nconvs, inchannelF,channelF,stride=(1,1,1),up=False, nstride=1,pool=False):
        super(decoderBlock2D, self).__init__()
        self.pool=pool
        stride = [stride]*nstride + [(1,1,1)] * (nconvs-nstride)
        self.convs = [sepConv3dBlock(inchannelF,channelF,stride=stride[0])]
        for i in range(1,nconvs):
            self.convs.append(sepConv3dBlock(channelF,channelF, stride=stride[i]))
        self.convs = nn.Sequential(*self.convs)

        self.classify = nn.Sequential(sepConv3d(channelF, channelF, 3, (1,1,1), 1),
                                       nn.ReLU(inplace=True),
                                       sepConv3d(channelF, 1, 3, (1,1,1),1,bias=True),
                                       nn.ReLU(inplace=True))

        self.up = False
        if up:
            self.up = True
            self.up = nn.Sequential(nn.Upsample(scale_factor=(1,2,2),mode='trilinear'),
                                 sepConv3d(channelF, channelF//2, 3, (1,1,1),1,bias=False),
                                 nn.ReLU(inplace=True))

        if pool:
            self.pool_convs = torch.nn.ModuleList([sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0),
                               sepConv3d(channelF, channelF, 1, (1,1,1), 0)])
            
 

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                n = m.kernel_size[0] * m.kernel_size[1]*m.kernel_size[2] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if hasattr(m.bias,'data'):
                    m.bias.data.zero_()


    def forward(self,fvl):
        # left
        fvl = self.convs(fvl)
        # pooling
        if self.pool:
            fvl_out = fvl
            _,_,d,h,w=fvl.shape
            for i,pool_size in enumerate(np.linspace(1,min(d,h,w)//2,4,dtype=int)):
                kernel_size = (int(d/pool_size), int(h/pool_size), int(w/pool_size))
                out = F.avg_pool3d(fvl, kernel_size, stride=kernel_size)       
                out = self.pool_convs[i](out)
                out = F.upsample(out, size=(d,h,w), mode='trilinear')
                fvl_out = fvl_out + 0.25*out
            fvl = F.relu(fvl_out/2.,inplace=True)


        if self.training:
            # classification
            costl = self.classify(fvl)
            if self.up:
                fvl = self.up(fvl)
        else:
            # classification
            if self.up:
                fvl = self.up(fvl)
                costl=fvl
            else:
                costl = self.classify(fvl)

        return fvl,costl.squeeze(1)
    
'''
x                 x
 \              /
 x               x
   \          /
    x       x
     \      /
       x    x

---------------------------------------
measures the 'quality' of the price-wise linear ness of the 
predicted blur values
'''
class LinearLoss(nn.Module):
    def __init__(self):
        super(LinearLoss, self).__init__()
        self.criterion = torch.nn.MSELoss()
    def forward(self, s1,s2_est,gt_disp,blur_est):
        valid_mask = (gt_disp>0).detach()
        s1=torch.unsqueeze(s1,dim=2).unsqueeze(dim=3)
        s1=torch.repeat_interleave(s1,repeats=gt_disp.shape[2],dim=2).repeat_interleave(repeats=gt_disp.shape[3],dim=3)
        s2_=torch.repeat_interleave(s2_est,repeats=s1.shape[1],dim=1)
        valid_mask=torch.repeat_interleave(valid_mask,repeats=s1.shape[1],dim=1)        
        mask=(s1<s2_).int()
        mask[mask==0]=-1
        GT_blur=(s2_-s1)*mask
        loss=self.criterion(GT_blur[valid_mask],blur_est[valid_mask])
        return loss


























