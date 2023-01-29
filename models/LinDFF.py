from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models.submodule import *
import pdb
from models.featExactor2 import FeatExactor

# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model

class DFFNet(nn.Module):
    def __init__(self, clean,level=1, use_diff=1):
        super(DFFNet, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()
        self.level = level

        self.use_diff = use_diff
        assert level >= 1 and level <= 4
        assert use_diff == 0 or use_diff == 1

        if level == 1:
            self.decoder3 = decoderBlock(2,16,16, stride=(1,1,1),up=False, nstride=1)
        elif level == 2:
            self.decoder3 = decoderBlock(2,32,32, stride=(1,1,1),up=False, nstride=1)
            self.decoder4 =  decoderBlock(2,32,32, up=True)
        elif level == 3:
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 64, 32, up=True)
            self.decoder5 = decoderBlock(2, 64, 64, up=True, pool=True)
        else:
            self.decoder3 = decoderBlock(2, 32, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 64, 32,  up=True)
            self.decoder5 = decoderBlock(2, 128, 64, up=True, pool=True)
            self.decoder6 = decoderBlock(2, 128, 128, up=True, pool=True)

        # reg
        self.distreg=distregression()


    def diff_feat_volume1(self, vol):
        vol_out = vol[:,:, :-1] - vol[:, :, 1:]
        return torch.cat([vol_out, vol[:,:, -1:]], dim=2) # last elem is  vol[:,:, -1] - 0

    def forward(self, stack, focal_dist):
        #create the mask 
        b, n, c, h, w = stack.shape

        mask=torch.zeros(2,b,n+1,n,w,w)
        for i in range(1,n+1):
            mask[0,:,i,:i,:,:]=1
        for i in range(0,n+1):
            mask[1,:,i,i:,:]=1
            
        input_stack = stack.reshape(b*n, c, h , w)

        conv4, conv3, conv2, conv1  = self.feature_extraction(input_stack)

        # conv3d take b, c, d, h, w
        _vol4, _vol3, _vol2, _vol1  = conv4.reshape(b, n, -1, h//32, w//32).permute(0, 2, 1, 3, 4), \
                                 conv3.reshape(b, n, -1, h//16, w//16).permute(0, 2, 1, 3, 4),\
                                 conv2.reshape(b, n, -1, h//8, w//8).permute(0, 2, 1, 3, 4),\
                                 conv1.reshape(b, n, -1, h//4, w//4).permute(0, 2, 1, 3, 4)


        if self.use_diff == 1:
            vol4, vol3, vol2, vol1 = self.diff_feat_volume1(_vol4), self.diff_feat_volume1(_vol3),\
                                     self.diff_feat_volume1(_vol2), self.diff_feat_volume1(_vol1)
        else:
            vol4, vol3, vol2, vol1 =  _vol4, _vol3, _vol2, _vol1

        if self.level == 1:
            _, cost3 = self.decoder3(vol1)

        elif self.level == 2:
            feat4_2x, cost4 = self.decoder4(vol2)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        elif self.level == 3:
            feat5_2x, cost5 = self.decoder5(vol3)
            feat4 = torch.cat((feat5_2x, vol2), dim=1)

            feat4_2x, cost4 = self.decoder4(feat4)
            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        else:
            feat6_2x, cost6 = self.decoder6(vol4)
            
            #divide cost (estimated blur) with blur when focused at infinity (i.e. last image of the focal stack) 
            cost6+=1e-5
            print('cost6:'+str(cost6.shape))
            infblur6=torch.unsqueeze(cost6[:,-1,:,:],dim=1)
            infblur6=torch.repeat_interleave(infblur6,cost6.shape[1],dim=1)
            cost6=cost6/infblur6
            print('post division cost6:'+str(cost6.shape))

            feat5 = torch.cat((feat6_2x, vol3), dim=1)
            feat5_2x, cost5 = self.decoder5(feat5)
            cost5+=1e-5
            infblur5=torch.unsqueeze(cost5[:,-1,:,:],dim=1)
            infblur5=torch.repeat_interleave(infblur5,cost5.shape[1],dim=1)
            cost5=cost5/infblur5

            feat4 = torch.cat((feat5_2x, vol2), dim=1)            
            feat4_2x, cost4 = self.decoder4(feat4)
            cost4+=1e-5
            infblur4=torch.unsqueeze(cost4[:,-1,:,:],dim=1)
            infblur4=torch.repeat_interleave(infblur4,cost4.shape[1],dim=1)
            cost4=cost4/infblur4

            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)
            cost3+=1e-5
            infblur3=torch.unsqueeze(cost3[:,-1,:,:],dim=1)
            infblur3=torch.repeat_interleave(infblur3,cost3.shape[1],dim=1)
            cost3=cost3/infblur3

        cost3 = F.interpolate(cost3, [h, w], mode='bilinear')
        #pred3, std3 = self.disp_reg(F.softmax(cost3,1),focal_dist, uncertainty=True)
        #create s1
        s1=torch.unsqueeze(focal_dist,dim=2).unsqueeze(dim=3)
        s1=torch.repeat_interleave(s1,cost3.shape[-1],dim=2).repeat_interleave(cost3.shape[-1],dim=3)
        pred3=self.distreg(s1,cost3,mask)
        std3=0

        # different output based on level
        stacked = [pred3]
        stds = [std3]
        if self.training :
            if self.level >= 2:
                cost4 = F.interpolate(cost4, [h, w], mode='bilinear')
                #pred4, std4 = self.disp_reg(F.softmax(cost4, 1), focal_dist, uncertainty=True)
                pred4=self.distreg(s1,cost4,mask)
                std4=0
                stacked.append(pred4)
                stds.append(std4)
                if self.level >=3 :
                    cost5 = F.interpolate((cost5).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                    pred5=self.distreg(s1,cost5,mask)
                    std5=0
                    #pred5, std5 = self.disp_reg(F.softmax(cost5, 1), focal_dist, uncertainty=True)
                    stacked.append(pred5)
                    stds.append(std5)
                    if self.level >=4 :
                        cost6 = F.interpolate((cost6).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
                        #pred6, std6 = self.disp_reg(F.softmax(cost6, 1), focal_dist, uncertainty=True)
                        pred6=self.distreg(s1,cost6,mask)
                        std6=0
                        stacked.append(pred6)
                        stds.append(std6)
            return stacked, stds, None
        else:
            return pred3,torch.squeeze(std3), F.softmax(cost3,1).squeeze()

model = DFFNet(clean=False,level=4, use_diff=1)
model = nn.DataParallel(model)
model.train()
model.cuda()

img_stack=torch.rand(2,5,3,256,256)
foc_dist=torch.rand(2,5)
foc_dist.cuda().get_device()

s1=torch.unsqueeze(foc_dist,dim=2).unsqueeze(dim=3)
s1=torch.repeat_interleave(s1,cost3.shape[-1],dim=2).repeat_interleave(cost3.shape[-1],dim=3)

out=model(img_stack,foc_dist)

'''
cost=torch.rand(2,5,256,256) 
cost[:,0,:,:]=cost[:,0,:,:]/cost[:,-1,:,:]
infblur=torch.unsqueeze(cost[:,-1,:,:],dim=1)
infblur=torch.repeat_interleave(infblur,cost.shape[1],dim=1)
a=cost/infblur
'''











