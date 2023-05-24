from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models.submodule import *
import pdb
from models.featExactor2 import FeatExactor

# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model

class LinDFF1(nn.Module):
    def __init__(self, clean,level=1, use_diff=1):
        super(LinDFF1, self).__init__()

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
            self.decoder3 = decoderBlock(2, 48, 32, stride=(1, 1, 1), up=False, nstride=1)
            self.decoder4 = decoderBlock(2, 96, 32,  up=True)
            self.decoder5 = decoderBlock(2, 192, 64, up=True, pool=True)
            self.decoder6 = decoderBlock(2, 256, 128, up=True, pool=True)

        # reg
        self.distreg=distregression()
        self.disp_reg=disparityregression(1)
        self.f=2.9e-3
        self.clip=0.01


    def diff_feat_volume1(self, vol):
        vol_out = vol[:,:, :-1] - vol[:, :, 1:]
        return torch.cat([vol_out, vol[:,:, -1:]], dim=2) # last elem is  vol[:,:, -1] - 0

    def forward(self, stack, focal_dist):
        #create the mask 
        b, n, c, h, w = stack.shape
            
        input_stack = stack.reshape(b*n, c, h , w)

        conv4, conv3, conv2, conv1  = self.feature_extraction(input_stack)

        # conv3d take b, c, d, h, w
        vol4_, vol3_, vol2_, vol1_  = conv4.reshape(b, n, -1, h//32, w//32), \
                                 conv3.reshape(b, n, -1, h//16, w//16),\
                                 conv2.reshape(b, n, -1, h//8, w//8),\
                                 conv1.reshape(b, n, -1, h//4, w//4)
        
        inf4_=vol4_[:,-1,:,:,:]
        inf3_=vol3_[:,-1,:,:,:]
        inf2_=vol2_[:,-1,:,:,:]
        inf1_=vol1_[:,-1,:,:,:]

        vol4=torch.empty([b,0,vol4_.shape[2]*2,h//32, w//32]).cuda()
        vol3=torch.empty([b,0,vol3_.shape[2]*2,h//16, w//16]).cuda()
        vol2=torch.empty([b,0,vol2_.shape[2]*2,h//8, w//8]).cuda()
        vol1=torch.empty([b,0,vol1_.shape[2]*2,h//4, w//4]).cuda()
        for i in range(n-1):
            b4_=vol4_[:,i,:,:,:]
            b3_=vol3_[:,i,:,:,:]
            b2_=vol2_[:,i,:,:,:]
            b1_=vol1_[:,i,:,:,:]

            # print('inf4:'+str(inf4_.shape))
            # print('b4:'+str(b4_.shape))
            ar4_=torch.cat((b4_,inf4_),dim=1)
            ar4_=torch.unsqueeze(ar4_,dim=1)
            # print('ar4:'+str(ar4_.shape))

            # print('inf3:'+str(inf3_.shape))
            # print('b3:'+str(b3_.shape))
            ar3_=torch.cat((b3_,inf3_),dim=1)
            ar3_=torch.unsqueeze(ar3_,dim=1)
            # print('ar3:'+str(ar3_.shape))

            # print('inf2:'+str(inf2_.shape))
            # print('b2:'+str(b2_.shape))
            ar2_=torch.cat((b2_,inf2_),dim=1)
            ar2_=torch.unsqueeze(ar2_,dim=1)
            # print('ar2:'+str(ar2_.shape))

            # print('inf1:'+str(inf1_.shape))
            # print('b1:'+str(b1_.shape))
            ar1_=torch.cat((b1_,inf1_),dim=1)
            ar1_=torch.unsqueeze(ar1_,dim=1)
            # print('ar1:'+str(ar1_.shape))

            vol4=torch.cat((vol4,ar4_),dim=1)
            vol3=torch.cat((vol3,ar3_),dim=1)
            vol2=torch.cat((vol2,ar2_),dim=1)
            vol1=torch.cat((vol1,ar1_),dim=1)
        # print('vol4:'+str(vol4.shape))   
        vol4=vol4.permute(0,2,1,3,4)
        # print('vol4:'+str(vol4.shape)) 
        vol3=vol3.permute(0,2,1,3,4)
        vol2=vol2.permute(0,2,1,3,4)
        vol1=vol1.permute(0,2,1,3,4)

        if self.level == 4:
            feat6_2x, cost6 = self.decoder6(vol4)
            # print('cost6:'+str(cost6.shape))
            # print('feat6:'+str(feat6_2x.shape))

            feat5 = torch.cat((feat6_2x, vol3), dim=1)
            feat5_2x, cost5 = self.decoder5(feat5)

            feat4 = torch.cat((feat5_2x, vol2), dim=1) 
            feat4_2x, cost4 = self.decoder4(feat4)

            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

        #to supervise blur
        # cost=torch.rand(1,6,256,256)
        # F.interpolate(cost, [10,10], mode='bilinear').shape

        cost3=F.interpolate(cost3, [h, w], mode='bilinear')
        if(self.training):
            cost4=F.interpolate(cost4, [h, w], mode='bilinear')
            cost5=F.interpolate(cost5, [h, w], mode='bilinear')
            cost6=F.interpolate(cost6, [h, w], mode='bilinear')
    
        #pred3, std3 = self.disp_reg(F.softmax(cost3,1),focal_dist, uncertainty=True)
        #create s1
        s1=torch.unsqueeze(focal_dist,dim=2).unsqueeze(dim=3)
        s1=torch.repeat_interleave(s1,cost3.shape[-1],dim=2).repeat_interleave(cost3.shape[-1],dim=3)
        s1=s1[:,0:n-1,:,:]

        #multiply blur with (s1-f)
        mul3=cost3*(s1-self.f)
        if(self.training):
            mul4=cost4*(s1-self.f)
            mul5=cost5*(s1-self.f)
            mul6=cost6*(s1-self.f)

        pred3=self.distreg(s1,mul3)
        #print('focal_dist:'+str(focal_dist.shape))
        #pred3, std3 = self.disp_reg(F.softmax(cost3, 1), focal_dist[:,0:-1], uncertainty=True)
        std3=0
        # different output based on level
        stacked = [pred3]
        stds = [std3]
        cost_stacked=[mul3]
        if self.training :
            if self.level >= 2:
                pred4=self.distreg(s1,mul4)
                std4=0
                stacked.append(pred4)
                stds.append(std4)
                cost_stacked.append(mul4)
                if self.level >=3 :
                    pred5=self.distreg(s1,mul5)
                    std5=0
                    #pred5, std5 = self.disp_reg(F.softmax(cost5, 1), focal_dist, uncertainty=True)
                    stacked.append(pred5)
                    stds.append(std5)
                    cost_stacked.append(mul5)
                    if self.level >=4 :
                        pred6=self.distreg(s1,mul6)
                        std6=0
                        stacked.append(pred6)
                        stds.append(std6)
                        cost_stacked.append(mul6)
            return stacked, stds, cost_stacked
        else:
            return pred3,std3,cost3
        
'''
model = LinDFF(clean=False,level=4, use_diff=0)
model = nn.DataParallel(model)
model.train()
model.cuda()

img_stack=torch.rand(2,10,3,256,256).cuda()
foc_dist=torch.rand(2,10).cuda()
foc_dist.cuda().get_device()

s1=torch.unsqueeze(foc_dist,dim=2).unsqueeze(dim=3)
#s1=torch.repeat_interleave(s1,cost3.shape[-1],dim=2).repeat_interleave(cost3.shape[-1],dim=3)
out=model(img_stack,foc_dist)

#test with real data
from dataloader import DDFF12,focalblender
blenderpath='C:\\Users\\lahir\\focalstacks\\datasets\\defocusnet_N1\\'
loaders, total_steps = focalblender.load_data(blenderpath,aif=False,train_split=0.8,fstack=1,WORKERS_NUM=0,
        BATCH_SIZE=2,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4,5],MAX_DPT=1.0)

for batch_idx, sample_batch in enumerate(loaders[0]):
    img_stack=sample_batch['input'].float()
    gt_disp=sample_batch['output'][:,-1,:,:]
    gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
    foc_dist=sample_batch['fdist'].float()

    model(img_stack.cuda(),foc_dist.cuda())
'''

'''
cost=torch.rand(2,5,256,256) 
cost[:,0,:,:]=cost[:,0,:,:]/cost[:,-1,:,:]
infblur=torch.unsqueeze(cost[:,-1,:,:],dim=1)
infblur=torch.repeat_interleave(infblur,cost.shape[1],dim=1)
a=cost/infblur
'''











