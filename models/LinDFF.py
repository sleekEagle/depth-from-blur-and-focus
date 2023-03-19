from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from models.submodule import *
import pdb
from models.featExactor2 import FeatExactor

# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model

class LinDFF(nn.Module):
    def __init__(self, clean,level=1, use_diff=1):
        super(LinDFF, self).__init__()

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
            infblur3=torch.unsqueeze(cost3[:,-1,:,:],dim=1)
            infblur3=torch.repeat_interleave(infblur3,cost3.shape[1],dim=1)
            infblur3=torch.clamp(infblur3,min=0.1)
            cost3=cost3/infblur3

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

            feat5 = torch.cat((feat6_2x, vol3), dim=1)
            feat5_2x, cost5 = self.decoder5(feat5)

            feat4 = torch.cat((feat5_2x, vol2), dim=1)            
            feat4_2x, cost4 = self.decoder4(feat4)

            feat3 = torch.cat((feat4_2x, vol1), dim=1)
            _, cost3 = self.decoder3(feat3)

            #divide cost (estimated blur) with blur when focused at infinity (i.e. last image of the focal stack) 
            #cost6=torch.add(cost6,1e-5)
            #print('cost6:'+str(cost6.shape))
            infblur6=torch.unsqueeze(cost6[:,-1,:,:],dim=1)
            infblur6=torch.repeat_interleave(infblur6,cost6.shape[1],dim=1)
            infblur6=torch.clamp(infblur6,min=self.clip)
            cost6=cost6/infblur6

            # cost5=torch.add(cost5,1e-5)
            infblur5=torch.unsqueeze(cost5[:,-1,:,:],dim=1)
            infblur5=torch.repeat_interleave(infblur5,cost5.shape[1],dim=1)
            infblur5=torch.clamp(infblur5,min=self.clip)
            cost5=cost5/infblur5
            
            # cost4=torch.add(cost4,1e-5)
            infblur4=torch.unsqueeze(cost4[:,-1,:,:],dim=1)
            infblur4=torch.repeat_interleave(infblur4,cost4.shape[1],dim=1)
            infblur4=torch.clamp(infblur4,min=self.clip)
            cost4=cost4/infblur4
            
            # cost3=torch.add(cost3,1e-5)
            infblur3=torch.unsqueeze(cost3[:,-1,:,:],dim=1)
            infblur3=torch.repeat_interleave(infblur3,cost3.shape[1],dim=1)
            infblur3=torch.clamp(infblur3,min=self.clip)
            cost3=cost3/infblur3
        
        #remove the infinity focused values
        cost3=cost3[:,0:n-1,:,:]
        cost4=cost4[:,0:n-1,:,:]
        cost5=cost5[:,0:n-1,:,:]
        cost6=cost6[:,0:n-1,:,:]

        cost3 = F.interpolate(cost3, [h, w], mode='bilinear')
        #pred3, std3 = self.disp_reg(F.softmax(cost3,1),focal_dist, uncertainty=True)
        #create s1
        s1=torch.unsqueeze(focal_dist,dim=2).unsqueeze(dim=3)
        s1=torch.repeat_interleave(s1,cost3.shape[-1],dim=2).repeat_interleave(cost3.shape[-1],dim=3)
        s1=s1[:,0:n-1,:,:]

        #multiply blur with (s1-f)
        cost3=cost3*(s1-self.f)
        pred3=self.distreg(s1,cost3)
        #print('focal_dist:'+str(focal_dist.shape))
        #pred3, std3 = self.disp_reg(F.softmax(cost3, 1), focal_dist[:,0:-1], uncertainty=True)
        std3=0

        # different output based on level
        stacked = [pred3]
        stds = [std3]
        cost_stacked=[cost3]
        if self.training :
            if self.level >= 2:
                cost4 = F.interpolate(cost4, [h, w], mode='bilinear')
                #pred4, std4 = self.disp_reg(F.softmax(cost4, 1), focal_dist, uncertainty=True)
                cost4=cost4*(s1-self.f)
                pred4=self.distreg(s1,cost4)
                std4=0
                stacked.append(pred4)
                stds.append(std4)
                cost_stacked.append(cost4)
                if self.level >=3 :
                    cost5 = F.interpolate(cost5, [h, w], mode='bilinear')
                    cost5=cost5*(s1-self.f)
                    pred5=self.distreg(s1,cost5)
                    std5=0
                    #pred5, std5 = self.disp_reg(F.softmax(cost5, 1), focal_dist, uncertainty=True)
                    stacked.append(pred5)
                    stds.append(std5)
                    cost_stacked.append(cost5)
                    if self.level >=4 :
                        cost6 = F.interpolate(cost6, [h, w], mode='bilinear')
                        cost6=cost6*(s1-self.f)
                        #pred6, std6 = self.disp_reg(F.softmax(cost6, 1), focal_dist, uncertainty=True)
                        pred6=self.distreg(s1,cost6)
                        std6=0
                        stacked.append(pred6)
                        stds.append(std6)
                        cost_stacked.append(cost6)
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











