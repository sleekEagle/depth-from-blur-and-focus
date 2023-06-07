from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from .submodule import *
import pdb
from models.featExactor2 import FeatExactor

# a=torch.ones(10,12,6,8,8)
# a[:,:,0,:,:]*=0
# a[:,:,1,:,:]*=1
# a[:,:,2,:,:]*=2
# a[:,:,3,:,:]*=3
# a[:,:,4,:,:]*=4
# a[:,:,5,:,:]*=5

# inf=a[:,:,-1]
# inf_=torch.unsqueeze(inf,dim=2)
# inf=torch.repeat_interleave(inf_,repeats=a.shape[2],dim=2)
# val=a/inf
# val=val[:,:,:-1,:,:]
# cat=torch.cat((val,inf_),dim=2)

# Ours-FV (use_diff=0) and Ours-DFV (use_diff=1) model

class LinBlur(nn.Module):
    def __init__(self, clean,level=1, use_div=1):
        super(LinBlur, self).__init__()

        self.clean = clean
        self.feature_extraction = FeatExactor()
        self.level = level

        self.use_div = use_div
        assert level >= 1 and level <= 4
        assert use_div == 0 or use_div == 1

        if level == 1:
            self.decoder3 = decoderBlock(2,16,16, stride=(1,1,1),up=False, nstride=1)
        elif level == 2:
            self.decoder3 = decoderBlock(2,32,32, stride=(1,1,1),up=False, nstride=1)
            self.decoder4 = decoderBlock(2,32,32, up=True)
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
        self.disp_reg = disparityregression(1)

    #divide feature sets 
    def div_feat_volume(self, vol):
        inf=vol[:,:,-1]
        inf_=torch.unsqueeze(inf,dim=2)
        inf=torch.repeat_interleave(inf_,repeats=vol.shape[2],dim=2)
        val=vol/inf
        val=val[:,:,:-1,:,:]
        cat=torch.cat((val,inf_),dim=2)
        return cat

    def forward(self, stack, focal_dist):
        b, n, c, h, w = stack.shape
        #the feature calculation is independent of the focal stack 
        #i.e. it treats every image independently
        input_stack = stack.reshape(b*n, c, h , w)

        conv4, conv3, conv2, conv1  = self.feature_extraction(input_stack)

        # conv3d take b, c, d, h, w
        _vol4, _vol3, _vol2, _vol1  = conv4.reshape(b, n, -1, h//32, w//32).permute(0, 2, 1, 3, 4), \
                                 conv3.reshape(b, n, -1, h//16, w//16).permute(0, 2, 1, 3, 4),\
                                 conv2.reshape(b, n, -1, h//8, w//8).permute(0, 2, 1, 3, 4),\
                                 conv1.reshape(b, n, -1, h//4, w//4).permute(0, 2, 1, 3, 4)

        print('before div _vol1:'+str(_vol1.shape))
        if self.use_div == 1:
            vol4, vol3, vol2, vol1 = self.div_feat_volume(_vol4), self.div_feat_volume(_vol3),\
                                     self.div_feat_volume(_vol2), self.div_feat_volume(_vol1)
            print('vol1:'+str(vol1.shape))
        else:
            vol4, vol3, vol2, vol1 =  _vol4, _vol3, _vol2, _vol1
            print('vol1:'+str(vol1.shape))
        
        if self.level == 1:
            _, cost3 = self.decoder3(vol1)
            print('cost3:'+str(cost3.shape))

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

        cost3 = F.interpolate(cost3, [h, w], mode='bilinear')
        return cost3
        
        # pred3, std3 = self.disp_reg(F.softmax(cost3,1),focal_dist, uncertainty=True)

        # different output based on level
        # stacked = [pred3]
        # stds = [std3]
        # if self.training :
        #     if self.level >= 2:
        #         cost4 = F.interpolate(cost4, [h, w], mode='bilinear')
        #         pred4, std4 = self.disp_reg(F.softmax(cost4, 1), focal_dist, uncertainty=True)
        #         stacked.append(pred4)
        #         stds.append(std4)
        #         if self.level >=3 :
        #             cost5 = F.interpolate((cost5).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
        #             pred5, std5 = self.disp_reg(F.softmax(cost5, 1), focal_dist, uncertainty=True)
        #             stacked.append(pred5)
        #             stds.append(std5)
        #             if self.level >=4 :
        #                 cost6 = F.interpolate((cost6).unsqueeze(1), [focal_dist.shape[1], h, w], mode='trilinear').squeeze(1)
        #                 pred6, std6 = self.disp_reg(F.softmax(cost6, 1), focal_dist, uncertainty=True)
        #                 stacked.append(pred6)
        #                 stds.append(std6)
        #     return stacked, stds, None
        # else:
        #     return pred3,torch.squeeze(std3), F.softmax(cost3,1).squeeze()
