from __future__ import print_function
import argparse
import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
torch.backends.cudnn.benchmark=True
from glob import glob
from dataloader import FoD500Loader

'''
database="C:\\usr\\wiss\\maximov\\RD\\DepthFocus\\Datasets\\focal_data\\"
focus_dist=[0.1, 0.15, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 10.0, float('inf')]
focus_dist_req=[0.1, 0.15, 0.3, 0.7,1.0,float('inf')]
FoD500_train, FoD500_val = FoD500Loader(database, scale=1,focus_dist=focus_dist,focus_dist_req=focus_dist_req)
FoD500_train, FoD500_val =  [FoD500_train], [FoD500_val]
dataset_train = torch.utils.data.ConcatDataset(FoD500_train)
TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=0, batch_size=20, shuffle=True, drop_last=True)


for batch_idx, (img_stack, gt_disp,foc_dist) in enumerate(TrainImgLoader):
        break

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

imgs=torch.swapaxes(img_stack,2,-1)
for i in range(6):
    imgplot = plt.imshow(imgs[10,i,:,:,0])
    plt.show()

'''
#check the variation of blur predicted by a trained model
from models import DFFNet as DFFNet
from dataloader import FoD500Loader
import matplotlib.pyplot as plt
import numpy as np

# construct model
model = DFFNet( clean=False,level=4, use_diff=1)
model = nn.DataParallel(model)
model.cuda()
loadmodel='C:\\Users\\lahir\\models\\best.tar'
ckpt_name = os.path.basename(os.path.dirname(loadmodel))# we use the dirname to indicate training setting

if loadmodel is not None:
    pretrained_dict = torch.load(loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items() if 'disp' not in k}
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)
else:
    print('run with random init')
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

focus_dist=[0.1,.15,.3,0.7,1.5]
focus_dist_req=[0.1,.15,.3,0.7,1.5]
datapth='C:\\Users\\lahir\\focusdata\\fs_6\\fs_6\\'
dataset_train, dataset_validation = FoD500Loader(datapth, scale=1,focus_dist=focus_dist,focus_dist_req=focus_dist_req)
dataloader = torch.utils.data.DataLoader(dataset=dataset_validation, num_workers=1, batch_size=10, shuffle=False)

for inx, (img_stack, gt_disp_, foc_dist) in enumerate(dataloader):
    break

i=5
gt_disp=gt_disp_[i,:,:,:]

plt.imshow(np.squeeze(gt_disp.cpu().detach().numpy()))
plt.show()

model.eval()
#predict
with torch.no_grad():
    torch.cuda.synchronize()
    pred_disp, std, focusMap = model(img_stack, (foc_dist))
    torch.cuda.synchronize()

plt.imshow(np.squeeze(pred_disp[i,:,:,:].cpu().detach().numpy()))
plt.show()
plt.imshow(np.squeeze(focusMap[i,:,:,:].cpu().detach().numpy()[0,:,:]))
plt.show()

plt.plot(focusMap[i,:,:,:].cpu().detach().numpy()[:,111,200])
plt.show()






