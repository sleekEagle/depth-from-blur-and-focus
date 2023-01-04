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

