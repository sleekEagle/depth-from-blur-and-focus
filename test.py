from models import LinBlur
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
 
model = LinBlur(clean=False,level=4, use_div=1)
model = nn.DataParallel(model)
model.cuda()

from dataloader import focalblender
blenderpath='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\'
loaders, total_steps = focalblender.load_data(blenderpath,aif=False,train_split=0.8,fstack=1,WORKERS_NUM=0,
    BATCH_SIZE=12,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,100000],REQ_F_IDX=[0,1,2,3,4,5],MAX_DPT=1.0)

for batch_idx, sample_batch in enumerate(loaders[0]):
    img_stack=sample_batch['input'].float()
    gt_disp=sample_batch['output'][:,-1,:,:]
    gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
    foc_dist=sample_batch['fdist'].float()
    blur=sample_batch['blur'].float()
    break

model.train()
img_stack_in   = Variable(torch.FloatTensor(img_stack))
gt_disp    = Variable(torch.FloatTensor(gt_disp))
img_stack, gt_disp, blur,foc_dist = img_stack_in.cuda(),  gt_disp.cuda(), blur[:,0:-1,:,:].cuda(),foc_dist.cuda()
stacked, stds, cost_stacked = model(img_stack, foc_dist)
