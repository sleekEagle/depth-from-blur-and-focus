from __future__ import print_function
import argparse
import os
from os.path import join
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from models import LinDFF,LinDFF1,DFFNet,LinDef,LinBlur1
torch.backends.cudnn.benchmark=True

'''
Main code for Ours-FV and Ours-DFV training 
'''
parser = argparse.ArgumentParser(description='DFVDFF')
# === dataset =====
parser.add_argument('--dataset', default=['blender'], nargs='+',  help='data Name')
parser.add_argument('--DDFF12_pth', default=None, help='DDFF12 data path')
parser.add_argument('--data_path', default='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\', help='FOD data path')
parser.add_argument('--FoD_scale', default=1.0,
                    help='FoD dataset gt scale for loss balance, because FoD_GT: 0.1-1.5, DDFF12_GT 0.02-0.28, '
                         'empirically we find this scale help improve the model performance for our method and DDFF')
# ==== hypo-param =========
parser.add_argument('--stack_num', type=int ,default=6, help='num of image in a stack, please take a number in [2, 10]')
parser.add_argument('--level', type=int ,default=4, help='num of layers in network, please take a number in [1, 4]')
parser.add_argument('--use_diff', default=0, type=int, choices=[0,1], help='if use differential feat, 0: None,  1: diff cost volume')
parser.add_argument('--lvl_w', nargs='+', default=[8./15, 4./15, 2./15, 1./15],  help='for std weight')

parser.add_argument('--batchsize', type=int, default=12, help='samples per batch')
parser.add_argument('--model', default='LinBlur1', help='save path')


# ====== log path ==========
parser.add_argument('--loadmodel', default='C:\\Users\\lahir\\Documents\\model_699.tar',   help='path to pre-trained checkpoint if any')
parser.add_argument('--figpath', default='C:\\Users\\lahir\\data\\lindefblur\\pred_blur_lindiff1_nondiv_700epochs_pred_infalso\\', help='save path')
parser.add_argument('--isVali', type=int, default=0, help='Save images for the valudation dataset ?. If 0 used train dataset')
parser.add_argument('--seed', type=int, default=2021, metavar='S',  help='random seed (default: 2021)')

args = parser.parse_args()
args.logname = '_'.join(args.dataset)


# ============ init ===============
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

start_epoch = 1
best_loss = 1e5
total_iter = 0

if args.model == 'LinDFF':
    model = LinDFF(clean=False,level=args.level, use_diff=args.use_diff)
    model = nn.DataParallel(model)
    model.cuda()
if args.model == 'LinDFF1':
    model = LinDFF1(clean=False,level=args.level, use_diff=args.use_diff)
    model = nn.DataParallel(model)
    model.cuda()
elif args.model == 'DFFNet':
    model = DFFNet(clean=False,level=args.level, use_diff=args.use_diff)
    model = nn.DataParallel(model)
    model.cuda()
elif args.model == 'LinDef':
    model = LinDef(3,1, 16, flag_step2=False)
    model = nn.DataParallel(model)
    model.cuda()
elif args.model == 'LinBlur1':
    model = LinBlur1(clean=False,level=1,use_div=0)
    model = nn.DataParallel(model)
    model.cuda()

# ========= load model if any ================
if args.loadmodel is not None:
    pretrained_dict = torch.load(args.loadmodel)
    pretrained_dict['state_dict'] =  {k:v for k,v in pretrained_dict['state_dict'].items()  } #if ('disp' not in k)
    model.load_state_dict(pretrained_dict['state_dict'],strict=False)

    if 'epoch' in pretrained_dict:
        start_epoch = pretrained_dict['epoch']

    if 'iters' in pretrained_dict:
        total_iter = pretrained_dict['iters']

    if 'best' in pretrained_dict:
        best_loss = pretrained_dict['best']

    print('load model from {}, start epoch {}, best_loss {}'.format(args.loadmodel, start_epoch, best_loss))

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))


# ============ data loader ==============
#Create data loader
if  'DDFF12' in args.dataset:
    from dataloader import DDFF12Loader
    database = '/data/DFF/my_ddff_trainVal.h5' if args.DDFF12_pth is None else  args.DDFF12_pth
    DDFF12_train = DDFF12Loader(database, stack_key="stack_train", disp_key="disp_train", n_stack=args.stack_num,
                                 min_disp=0.02, max_disp=0.28)
    DDFF12_val = DDFF12Loader(database, stack_key="stack_val", disp_key="disp_val", n_stack=args.stack_num,
                                      min_disp=0.02, max_disp=0.28, b_test=False)
    DDFF12_train, DDFF12_val = [DDFF12_train], [DDFF12_val]
else:
    DDFF12_train, DDFF12_val = [], []

if 'FoD500' in args.dataset:
    from dataloader import FoD500Loader
    database = '/data/DFF/baseline/defocus-net/data/fs_6/' if args.FoD_pth is None else  args.FoD_pth
    FoD500_train, FoD500_val = FoD500Loader(database, scale=args.FoD_scale)
    FoD500_train, FoD500_val =  [FoD500_train], [FoD500_val]
else:
    FoD500_train, FoD500_val = [], []

if 'blender' not in args.dataset:
    dataset_train = torch.utils.data.ConcatDataset(DDFF12_train  + FoD500_train )
    dataset_val = torch.utils.data.ConcatDataset(FoD500_val) # we use the model perform better on  DDFF12_val

    TrainImgLoader = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=4, batch_size=args.batchsize, shuffle=True, drop_last=True)
    ValImgLoader = torch.utils.data.DataLoader(dataset=dataset_val, num_workers=1, batch_size=1, shuffle=False, drop_last=True)

    print('%d batches per epoch'%(len(TrainImgLoader)))

if 'blender' in args.dataset:
    from dataloader import focalblender
    loaders, total_steps = focalblender.load_data(args.data_path,aif=False,train_split=0.8,fstack=1,WORKERS_NUM=0,
        BATCH_SIZE=args.batchsize,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,-1],REQ_F_IDX=[0,1,2,3,4,5],MAX_DPT=1.0)
    

def save_blurpred(img,blur,depth,s1,n,savepath):
    import matplotlib.pyplot as plt
    import numpy as np
    fdist=s1.numpy()[0,:]
    cost3=model(img,s1)
    for idx in range(n):
        i,j=(np.random.random(size=2)*depth.shape[-1]).tolist()
        i,j=int(i),int(j)
        b_pred=cost3[0,:-1,i,j].cpu().detach().numpy()
        b_pred_=b_pred*(fdist[:-1]-2.9e-3)
        s2_pred=cost3[0,-1,i,j].cpu().detach().numpy()
        #get GT blur
        b=blur[0,:-1,i,j].numpy()*(fdist[:-1]-2.9e-3)
        d=depth[0,0,i,j].numpy().item()

        fig = plt.figure()
        ax = plt.subplot(111)

        line1,=ax.plot(fdist[:-1],b_pred_, marker="o", markersize=9, markeredgecolor="red")
        line2,=ax.plot(fdist[:-1],b, marker="x", markersize=9, markeredgecolor="blue")
        line3,=ax.plot([d],[0], marker="o", markersize=9, markeredgecolor="red")
        line4,=ax.plot([1/s2_pred],[0], marker="^", markersize=9, markeredgecolor="green")
        line1.set_label('predicteed blur')
        line2.set_label('GT blur')
        line3.set_label('GT depth')
        line4.set_label('depth predicted from inf image')
        ax.legend()
        fig.savefig(join(savepath,str(idx)+'.jpg'))

loaders, total_steps = focalblender.load_data(args.data_path,aif=False,train_split=0.8,fstack=1,WORKERS_NUM=0,
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,-1],REQ_F_IDX=[0,1,2,3,4,5],MAX_DPT=1.0)

train_path=os.path.join(args.figpath,'train')
test_path=os.path.join(args.figpath,'test')
if not os.path.exists(train_path):
    os.makedirs(train_path)
if not os.path.exists(test_path):
    os.makedirs(test_path)

#save train images
for st_iter, sample_batch in enumerate(loaders[0]):
    # Setting up input and output data
    img = sample_batch['input'].float()
    blur = sample_batch['blur'].float()
    depth = sample_batch['output'].float()
    s1=sample_batch['fdist']
    break   
save_blurpred(img,blur,depth,s1,100,train_path)

#save test images
for st_iter, sample_batch in enumerate(loaders[1]):
    # Setting up input and output data
    img = sample_batch['input'].float()
    blur = sample_batch['blur'].float()
    depth = sample_batch['output'].float()
    s1=sample_batch['fdist']
    break   
save_blurpred(img,blur,depth,s1,100,test_path)
