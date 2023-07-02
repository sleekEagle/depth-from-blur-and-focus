from __future__ import print_function
import argparse
import os
from os.path import join
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import time
from models import LinDFF,LinDFF1,DFFNet,LinDef,LinBlur1
torch.backends.cudnn.benchmark=True
from glob import glob
from models.submodule import LinearLoss
import logging
from datetime import datetime
from utils import logger, write_log


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

parser.add_argument('--lr', type=float, default=0.0001,  help='learning rate')
parser.add_argument('--epochs', type=int, default=700, help='number of epochs to train')
parser.add_argument('--batchsize', type=int, default=12, help='samples per batch')
parser.add_argument('--model', default='LinBlur1', help='save path')
parser.add_argument('--bweight', type=float, default=1.0,  help='learning rate')


# ====== log path ==========
parser.add_argument('--loadmodel', default=None,   help='path to pre-trained checkpoint if any')
parser.add_argument('--resultspth', default='C:\\Users\\lahir\\code\\defocus\\linmodels\\pp', help='save path')
parser.add_argument('--seed', type=int, default=2021, metavar='S',  help='random seed (default: 2021)')

args = parser.parse_args()
args.logname = '_'.join(args.dataset)

#setting up logging
if not os.path.exists(args.resultspth):
    os.makedirs(args.resultspth)
now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H_%M_%S")+'.log'
logpath=join(args.resultspth,dt_string)
logging.basicConfig(filename=logpath, format='%(levelname)s:%(asctime)s:%(message)s', datefmt='%m/%d/%Y %I:%M:%S %p',
                    filemode='w',encoding='utf-8', level=logging.INFO)
logging.info('Starting training')
 
logging.info(args)


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

optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
lin_blur_loss=LinearLoss()

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

    if 'optimize' in pretrained_dict:
        optimizer.load_state_dict(pretrained_dict['optimize'])

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
    
# =========== Train func. =========
def train(img_stack,gt_disp,blur,foc_dist):
    model.train()
    img_stack   = Variable(torch.FloatTensor(img_stack))
    gt_disp    = Variable(torch.FloatTensor(gt_disp))
    img_stack, gt_disp, blur,foc_dist = img_stack.cuda(),  gt_disp.cuda(), blur.cuda(),foc_dist.cuda()

    #---------
    # max_val = torch.where(foc_dist>=100, torch.zeros_like(foc_dist), foc_dist) # exclude padding value
    # min_val = torch.where(foc_dist<=0, torch.ones_like(foc_dist)*10, foc_dist)  # exclude padding value
    # mask = (gt_disp >= min_val.min(dim=1)[0].view(-1,1,1,1)) & (gt_disp <= max_val.max(dim=1)[0].view(-1,1,1,1)) #
    # mask.detach_()

    mask = gt_disp > 0
    mask.detach_()
    blur_mask=torch.repeat_interleave(mask,repeats=img_stack.shape[1],dim=1)
    #----
    optimizer.zero_grad()
    beta_scale = 1 # smooth l1 do not have beta in 1.6, so we increase the input to and then scale back -- no significant improve according to our trials
    cost3,s2_pred= model(img_stack, foc_dist)
    #cost3 shape: torch.Size([12, 6, 256, 256])
    #gt_disp shape : torch.Size([12, 1, 256, 256])
    #foc_dist shape : torch.Size([12, 6])
    s1=foc_dist[:,:-1]
    linloss=lin_blur_loss(s1,gt_disp,gt_disp,cost3[:,:-1,:,:])
    blurloss = F.smooth_l1_loss(cost3[blur_mask] * beta_scale, blur[blur_mask]* beta_scale, reduction='none').mean()
    dmask=torch.squeeze(mask,dim=1)
    depthloss = F.smooth_l1_loss(s2_pred[dmask] * beta_scale, gt_disp[:,0,:,:][dmask]* beta_scale, reduction='none').mean()
    loss=args.bweight*blurloss+depthloss
    torch.autograd.set_detect_anomaly(True) 
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    optimizer.step()
    blurlossvalue = blurloss.data
    linlossvalue = linloss.data
    depthlossvalue=depthloss.data
    return blurlossvalue,linlossvalue,depthlossvalue


def valid(img_stack,gt_disp,blur,foc_dist):
    model.eval()
    img_stack = Variable(torch.FloatTensor(img_stack))
    gt_disp = Variable(torch.FloatTensor(gt_disp))
    img_stack, gt_disp,foc_dist,blur = img_stack.cuda(),gt_disp.cuda(),foc_dist.cuda(),blur.cuda()
    gt_disp=torch.squeeze(gt_disp)
    #---------
    mask = gt_disp > 0
    mask.detach_()
    blur_mask=torch.repeat_interleave(torch.unsqueeze(mask,dim=1),repeats=img_stack.shape[1],dim=1)
    #----
    with torch.no_grad():
        cost3,pred_disp= model(img_stack, foc_dist)
        if args.model == 'LinDFF' or args.model == 'LinDFF1':
            pred_disp=torch.unsqueeze(pred_disp,dim=1)
        depth_loss = (F.mse_loss(pred_disp[mask] , gt_disp[mask] , reduction='mean')) # use MSE loss for val
        blur_loss = (F.mse_loss(cost3[blur_mask] , blur[blur_mask] , reduction='mean')) # use MSE loss for val

    return depth_loss,blur_loss



def adjust_learning_rate(optimizer, epoch):
    # turn out we do not need adjust lr, the results is already good enough
    if epoch <= args.epochs:
        lr = args.lr
    else:
        lr = args.lr * 0.1 #1e-5  will not used in this project
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr

def main():
    global  start_epoch, best_loss, total_iter
    saveName = args.logname + "_scale{}_nsck{}_lr{}_ep{}_b{}_lvl{}_model{}".format(args.FoD_scale,args.stack_num,
                                                                         args.lr, args.epochs, args.batchsize, args.level,args.model)
    if args.use_diff > 0:
        saveName = saveName + '_diffFeat{}'.format(args.use_diff)

    total_iters = total_iter

    for epoch in range(start_epoch, args.epochs+1):
        total_train_loss_blur,total_train_loss_lin,total_train_loss_depth = 0,0,0
        lr_ = adjust_learning_rate(optimizer,epoch)
        # train_log.scalar_summary('lr_epoch', lr_, epoch)

        ## training ##
        for batch_idx, sample_batch in enumerate(loaders[0]):
            img_stack=sample_batch['input'].float()
            gt_disp=sample_batch['output'][:,-1,:,:]
            gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
            foc_dist=sample_batch['fdist'].float()
            blur=sample_batch['blur'].float()

            start_time = time.time()
            blurlossvalue,linlossvalue,depthlossvalue = train(img_stack, gt_disp, blur,foc_dist)

            total_train_loss_blur += blurlossvalue
            total_train_loss_lin += linlossvalue
            total_train_loss_depth+=depthlossvalue
            total_iters += 1
        print('[train] blur loss=%2.5f linear loss=%2.5f depth loss=%2.5f' %(total_train_loss_blur/total_iters,total_train_loss_lin/total_iters,total_train_loss_depth/total_iters))
        logging.info('[train] blur loss=%2.5f linear loss=%2.5f depth loss=%2.5f' , total_train_loss_blur/total_iters,total_train_loss_lin/total_iters,total_train_loss_depth/total_iters)

        # save model
        torch.save({
            'epoch': epoch + 1,
            'iters': total_iters + 1,
            'best': best_loss,
            'state_dict': model.state_dict(),
            'optimize':optimizer.state_dict(),
        },  os.path.abspath(args.resultspth)+'/model_{}.tar'.format(epoch))

        # Vaild
        if epoch % 10 == 0:
            total_val_depth_loss,total_val_blur_loss = 0,0

            for batch_idx, sample_batch in enumerate(loaders[1]):
                img_stack=sample_batch['input'].float()
                gt_disp=sample_batch['output'][:,-1,:,:]
                gt_disp=torch.unsqueeze(gt_disp,dim=1).float()
                foc_dist=sample_batch['fdist'].float()
                blur=sample_batch['blur'].float()

                with torch.no_grad():
                    start_time = time.time()
                    val_depth_loss,val_blur_loss = valid(img_stack, gt_disp,blur,foc_dist)

                # if batch_idx %10 == 0:
                #     torch.cuda.synchronize()
                #     print('[val] epoch %d : %d/%d val_loss = %.6f , time = %.2f' % (epoch, batch_idx, len(loaders[1]), val_loss, time.time() - start_time))
                #     logging.info('[val] val_loss=%2.5f', val_loss)
                total_val_depth_loss += val_depth_loss
                total_val_blur_loss += val_blur_loss


            avg_val_depth_loss = total_val_depth_loss / len(loaders[1])
            avg_val_blur_loss = total_val_blur_loss / len(loaders[1])
            err_thres = 0.05 # for validation purpose
            # write_log(viz, img_stack[:, 0], img_stack[:, -1], gt_disp, val_log, epoch, thres=err_thres)
            # val_log.scalar_summary('avg_loss', avg_val_loss, epoch)
            print('[val] avg val depth loss %2.5f average val blur loss %2.5f' %(avg_val_depth_loss,avg_val_blur_loss))
            logging.info('[val] avg val depth loss=%2.5f average val blur loss %2.5f', avg_val_depth_loss,avg_val_blur_loss)

        #     # save best
        #     if avg_val_loss < best_loss:
        #         best_loss = avg_val_loss
        #         torch.save({
        #             'epoch': epoch + 1,
        #             'iters': total_iters + 1,
        #             'best': best_loss,
        #             'state_dict': model.state_dict(),
        #             'optimize': optimizer.state_dict(),
        #         },  os.path.abspath(args.savemodel) + '/' + saveName + '/best.tar')


        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()



#testing. delete

# model = DFFNet(clean=False,level=1, use_diff=args.use_diff)
# model = nn.DataParallel(model)
# model.cuda()


# a=torch.rand(6,5,3,224,224).cuda()
# f=torch.rand(6,5)
# out=model(a,f)



