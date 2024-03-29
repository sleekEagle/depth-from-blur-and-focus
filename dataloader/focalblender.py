import torch
import torch.nn as nn
import torch.utils.data
from torchvision import transforms, utils
from os import listdir
from os.path import isfile, join
import numpy as np
import random
import OpenEXR
from PIL import Image

# reading depth files
def read_dpt(img_dpt_path): 
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.frombuffer(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt

'''
output |s2-s1|
'''
def get_blur(s1,s2,f):
    if(s1<50):
        blur=abs(s2-s1)
    else:
        blur=1/s2
    return blur

def get_blur_ratio(s1,s2,f):
    blur=np.abs(s1-s2)/(s1-f)
    return blur

#blur at infinity depends only on s2 and the camera parameters (which we ignore due to normalization)
def get_inf_blur(s2):
    blur=1/s2
    return blur

# s1=[0.1,0.15,0.3,0.7,1.5]
# s2=[0.1,0.15,0.3,0.7,1.5,1.83]
# f=2.9e-3
# b=[]
# for s1_ in s1:
#     for s2_ in s2:
#         b.append(abs(s2_-s1_)/(s1_-f))
# min(b),max(b)

'''
All in-focus image is attached to the input matrix after the RGB image

focus_dist - available focal dists in the dataset
req_f_indx - a list of focal dists we require.

fstack=0
a single image from the focal stack will be returned.
The image will be randomly selected from indices in req_f_indx
fstack=1
several images comprising of the focal stack will be returned. 
indices of the focal distances will be selected from req_f_indx

if aif=1 : 
    all in focus image will be appended at the begining of the focal stack

input matrix channles :
[batch,image,rgb_channel,256,256]

if aif=1 and fstack=1
image 0 : all in focus image
image 1:-1 : focal stack
if aif=0
image 0:-1 : focal stack

if fstack=0

output: [depth, blur1, blur2,]
blur1,blur2... corresponds to the focal stack
'''

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, root_dir, transform_fnc=None,blur=1,aif=0,fstack=0,focus_dist=[0.1,.15,.3,0.7,1.5,-1],
                req_f_indx=[0,2], max_dpt = 3.):

        self.root_dir = root_dir
        print("image data root dir : " +str(self.root_dir))
        self.transform_fnc = transform_fnc

        self.blur=blur
        self.aif=aif
        self.fstack=fstack
        self.img_num = len(focus_dist)
        
        assert focus_dist[-1]==-1 , "the last entry of the focus distances should be -1 (to mean infinity)"

        self.focus_dist = focus_dist
        self.req_f_idx=req_f_indx

        ##### Load and sort all images
        self.imglist_all = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "All.tif"]
        self.imglist_dpt = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "Dpt.exr"]
        self.imglist_allif = [f for f in listdir(root_dir) if isfile(join(root_dir, f)) and f[-7:] == "Aif.tif"]

        print("Total number of samples", len(self.imglist_dpt), "  Total number of seqs", len(self.imglist_dpt) / self.img_num)

        self.imglist_all.sort()
        self.imglist_dpt.sort()
        self.imglist_allif.sort()

        self.max_dpt = max_dpt
        self.f=2.9e-3

    def __len__(self):
        return int(len(self.imglist_dpt))

    def __getitem__(self, idx):
        if(self.fstack):
            #if stack is needed return all fdists from req_f_idx
            reqar=self.req_f_idx
        else:
            reqar=[random.choice(self.req_f_idx)]

        # add RGB, CoC, Depth inputs
        mats_input = np.zeros((256, 256, 3,0))
        mats_output = np.zeros((256, 256, 0))
        mats_blur = np.zeros((256, 256, 0))

        ##### Read and process depth image
        idx_dpt = int(idx)
        img_dpt = read_dpt(self.root_dir + self.imglist_dpt[idx_dpt])
        mat_dpt_scaled = img_dpt/self.max_dpt
        mat_dpt = mat_dpt_scaled.copy()[:, :, np.newaxis]

        ind = idx * self.img_num

        #if all in-focus image is also needed append that to the input matrix
        if self.aif:
            im = Image.open(self.root_dir + self.imglist_allif[idx])
            img_all = np.array(im)
            mat_all = img_all.copy() / 255.
            mat_all=np.expand_dims(mat_all,axis=-1)
            mats_input = np.concatenate((mats_input, mat_all), axis=3)
        fdist=np.zeros((0))
        for req in reqar:
            im = Image.open(self.root_dir + self.imglist_all[ind + req])
            img_all = np.array(im)
            mat_all = img_all.copy() / 255.
            mat_all=np.expand_dims(mat_all,axis=-1)
            mats_input = np.concatenate((mats_input, mat_all), axis=3)
            if(not self.focus_dist[req]==-1):
                img_msk = get_blur_ratio(self.focus_dist[req], img_dpt,self.f)
            else:
                img_msk = get_inf_blur(img_dpt)
            mat_msk = img_msk.copy()[:, :, np.newaxis]
            #append blur to the output
            mats_blur = np.concatenate((mats_blur, mat_msk), axis=2)
            fdist=np.concatenate((fdist,[self.focus_dist[req]]),axis=0)
        
        #append depth to the output
        mats_output = np.concatenate((mats_output, mat_dpt), axis=2)
        
        sample = {'input': mats_input, 'output': mats_output,'blur':mats_blur}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)
        sample = {'input': sample['input'], 'output': sample['output'],'blur':sample['blur'],'fdist':fdist}
        return sample


class ToTensor(object):
    def __call__(self, sample):
        mats_input, mats_output,mats_blur = sample['input'], sample['output'],sample['blur']

        mats_input=mats_input.transpose((3,2, 0, 1))
        mats_output=mats_output.transpose((2, 0, 1))
        mats_blur=mats_blur.transpose((2, 0, 1))
        return {'input': torch.from_numpy(mats_input),
                'output': torch.from_numpy(mats_output),
                'blur':torch.from_numpy(mats_blur)}

def load_data(data_dir,aif,train_split,fstack,
              WORKERS_NUM, BATCH_SIZE, FOCUS_DIST, REQ_F_IDX, MAX_DPT):
    img_dataset = ImageDataset(root_dir=data_dir,aif=aif,transform_fnc=transforms.Compose([ToTensor()]),
                               focus_dist=FOCUS_DIST,fstack=fstack,req_f_indx=REQ_F_IDX, max_dpt=MAX_DPT)

    indices = list(range(len(img_dataset)))
    split = int(len(img_dataset) * train_split)

    indices_train = indices[:split]
    indices_valid = indices[split:]

    dataset_train = torch.utils.data.Subset(img_dataset, indices_train)
    dataset_valid = torch.utils.data.Subset(img_dataset, indices_valid)

    loader_train = torch.utils.data.DataLoader(dataset=dataset_train, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=True)
    loader_valid = torch.utils.data.DataLoader(dataset=dataset_valid, num_workers=WORKERS_NUM, batch_size=BATCH_SIZE, shuffle=False)

    total_steps = int(len(dataset_train) / BATCH_SIZE)
    print("Total number of steps per epoch:", total_steps)
    print("Total number of training sample:", len(dataset_train))
    print("Total number of validataion sample:", len(dataset_valid))

    return [loader_train, loader_valid], total_steps

def get_data_stats(datapath):
    loaders, total_steps = load_data(datapath,aif=0,train_split=0.8,fstack=1,WORKERS_NUM=0,
        BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,-1],REQ_F_IDX=[0,1,2,3,4,5],MAX_DPT=1.0)
    print('stats of train data')
    get_loader_stats(loaders[0])
    print('______')

#data statistics of the input images
def get_loader_stats(loader):
    xmin,xmax,xmean,count=100,0,0,0
    depthmin,depthmax,depthmean=100,0,0
    blurmin,blurmax,blurmean=100,0,0
    for st_iter, sample_batch in enumerate(loader):
        # Setting up input and output data
        X = sample_batch['input'][:,0,:,:,:].float()
        Y = sample_batch['blur'].float()
        D = sample_batch['output'].float()

        xmin_=torch.min(X).cpu().item()
        if(xmin_<xmin):
            xmin=xmin_
        xmax_=torch.max(X).cpu().item()
        if(xmax_>xmax):
            xmax=xmax_
        xmean+=torch.mean(X).cpu().item()
        count+=1
    
        #blur (|s2-s1|/(s2*(s1-f)))
        gt_step1 = Y[:, :-1, :, :]
        #depth in m
        gt_step2 = D[:, -1:, :, :]

        depthmin_=torch.min(gt_step2).cpu().item()
        if(depthmin_<depthmin):
            depthmin=depthmin_
        depthmax_=torch.max(gt_step2).cpu().item()
        if(depthmax_>depthmax):
            depthmax=depthmax_
        depthmean+=torch.mean(gt_step2).cpu().item()

        blurmin_=torch.min(gt_step1).cpu().item()
        if(blurmin_<blurmin):
            blurmin=blurmin_
        blurmax_=torch.max(gt_step1).cpu().item()
        if(blurmax_>blurmax):
            blurmax=blurmax_
        blurmean+=torch.mean(gt_step1).cpu().item()

    print('X min='+str(xmin))
    print('X max='+str(xmax))
    print('X mean='+str(xmean/count))

    print('depth min='+str(depthmin))
    print('depth max='+str(depthmax))
    print('depth mean='+str(depthmean/count))

    print('blur min='+str(blurmin))
    print('blur max='+str(blurmax))
    print('blur mean='+str(blurmean/count))

'''
blur_thres=7.0
p=3.1/256*1e-3 # pixel width in m
N=2
f=6e-3
s2range=[0.1,2.0]
s1range=[0.1,1.5]

get_workable_s1s2ranges(p,N,f,s2range,s1range,blur_thres)
'''

# def plot_blur_ratio(blur,s1,s2,i,j):
#     b=blur[0,:,i,j].numpy()
#     d=s2[0,0,i,j].numpy().item()
#     plt.plot(fdist,b, marker="o", markersize=9, markeredgecolor="red")
#     plt.plot([d],[0], marker="o", markersize=9, markeredgecolor="red")
#     plt.show()

#save n number of relative blur plots
def save_blur_ratio(blur,fdist,s2,n,savepath):
    import matplotlib.pyplot as plt
    np.random.seed(42)
    for idx in range(n):
        i,j=(np.random.random(size=2)*s2.shape[-1]).tolist()
        i,j=int(i),int(j)
        b=blur[0,:,i,j].numpy()*(fdist-2.9e-3)
        d=s2[0,0,i,j].numpy().item()
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(fdist,b, marker="o", markersize=9, markeredgecolor="red")
        ax.plot([d],[0], marker="o", markersize=9, markeredgecolor="red")
        ax.set_xlabel("fdist")
        ax.set_ylabel("|s1-s2|")
        fig.savefig(join(savepath,str(idx)+'.jpg'))

#plot blur ratio

# datapath='C:\\Users\\lahir\\focalstacks\\datasets\\mediumN1\\'
# loaders, total_steps = load_data(datapath,aif=0,train_split=0.8,fstack=1,WORKERS_NUM=0,
#         BATCH_SIZE=1,FOCUS_DIST=[0.1,.15,.3,0.7,1.5,-1],REQ_F_IDX=[0,1,2,3,4,5],MAX_DPT=1.0)
# for st_iter, sample_batch in enumerate(loaders[0]):
#     # Setting up input and output data
#     X = sample_batch['input'].float()
#     Y = sample_batch['blur'].float()
#     D = sample_batch['output'].float()
#     s1=sample_batch['fdist']
#     fdist=s1.numpy()[0,:]
#     break

# Y[0,-1,:,:]

# invs2=1/D

# torch.mean(invs2-Y[0,-1,:,:])

# savepath='C:\\Users\\lahir\\data\\lindefblur\\realative_blur_figures\\'
# save_blur_ratio(Y,fdist,D,100,savepath)










