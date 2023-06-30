import matplotlib.pyplot as plt
import numpy as np
import torch

#simulate blur
def get_sim_blur(s1_start=0,s1_end=2,n=100,noise_std=1,s2=1.3):
    s1 = np.linspace(s1_start,s1_end,n)
    L=np.abs(s2-s1)
    noise = np.random.normal(0,noise_std,n)
    #add noise
    L+=noise
    return s1,L

def get_s2est(s1,L):
    #get the first estimates for s2
    s1est1=s1-L # for the +45 degrees line
    s1est2=s1+L # for the -45 degrees line

    vals=[]
    for i in range(len(s1est1)+1):
        tmp=[]
        for j in range(len(s1est1)):
            if(j>=i):
                tmp.append(s1est1[j])
            else:
                tmp.append(s1est2[j])
        vals.append(tmp)


    valstds=[]
    for i in range(len(vals)):
        valstds.append(np.std(vals[i]))
    minstdarg=np.argmin(valstds)
    S_estlist=vals[minstdarg]
    S_est=np.mean(S_estlist)

    return S_est

#check if the above functions can solve for s2 
s2=0.6
s1,L=get_sim_blur(0.1,2.7,5,0.001,s2)

#plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(s1,L,marker='o',color='r')
plt.show()

S_est=get_s2est(s1,L)
print(S_est)

#do the above same thing with pytorch
import torch

s2=0.3
s1,L=get_sim_blur(100,104,5,0.1,s2)

#plot the data
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(s1,L,marker='o',color='r')
plt.show()

s1=torch.from_numpy(s1)
L=torch.from_numpy(L)
s1est1=s1-L # for the +45 degrees line
s1est1=s1est1.unsqueeze(0)
s1est1=torch.repeat_interleave(s1est1,s1.shape[0]+1,dim=0)
s1est2=s1+L # for the -45 degrees line
s1est2=s1est2.unsqueeze(0)
s1est2=torch.repeat_interleave(s1est2,s1.shape[0]+1,dim=0)

mask=torch.zeros(2,s1.shape[0]+1,s1.shape[0])
for i in range(1,mask.shape[1]):
    mask[0,i,:i]=1
for i in range(0,mask.shape[1]):
    mask[1,i,i:]=1

s1est=s1est1*mask[1]+s1est2*mask[0]
s1eststd=torch.std(s1est,dim=1)
s2est=torch.mean(s1est[torch.argmin(s1eststd),:])
print(s2est)


#extend this to a 3d array of blurs (a batch of 2d images)
def get_sim_blur_3d(s1_start=0,s1_end=2,n=5,bs=4,imgsize=24,noise_std=0.1,s2mean=3):
    s1=np.linspace(s1_start,s1_end,n)
    s1=torch.tensor(s1)
    s1=torch.unsqueeze(s1,dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
    s1=torch.repeat_interleave(s1,bs,0).repeat_interleave(imgsize,2).repeat_interleave(imgsize,3)

    s2 = torch.tensor(np.random.rand(bs,imgsize,imgsize))*s2mean
    s2=torch.unsqueeze(s2,dim=1).repeat_interleave(n,1)

    L=torch.abs(s2-s1)
    L=L+(noise_std**0.5)*torch.randn(bs,n,imgsize,imgsize)
    return s1,L,s2

bs=4
n=5
imgsize=24
s2mean=200
s1,L,s2=get_sim_blur_3d(bs=bs,n=n,imgsize=imgsize,noise_std=0.00001,s2mean=s2mean)

def regresss2(s1,blur):
    #for +45 degrees lines
    s1est1=s1-blur
    s1est1=s1est1.unsqueeze(dim=1)
    s1est1=torch.repeat_interleave(s1est1,s1.shape[1]+1,dim=1)

    #for -45 degrees lines
    s1est2=s1+blur
    s1est2=s1est2.unsqueeze(dim=1)
    s1est2=torch.repeat_interleave(s1est2,s1.shape[1]+1,dim=1)

    mask=torch.zeros(2,bs,n+1,n,imgsize,imgsize)
    for i in range(1,n+1):
        mask[0,:,i,:i,:,:]=1
    for i in range(0,n+1):
        mask[1,:,i,i:,:]=1
    s1est=s1est1*mask[1]+s1est2*mask[0]

    s1eststd=torch.std(s1est,dim=2)
    s1eststd[0,:,20,0]

    argmin=torch.argmin(s1eststd,dim=1)

    argmin=torch.unsqueeze(argmin,dim=1).unsqueeze(dim=2)
    argmin=torch.repeat_interleave(argmin,repeats=n,dim=2)

    sel=torch.gather(s1est,dim=1,index=argmin)
    s2_pred=torch.mean(sel,dim=2)[:,0,:,:]
    return s2_pred


s2pred=regresss2(s1,L)

torch.mean(torch.abs(s2pred-s2[:,0,:,:]))


#make this into a nn module
import torch.nn as nn

class distregression(nn.Module):
    def __init__(self):
        super(distregression, self).__init__()

    def forward(self, s1,blur):
        #for +45 degrees lines
        s1est1=s1-blur
        s1est1=s1est1.unsqueeze(dim=1)
        s1est1=torch.repeat_interleave(s1est1,s1.shape[1]+1,dim=1)
        imgsize=s1.shape[-1]
        bs=s1.shape[0]
        n=s1.shape[1]

        #for -45 degrees lines
        s1est2=s1+blur
        s1est2=s1est2.unsqueeze(dim=1)
        s1est2=torch.repeat_interleave(s1est2,s1.shape[1]+1,dim=1)

        mask=torch.zeros(2,bs,n+1,n,imgsize,imgsize)
        for i in range(1,n+1):
            mask[0,:,i,:i,:,:]=1
        for i in range(0,n+1):
            mask[1,:,i,i:,:]=1
        s1est=s1est1*mask[1]+s1est2*mask[0]

        s1eststd=torch.std(s1est,dim=2)
        s1eststd[0,:,20,0]

        argmin=torch.argmin(s1eststd,dim=1)

        argmin=torch.unsqueeze(argmin,dim=1).unsqueeze(dim=2)
        argmin=torch.repeat_interleave(argmin,repeats=n,dim=2)

        sel=torch.gather(s1est,dim=1,index=argmin)
        s2_pred=torch.mean(sel,dim=2)[:,0,:,:]
        return s2_pred

distreg=distregression()
s2pred=distreg(s1,L)
torch.mean(torch.abs(s2pred-s2[:,0,:,:]))

#test the module in models/submodule - using cuda
#start from a fresh terminal
from models.submodule import *

bs=4
n=5
imgsize=24
s2mean=200
s1,L,s2=get_sim_blur_3d(bs=bs,n=n,imgsize=imgsize,noise_std=0.00001,s2mean=s2mean)
s1=s1.cuda()
L=L.cuda()
s2=s2.cuda()
distreg=distregression()
s2pred=distreg(s1,L)
torch.mean(torch.abs(s2pred-s2[:,0,:,:]))




'''
Testing creating loss function that mesures the accuracy of outputting linear blur
a.k.a linear blur estimation
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

def get_s1blur_data(s2,n):
    s1_=np.array(list(range(1,30,1)))/10
    s1_arg1_=np.argwhere((s1_<s2))[:,0]
    s1_arg2_=np.argwhere((s1_>=s2))[:,0]
    s1_arg1=random.sample(list(s1_arg1_),n[0])
    s1_arg2=random.sample(list(s1_arg2_),n[1])
    s1_arg1.sort()
    s1_arg2.sort()

    blur1,blur2=[],[]
    for val in s1_arg1:
        x=s1_[val]
        y=s2-x
        blur1.append(y)
    for val in s1_arg2:
        x=s1_[val]
        y=x-s2
        blur2.append(y)

    s1=np.concatenate((s1_[s1_arg1],s1_[s1_arg2]))
    blurs=np.concatenate((blur1,blur2))

    #add noise
    noise_std=0.1
    noise = np.random.normal(0,noise_std,blurs.shape[0])
    blurs_noise=blurs+noise
    
    plt.plot(s1 ,blurs, marker="o", markersize=9)
    plt.plot(s1 ,blurs_noise, marker="x", markersize=9)
    plt.show()

    return s1,blurs_noise,blurs

s2=1.3
#nuber of points on the left and right side of s2 point
n=[3,5]
s1,blurs_noise,blurs=get_s1blur_data(s2,n)
s1=torch.tensor(s1)
blurs_noise=torch.tensor(blurs_noise)
s2=torch.tensor([s2])
mask=(s1<s2).int()
mask[mask==0]=-1
gt_blur=(s2-s1)*mask


#generate 3D blur data
def get_3D_blur_data(n=[3,5],noise_std=0.001):
    s2=torch.rand((10,20,20))*3
    s1=[0.1,0.2,0.4,1.5,2.3,2.9,3.6]
    s1=torch.tensor(s1)
    s1_=torch.unsqueeze(s1,dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
    s1_=torch.repeat_interleave(s1_,repeats=s2.shape[0],dim=0)
    s1_=torch.repeat_interleave(s1_,repeats=s2.shape[1],dim=2)
    s1_=torch.repeat_interleave(s1_,repeats=s2.shape[2],dim=3)
    s2_=torch.unsqueeze(s2,dim=1)
    s2_=torch.repeat_interleave(s2_,repeats=s1_.shape[1],dim=1)

    mask=(s1_<s2_).int()
    mask[mask==0]=-1
    GT_blur=(s2_-s1_)*mask

    #create noise
    noise_blur=GT_blur+torch.randn_like(GT_blur)*(noise_std**0.5)

    s1_tmp=s1_[0,:,0,0].numpy()
    b_real_tmp=GT_blur[0,:,0,0].numpy()
    b_noise_tmp=noise_blur[0,:,0,0].numpy()

    plt.plot(s1_tmp ,b_real_tmp, marker="o", markersize=9)
    plt.plot(s1_tmp ,b_noise_tmp, marker="x", markersize=9)
    plt.show()

    return s1,noise_blur,GT_blur

criterion = torch.nn.MSELoss()


s1,blur_est,GT_blur=get_3D_blur_data(noise_std=0.004)
s2_est=torch.rand((10,20,20))*3

def get_loss(s1,s2_est,s2_gt,blur_est):
    s1_=torch.unsqueeze(s1,dim=0).unsqueeze(dim=2).unsqueeze(dim=3)
    s1_=torch.repeat_interleave(s1_,repeats=s2_est.shape[0],dim=0)
    s1_=torch.repeat_interleave(s1_,repeats=s2_est.shape[1],dim=2)
    s1_=torch.repeat_interleave(s1_,repeats=s2_est.shape[2],dim=3)
    s2_=torch.unsqueeze(s2_est,dim=1)
    s2_=torch.repeat_interleave(s2_,repeats=s1_.shape[1],dim=1)

    mask=(s1_<s2_).int()
    mask[mask==0]=-1
    GT_blur=(s2_-s1_)*mask

    loss=criterion(GT_blur,blur_est)
    return loss

get_loss(s1,s2_est,s2_est,blur_est)




















        























































