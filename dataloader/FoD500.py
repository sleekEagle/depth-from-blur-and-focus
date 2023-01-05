#! /usr/bin/python3

import os
import numpy as np
from torch.utils.data import Dataset
import torchvision
import torch
from PIL import Image

from torchvision import transforms
import random
import numbers
import OpenEXR
from os import listdir, mkdir
from os.path import isfile, join, isdir
import cv2


# code adopted from https://github.com/soyers/ddff-pytorch/blob/master/python/ddff/dataproviders/datareaders/FocalStackDDFFH5Reader.py

def read_dpt(img_dpt_path):
    # pt = Imath.PixelType(Imath.PixelType.HALF)  # FLOAT HALF
    dpt_img = OpenEXR.InputFile(img_dpt_path)
    dw = dpt_img.header()['dataWindow']
    size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
    (r, g, b) = dpt_img.channels("RGB")
    dpt = np.fromstring(r, dtype=np.float16)
    dpt.shape = (size[1], size[0])
    return dpt

class ImageDataset(torch.utils.data.Dataset):
    """Focal place dataset."""

    def __init__(self, root_dir, img_list, dpth_list,  transform_fnc=None, flag_shuffle=False, data_ratio=0,
                 flag_inputs=[True, False], flag_outputs=[False, True], focus_dist=[0.1,.15,.3,0.7,1.5], focus_dist_req=[0.1,.15,.3,0.7,1.5], scale=1):
        self.root_dir = root_dir
        self.transform_fnc = transform_fnc
        self.flag_shuffle = flag_shuffle

        self.flag_rgb = flag_inputs[0]
        self.flag_coc = flag_inputs[1]

        self.data_ratio = data_ratio

        self.flag_out_coc = flag_outputs[0]
        self.flag_out_depth = flag_outputs[1]

        self.focus_dist = focus_dist #torch.tensor(focus_dist) / scale
        self.focus_dist_req = focus_dist_req #torch.tensor(focus_dist) / scale
        self.dpth_scale = scale


        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape([1,1,3])#[0.278, 0.250, 0.227]
        self.img_std= np.array([0.229, 0.224, 0.225]).reshape([1,1,3])#[0.185, 0.178, 0.178]

        self.guassian_kernel =  (35,35) # large blur kernel for pad image

        ##### Load all images
        self.imglist_all = img_list
        self.imglist_dpt = dpth_list


    def __len__(self):
        return int(len(self.imglist_dpt))

    def dpth2disp(self, dpth):
        disp = 1 / dpth
        disp[dpth==0] = 0
        return disp

    def __getitem__(self, idx):
        ##### Read and process an image
        idx_dpt = int(idx)
        img_dpt = read_dpt(self.root_dir + self.imglist_dpt[idx_dpt])
        print(np.max(img_dpt))

        foc_dist = self.focus_dist.copy()
        mat_dpt = img_dpt.copy()[:, :, np.newaxis]

        ind = idx * len(foc_dist)
        num_list = list(range(len(foc_dist)))

        # add RGB, CoC, Depth inputs
        mats_input = []
        mats_output = np.zeros((256, 256, 0))

        # load existing image
        for i in range(len(foc_dist)):
            if self.flag_rgb:
                im = Image.open(self.root_dir + self.imglist_all[ind + num_list[i]])
                img_all = np.array(im)
                # img Norm
                mat_all = img_all.copy() / 255.
                mat_all = (mat_all - self.img_mean) / self.img_std
                mats_input.append(mat_all)
        
        mats_input = np.stack(mats_input)
        
        #filter out the required focal lengths
        avail_focus_dist=[i for i in foc_dist if (i in self.focus_dist_req)]
        valididx=np.argwhere([(i in self.focus_dist_req) for i in foc_dist])
        mats_input_selected=np.take(mats_input,valididx[:,0],axis=0) 

        if self.flag_out_depth:
            mats_output = np.concatenate((mats_output,(mat_dpt) * self.dpth_scale), axis=2) # first 5 is COC last is depth  self.dpth2disp

        sample = {'input': mats_input_selected, 'output': mats_output}

        if self.transform_fnc:
            sample = self.transform_fnc(sample)

        return sample['input'], sample['output'],(torch.tensor(avail_focus_dist)) * self.dpth_scale # to match the scale of DDFF12  self.dpth2disp

class ToTensor(object):
    def __call__(self, sample):
        mats_input, mats_output = sample['input'], sample['output']

        mats_input = mats_input.transpose((0, 3, 1, 2))
        mats_output = mats_output.transpose((2, 0, 1))
        return {'input': torch.from_numpy(mats_input).float(),
                'output': torch.from_numpy(mats_output).float()}


class RandomCrop(object):
    """ Randomly crop images
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        inputs, target = sample['input'], sample['output']
        n, h, w, _ = inputs.shape
        th, tw = self.size
        if w < tw: tw=w
        if h < th: th=h

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        inputs = inputs[:, y1: y1 + th,x1: x1 + tw]
        return {'input':inputs,
                'output':target[y1: y1 + th,x1: x1 + tw]}


class RandomFilp(object):
    """ Randomly crop images
    """

    def __init__(self, ratio=0.5):
        self.ratio = ratio

    def __call__(self, sample):
        inputs, target = sample['input'], sample['output']

        # hori filp
        if np.random.binomial(1, self.ratio):
            inputs = inputs[:,:, ::-1]
            target = target[:,::-1]

        # vert flip
        if np.random.binomial(1, self.ratio):
            inputs = inputs[:, ::-1]
            target = target[::-1]

        return {'input': np.ascontiguousarray(inputs), 'output': np.ascontiguousarray(target)}

def FoD500Loader(data_dir, scale=1, focus_dist=[0.1, 0.15, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, 10.0, float('inf')],focus_dist_req=[0.1,.15,.3,0.7,1.5]):

    img_train_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "All.tif" and int(f[:6]) < 200]
    dpth_train_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) < 200]

    img_train_list.sort()
    dpth_train_list.sort()

    img_val_list =  [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "All.tif" and int(f[:6]) >= 200]
    dpth_val_list = [f for f in listdir(data_dir) if isfile(join(data_dir, f)) and f[-7:] == "Dpt.exr" and int(f[:6]) >=200]

    img_val_list.sort()
    dpth_val_list.sort()

    train_transform = transforms.Compose([
                        RandomCrop(224),
                        RandomFilp(0.5),
                        ToTensor()])
    dataset_train = ImageDataset(root_dir=data_dir, img_list=img_train_list, dpth_list=dpth_train_list,
                                 transform_fnc=train_transform,  focus_dist=focus_dist,
                                 focus_dist_req=focus_dist_req, scale=scale)

    val_transform = transforms.Compose([ToTensor()])
    dataset_valid = ImageDataset(root_dir=data_dir, img_list=img_val_list, dpth_list=dpth_val_list,
                                 transform_fnc=val_transform, focus_dist=focus_dist, 
                                 focus_dist_req=focus_dist_req,scale=scale)


    return dataset_train, dataset_valid


