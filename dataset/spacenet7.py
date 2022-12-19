from typing import List, Tuple
from collections import Sized
import os
import glob
import random
from os.path import join
import albumentations as alb
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import Normalize

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch import Tensor
from torch.nn import Upsample

import yaml
import errno

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)

    return config

cfg = load_config("./config.yaml")

def seg2patches(img, data_config=cfg['data_config']):
    dc      = data_config
    h, w    = dc['val_img_shape']
    p_size  = dc['patch_size']
    ov_size = dc['overlap_size']

    mask    = np.zeros((h, w))
    patches = []

    if ov_size == 0:
        mask[mask == 0] = 1

        nomi   = h
        denomi = p_size
        if nomi % denomi != 0:
            c_pnum = int(nomi/denomi)+1
        else:
            c_pnum = int(nomi/denomi)
    else:
        nomi    = h - ov_size
        denomi  = p_size - ov_size

        if nomi%denomi != 0:
            c_pnum = int(nomi/denomi)+1
        else:
            c_pnum = int(nomi/denomi)

    for h_idx in range(c_pnum):
        for w_idx in range(c_pnum):
            if (h_idx != c_pnum-1) or (w_idx != c_pnum-1):
                patch = img[h_idx * p_size - h_idx * ov_size : (h_idx+1) * p_size - h_idx * ov_size,
                            w_idx * p_size - w_idx * ov_size : (w_idx+1) * p_size - w_idx * ov_size,
                            :]
                mask[h_idx * p_size - h_idx * ov_size : (h_idx+1) * p_size - h_idx * ov_size,
                     w_idx * p_size - w_idx * ov_size : (w_idx+1) * p_size - w_idx * ov_size] += 1
            else:
                if nomi % denomi == 0:
                    patch = img[h_idx * p_size - h_idx * ov_size : (h_idx+1) * p_size - h_idx * ov_size,
                                w_idx * p_size - w_idx * ov_size : (w_idx+1) * p_size - w_idx * ov_size,
                                :]
                    mask[h_idx * p_size - h_idx * ov_size : (h_idx+1) * p_size - h_idx * ov_size,
                         w_idx * p_size - w_idx * ov_size : (w_idx+1) * p_size - w_idx * ov_size] += 1
                else:
                    h_max = (h_idx+1) * p_size - h_idx * ov_size
                    w_max = (w_idx+1) * p_size - w_idx * ov_size

                    if h_idx == c_pnum - 1:
                        h_max = h
                    if w_idx == c_pnum - 1:
                        w_max = w

                    patch = img[h_idx * p_size - h_idx * ov_size : h_max,
                                w_idx * p_size - w_idx * ov_size : w_max,
                                :]
                    mask[h_idx * p_size - h_idx * ov_size : h_max,
                         w_idx * p_size - w_idx * ov_size : w_max] += 1

            patches.append(patch)
    return patches, mask

def _augmentation(full_scene=True, mode='train', data_config=cfg['data_config']):
    dc = data_config
    pr = 0.5
    if full_scene is True:
        if mode == 'train':
            aug = alb.Compose([
                alb.HorizontalFlip(p=pr),
                alb.VerticalFlip(p=pr),
                alb.RandomRotate90(p=pr),
                alb.Normalize(mean=dc['mean'], std=dc['std']),
                ToTensorV2(),
                ])
            #elif mode == 'val':
        else:
            aug = alb.Compose([
                alb.Normalize(mean=dc['mean'], std=dc['std']),
                ToTensorV2(),
                ])
    else:
        if mode == 'train':
            aug = alb.Compose([
                alb.RandomCrop(dc['patch_size'], dc['patch_size']),
                alb.Resize(dc['patch_size'] * dc['resize_scale'], dc['patch_size'] * dc['resize_scale']),
                alb.HorizontalFlip(p=pr),
                alb.VerticalFlip(p=pr),
                alb.RandomRotate90(p=pr),
                alb.Normalize(mean=dc['mean'], std=dc['std']),
                ToTensorV2(),
                ])
            #elif mode == 'val':
        else:
            aug = alb.Compose([
                alb.Normalize(mean=dc['mean'], std=dc['std']),
                ToTensorV2(),
                ])

    return aug

def load_data(pre_path=None,
              pos_path=None,
              stat_path=None,
              phase='train',
              data_config=cfg['data_config']):
    dc = data_config
    img_shape = dc['train_img_shape']
    preimg = Image.open(pre_path).resize(img_shape)
    posimg = Image.open(pos_path).resize(img_shape)

    if phase == 'inference':
        statimg = None
        if stat_path is not None:
            statimg = Image.open(stat_path).resize(dc['train_img_shape'])
        return preimg, posimg, statimg
    
    else:
        preseg_path = pre_path.replace('images', 'labels').split('.p')[0]+'_Buildings.png'
        posseg_path = pos_path.replace('images', 'labels').split('.p')[0]+'_Buildings.png'
        
        preseg = Image.open(preseg_path).convert('L').resize(dc['train_img_shape'])
        posseg = Image.open(posseg_path).convert('L').resize(dc['train_img_shape'])
        
        if max(np.unique(preseg)) == 255:
            preseg = np.asarray(preseg)
            posseg = np.asarray(posseg)
            preseg = preseg // 255.0
            posseg = posseg // 255.0

        cd_gt = np.logical_xor(preseg, posseg)
        cd_gt = cd_gt.astype(np.uint8)
        
        statimg = None
        if stat_path is not None:
            statimg = Image.open(stat_path).resize(dc['train_img_shape'])
        
        return preimg, posimg, preseg, posseg, cd_gt, statimg

class SPN7Loader(Dataset):
    def __init__(self,
            data_path,
            phase,
            full_scene=True,
            data_config=cfg['data_config']):

        self.fs     = full_scene
        self.phase  = phase
        self.dc     = data_config
        self.root   = data_path
        self.img_dir_path   = os.path.join(self.root, self.phase, "images")
        self.label_dir_path = os.path.join(self.root, self.phase, "labels")

        self.subimg_dirlist = os.listdir(self.img_dir_path)

        if self.phase != 'inference':
            self.subgt_dirlist = os.listdir(self.label_dir_path)
            self.num_samples = len(glob.glob(os.path.join(self.img_dir_path, '**/*.png'), recursive=True))

        else:
            self.num_samples = len(os.listdir(img_dir_path))

        self.aug = _augmentation(self.fs, self.phase)
        self.totensor = alb.Compose([ToTensorV2(),])
    def __getitem__(self, idx):
        #### data path setting
        if self.phase != 'inference':
            diridx      = idx % len(self.subimg_dirlist)
            img_root    = os.path.join(self.img_dir_path, self.subimg_dirlist[diridx])

            pre_name, pos_name = random.sample(sorted(os.listdir(img_root))[:-3], 2)
            print('pre : {}/post : {}'.format(pre_name, pos_name))

            pre_img_path = img_root + '/' + pre_name
            pos_img_path = img_root + '/' + pos_name

        else:
            img_root    = os.path.join(self.img_dir_path, self.subimg_dirlist[idx])
            img_list    = os.listdir(img_root)
            pre_img_path = img_root + '/' + img_list[0]
            pos_img_path = img_root + '/' + img_list[1]
            sv_pre_name = 'Planet_20210322_' + self.subimg_dirlist[idx]
            sv_pos_name = 'Planet_20220409_' + self.subimg_dirlist[idx]
        
        stat_path = None
        if self.dc['transductive'] is True:
            stat_path = img_root + '/std_edamap.png'
        
        #### data load
        if self.phase != 'inference':
            preimg, posimg, preseg, posseg, cd_gt, statimg = load_data(pre_img_path,
                                                                        pos_img_path,
                                                                        stat_path,
                                                                        self.phase)
            if statimg is None:
                data_total = np.concatenate((preimg, posimg), axis=2)
            else:
                data_total = np.concatenate((preimg, posimg, statimg), axis=2)

        else:
            preimg, posimg, statimg = load_data(pre_img_path,
                                                pos_img_path,
                                                stat_path,
                                                self.phase)
            
            if statimg is None:
                data_total = np.concatenate((preimg, posimg), axis=2)
            else:
                data_total = np.concatenate((preimg, posimg, statimg), axis=2)

        #### inference mode patch processing
        if self.phase != 'train':
            patches, ov_guidance = seg2patches(data_total)
        
        #### Garbage value settings for dataloading
        sta_patches = np.zeros((1,1,1))
        #### data augmentation and wrap it to tensor and return
        if self.phase != 'train':
            pre_patches = []
            pos_patches = []
            sta_patches = []

            for patch in patches:
                tf_data = self.aug(image=patch)  
                
                pre_p, pos_p = tf_data['image'][0:3,:,:], tf_data['image'][3:6,:,:]
                pre_patches.append(pre_p)
                pos_patches.append(pos_p)
                
                if self.dc['transductive'] is True:
                    sta_p = tf_data['image'][6:9,:,:]
                    sta_patches.append(sta_p)
            
            if self.phase == 'val':
                preseg = np.asarray(preseg)
                preseg = self.totensor(image=preseg)
                preseg = preseg['image']
                posseg = np.asarray(posseg)
                posseg = self.totensor(image=posseg)
                posseg = posseg['image']
                cd_gt = self.totensor(image=cd_gt)
                cd_gt = cd_gt['image']
                
                return {'pre': pre_patches, 
                        'pos': pos_patches,
                        'sta': sta_patches,
                        'pregt': preseg,
                        'posgt': posseg,
                        'cdgt': cd_gt,
                        'ov_g': ov_guidance,
                        'prename': pre_name,
                        'posname': pos_name}
            else:
                
                return {'pre': pre_patches,
                        'pos': pos_patches,
                        'sta': sta_patches,
                        'ov_g': ov_guidance,
                        'prename': sv_pre_name,
                        'posname': sv_pos_name}
        else:
            preseg = np.asarray(preseg)
            posseg = np.asarray(posseg)
            gt_total    = (preseg, posseg, cd_gt)
            tf_data     = self.aug(image=data_total, masks=gt_total)
            
            pre_d, pos_d = tf_data['image'][0:3,:,:], tf_data['image'][3:6,:,:]
            preseg, posseg, cd_gt = tf_data['masks'][0], tf_data['masks'][1], tf_data['masks'][2]
            
            sta_d = np.zeros((1,1,1))
            if self.dc['transductive'] is True:
                sta_d = tf_data['image'][6:9,:,:]

            return {'pre': pre_d, 
                    'pos': pos_d, 
                    'sta': sta_d, 
                    'pregt': preseg,
                    'posgt': posseg,
                    'cdgt': cd_gt}
        
    def __len__(self,):
        return self.num_samples
