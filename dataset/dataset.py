from typing import List, Tuple
from collections import Sized
import os
import albumentations as alb
from torchvision.transforms import Normalize

import numpy as np
import torch
from matplotlib.image import imread
import cv2
from torch.utils.data import Dataset
from torch import Tensor

class MyDataset(Dataset, Sized):
    def __init__(
        self,
        data_path: str,
        mode: str,
    ) -> None:
        """
        data_path: Folder containing the sub-folders:
            "A" for test images,
            "B" for ref images, 
            "label" for the gt masks,
            "list" containing the image list files ("train.txt", "test.txt", "eval.txt").
        """
        # Store the path data path + mode (train,val,test):
        self._mode = mode
        #self._A = join(data_path, "A")
        #self._B = join(data_path, "B")
        #self._label = join(data_path, "label")
        if mode == 'train':
            self.dir_path_image = os.path.join(data_path, mode, "images")
            self.dir_path_label = os.path.join(data_path, mode, "labels")
        elif mode == 'val' :
            self.dir_path_image = os.path.join(data_path, mode, "images")
            self.dir_path_label = os.path.join(data_path, mode, "labels")
        # In all the dirs, the files share the same names:
        #self._list_images = self._read_images_list(data_path)

        # Initialize augmentations:
        if mode == 'train':
            self._augmentation = _create_shared_augmentation()
            self._aberration = _create_aberration_augmentation()
        
        # Initialize normalization:
        self._normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    def __getitem__(self, indx):
        # Current image set name:
        #imgname = self._list_images[indx].strip('\n')
        list_images_1 =  []
        list_images_2 = []
        list_labels_1 = []
        list_labels_2 = []
        for (root, directories, files) in os.walk(self.dir_path_image):
            files.sort()
            length = len(files)
            for i in range(len(files)-3):
                if i==0:
                    list_images_1.append(os.path.join(root, files[i]))
                elif i==len(files)-4:
                    list_images_2.append(os.path.join(root, files[i]))
                else:
                    list_images_1.append(os.path.join(root, files[i]))
                    list_images_2.append(os.path.join(root, files[i]))


        for (root, directories, files) in os.walk(self.dir_path_label):
            files.sort()
            for i in range(len(files)):
                if i==0:
                    list_labels_1.append(os.path.join(root, files[i]))
                elif i==len(files)-1:
                    list_labels_2.append(os.path.join(root, files[i]))
                else:
                    list_labels_1.append(os.path.join(root, files[i]))
                    list_labels_2.append(os.path.join(root, files[i]))
             
        # Loading the images:
        x_ref = cv2.imread(list_labels_1[indx])
        x_test = cv2.imread(list_labels_2[indx])
        x_mask_ref = _binarize(cv2.imread(list_labels_1[indx], cv2.IMREAD_GRAYSCALE))
        x_mask_test = _binarize(cv2.imread(list_labels_2[indx], cv2.IMREAD_GRAYSCALE))

        x_ref = np.resize(x_ref, (256, 256, 3))
        x_test = np.resize(x_test, (256, 256, 3))
        x_mask_ref = np.resize(x_mask_ref, (256, 256))
        x_mask_test = np.resize(x_mask_test, (256, 256))
       
        # Data augmentation in case of training:
        if self._mode == "train":
            x_ref, x_test, x_mask_ref, x_mask_test = self._augment(x_ref, x_test, x_mask_ref, x_mask_test)
        
        x_mask = np.logical_xor(x_mask_ref, x_mask_test)
        
        x_ref = x_ref.astype(np.float)
        x_test = x_test.astype(np.float)
        x_mask = x_mask.astype(np.float)

        # Trasform data from HWC to CWH:
        x_ref, x_test, x_mask = self._to_tensors(x_ref, x_test, x_mask)

        return (x_ref, x_test), x_mask

    def _num(self):
        num = 1
        for (root, directories, files) in os.walk(self.dir_path_label):
            num += (len(files) -1)
        return num

    def __len__(self):
        return self._num()
    
    def _augment(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask_ref: np.ndarray, x_mask_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # First apply augmentations in equal manner to test/ref/x_mask:
        transformed = self._augmentation(image=x_ref, image0=x_test, x_mask0=x_mask_ref, x_mask1=x_mask_test)
        x_ref = transformed["image"]
        x_test = transformed["image0"]
        x_mask_ref = transformed["x_mask0"]
        x_mask_test = transformed["x_mask1"]

        # Then apply augmentation to single test ref in different way:
        x_ref = self._aberration(image=x_ref)["image"]
        x_test = self._aberration(image=x_test)["image"]

        return x_ref, x_test, x_mask_ref, x_mask_test
    
    def _to_tensors(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self._normalize(torch.tensor(x_ref).permute(2, 0, 1)),
            self._normalize(torch.tensor(x_test).permute(2, 0, 1)),
            torch.tensor(x_mask),
        )


def _create_shared_augmentation():
    return alb.Compose(
        [
            alb.Flip(p=0.5),
            alb.Rotate(limit=5, p=0.5),
        ],
        additional_targets={"image0": "image", "x_mask0": "mask"},
    )


def _create_aberration_augmentation():
    return alb.Compose([
        alb.RandomBrightnessContrast(
            brightness_limit=0.2, contrast_limit=0.2, p=0.5
        ),
        alb.GaussianBlur(blur_limit=[3, 5], p=0.5),
    ])

def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.clip(mask * 255, 0, 1).astype(int)