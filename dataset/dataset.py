from typing import List, Tuple
from collections import Sized
import os
from os.path import join
import albumentations as alb
from torchvision.transforms import Normalize

import numpy as np
import torch
from matplotlib.image import imread
import cv2
from torch.utils.data import Dataset
from torch import Tensor


class LEVIRLoader(Dataset, Sized):
    # LEVIR dataloader
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
        self._A = join(data_path, "A")
        self._B = join(data_path, "B")
        self._label = join(data_path, "label")

        # In all the dirs, the files share the same names:
        self._list_images = self._read_images_list(data_path)

        # Initialize augmentations:
        if mode == 'train':
            self._augmentation = _create_shared_augmentation()
            self._aberration = _create_aberration_augmentation()

        # Initialize normalization:
        self._normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    def __getitem__(self, indx):
        # Current image set name:
        imgname = self._list_images[indx].strip('\n')

        # Loading the images:
        x_ref = imread(join(self._A, imgname))
        x_test = imread(join(self._B, imgname))
        x_mask = _binarize(imread(join(self._label, imgname)))

        # Data augmentation in case of training:
        if self._mode == "train":
            x_ref, x_test, x_mask = self._augment(x_ref, x_test, x_mask)

        # Trasform data from HWC to CWH:
        x_ref, x_test, x_mask = self._to_tensors(x_ref, x_test, x_mask)

        return (x_ref, x_test), x_mask

    def __len__(self):
        return len(self._list_images)

    def _read_images_list(self, data_path: str) -> List[str]:
        images_list_file = join(data_path, 'list', self._mode + ".txt")
        with open(images_list_file, "r") as f:
            return f.readlines()

    def _augment(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # First apply augmentations in equal manner to test/ref/x_mask:
        transformed = self._augmentation(
            image=x_ref, image0=x_test, x_mask0=x_mask)
        x_ref = transformed["image"]
        x_test = transformed["image0"]
        x_mask = transformed["x_mask0"]

        # Then apply augmentation to single test ref in different way:
        x_ref = self._aberration(image=x_ref)["image"]
        x_test = self._aberration(image=x_test)["image"]

        return x_ref, x_test, x_mask

    def _to_tensors(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask: np.ndarray
    ) -> Tuple[Tensor, Tensor, Tensor]:
        return (
            self._normalize(torch.tensor(x_ref).permute(2, 0, 1)),
            self._normalize(torch.tensor(x_test).permute(2, 0, 1)),
            torch.tensor(x_mask),
        )


class SPN7Loader(Dataset, Sized):
    # SPN7 dataloader
    def __init__(
        self,
        data_path: str,
        mode: str,
    ) -> None:
        # Set image and label directory path
        self._mode = mode
        self.dir_path_image = os.path.join(data_path, mode, "images")
        self.dir_path_label = os.path.join(data_path, mode, "labels")
        # Perform augmentation only when training
        if mode == 'train':
            self._augmentation = _create_shared_augmentation()
            self._aberration = _create_aberration_augmentation()
        # Normalization
        self._normalize = Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

    def __getitem__(self, indx):
        list_images_1 = []
        list_images_2 = []
        list_labels_1 = []
        list_labels_2 = []
        # Make image pair and save path in each list1 and list2 (list1 : before, list2 : after)
        # Set image pair for adjacent period
        for (root, directories, files) in os.walk(self.dir_path_image):
            files.sort()
            length = len(files)
            for image_file in files:
                if files.index(image_file) == 0:
                    list_images_1.append(os.path.join(root, image_file))
                elif files.index(image_file) == length-4:
                    list_images_2.append(os.path.join(root, image_file))
                else:
                    list_images_1.append(os.path.join(root, image_file))
                    list_images_2.append(os.path.join(root, image_file))

        # Make label pair and save path in each list1 and list2 (list1 : before, list2 : after)
        # Set label pair for adjacent period
        for (root, directories, files) in os.walk(self.dir_path_label):
            files.sort()
            length = len(files)
            for label_file in files:
                if files.index(label_file) == 0:
                    list_labels_1.append(os.path.join(root, label_file))
                elif files.index(label_file) == length-4:
                    list_labels_2.append(os.path.join(root, label_file))
                else:
                    list_labels_1.append(os.path.join(root, label_file))
                    list_labels_2.append(os.path.join(root, label_file))

        # Loading the images:
        x_ref = cv2.imread(list_images_1[indx])
        x_test = cv2.imread(list_images_2[indx])
        x_mask_ref = _binarize(cv2.imread(
            list_labels_1[indx], cv2.IMREAD_GRAYSCALE))
        x_mask_test = _binarize(cv2.imread(
            list_labels_2[indx], cv2.IMREAD_GRAYSCALE))

        # Resize the size to 256*256*3 and 256*256 which is the fixed input size of TinyCD
        x_ref = np.resize(x_ref, (256, 256, 3))
        x_test = np.resize(x_test, (256, 256, 3))
        x_mask_ref = np.resize(x_mask_ref, (256, 256))
        x_mask_test = np.resize(x_mask_test, (256, 256))

        # Data augmentation in case of training:
        if self._mode == "train":
            x_ref, x_test, x_mask_ref, x_mask_test = self._augment(
                x_ref, x_test, x_mask_ref, x_mask_test)

        # Change mask generation
        x_mask = np.logical_xor(x_mask_ref, x_mask_test)

        # Change datatype from bool to float
        x_ref, x_test, x_mask = x_ref.astype(np.float), x_test.astype(
            np.float), x_mask.astype(np.float)

        # Trasform data from HWC to CWH:
        x_ref, x_test, x_mask = self._to_tensors(x_ref, x_test, x_mask)

        return (x_ref, x_test), x_mask

    # Total number of image and label pair
    def __len__(self):
        num = 0
        for (root, directories, files) in os.walk(self.dir_path_label):
            if len(files) != 0:
                num += (len(files) - 1)
        return num

    def _augment(
        self, x_ref: np.ndarray, x_test: np.ndarray, x_mask_ref: np.ndarray, x_mask_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # First apply augmentations in equal manner to test/ref/x_mask:
        transformed = self._augmentation(
            image=x_ref, image0=x_test, x_mask0=x_mask_ref, x_mask1=x_mask_test)
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
