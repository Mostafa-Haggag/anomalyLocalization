# data loader 
import os
import numpy as np
from PIL import Image

import torch
from torch.utils import data
from torchvision import transforms as T
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
import glob


class CenterCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, image):
        width, height = image.size
        new_width, new_height = self.crop_size

        left = (width - new_width) / 2
        top = (height - new_height) / 2
        right = (width + new_width) / 2
        bottom = (height + new_height) / 2

        return image.crop((left, top, right, bottom))


class MVTecAD(data.Dataset):
    """Dataset class for the MVTecAD dataset."""

    def __init__(self, image_dir, transform):
        """Initialize and preprocess the MVTecAD dataset."""
        self.image_dir = image_dir
        self.transform = transform
        self.image_paths = sorted([os.path.join(image_dir, img) for img in os.listdir(image_dir) if img.endswith(('.jpg', '.bmp', '.png'))])

    def __getitem__(self, index):
        """Return one image"""
        image_path = self.image_paths[index]
        #image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = Image.open(image_path).convert('RGB')

        #image = Image.open(image_path)  # not gray scale as picture is colored

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """Return the number of images."""
        return len(self.image_paths)


def return_MVTecAD_loader(image_dir, batch_size=256, train=True):
    """Build and return a data loader."""
    transform = []
    mean = [0.5, 0.5, 0.5]  # (assuming grayscale or RGB)
    std = [0.5, 0.5, 0.5]  # (assuming grayscale or RGB)
    mean = [0.0622145,  0.0864737,  0.07538847]
    std = [0.09039213, 0.08525948, 0.09549119]
    # Desired output size of the crop
    crop_size = (400, 1936)  # Desired crop size
    #CenterCrop(crop_size)
    #transform.append(CenterCrop(crop_size))
    transform.append(T.Resize((112, 112)))

    #transform.append(T.RandomCrop((128,128)))
    #transform.append(T.RandomHorizontalFlip(p=0.5))
    #transform.append(T.RandomVerticalFlip(p=0.5))
    transform.append(T.ToTensor())
    ##transform.append(T.Normalize(mean, std))
    transform = T.Compose(transform)

    dataset = MVTecAD(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  drop_last=train,
                                  num_workers=8,
                                  pin_memory=True)
    return data_loader
