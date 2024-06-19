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
import cv2


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
        image = Image.open(image_path)
        # opencv_image = np.array(image)
        #
        # opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_RGB2BGR)
        # gaussian_blurred = cv2.GaussianBlur(opencv_image, (15, 15), 0)
        # enhanced_image = cv2.convertScaleAbs(gaussian_blurred, alpha=1.5, beta=0)
        # enhanced_image_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)
        # image = Image.fromarray(enhanced_image_rgb)
        #
        # image = Image.open(image_path)  # not gray scale as picture is colored

        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):
        """Return the number of images."""
        return len(self.image_paths)


def get_sorted_file_paths(image_dir, mask_dir, extensions):
    """
    Returns a sorted list of tuples (image_path, mask_path) ensuring that filenames match.
    """
    image_files = [file for file in os.listdir(image_dir) if file.endswith(extensions)]
    mask_files = [file for file in os.listdir(mask_dir) if file.endswith(extensions)]

    # Sort the lists
    image_files.sort()
    mask_files.sort()

    # Ensure the number of images and masks match
    if len(image_files) != len(mask_files):
        raise ValueError("The number of images and masks do not match!")

    # Create tuples of image and mask paths, ensuring they have matching filenames
    paired_paths = []
    for image_file, mask_file in zip(image_files, mask_files):
        if image_file.split('.')[0] != mask_file.split('.')[0]:
            raise ValueError(f"Mismatched filenames: {image_file} and {mask_file}")
        image_path = os.path.join(image_dir, image_file)
        mask_path = os.path.join(mask_dir, mask_file)
        paired_paths.append((image_path, mask_path))

    return paired_paths


def binarize(img):
    width, height = img.size
    for x in range(width):
        for y in range(height):
            if img.getpixel((x,y)) > 0:
                img.putpixel((x,y),255)

            else:
                img.putpixel((x,y),0)
    return img

class MVTecAD_test_set_with_masks(data.Dataset):
    """Dataset class for the MVTecAD dataset."""

    def __init__(self, image_dir, transform):
        """Initialize and preprocess the MVTecAD dataset."""
        self.image_dir = image_dir
        self.transform = transform
        _extensions = ('.jpg', '.bmp', '.png')
        mask_dir = f"{image_dir}_mask"
        self.image_masks_paths = get_sorted_file_paths(image_dir, mask_dir, _extensions)

    def __getitem__(self, index):
        """Return one image"""

        image_path, mask_path = self.image_masks_paths[index]
        # This is the mask
        # mask_path = self.mask_paths[index]

        #image = Image.open(image_path).convert('L')  # Convert to grayscale
        image = Image.open(image_path)
        mask = Image.open(mask_path).convert('L')
        # mask = binarize(mask)
        # mask = mask.point(lambda p: 255 if p > 0 else 0)
        # Step 2: Convert to NumPy array
        mask_np = np.array(mask)
        #
        # Check for NaN values
        nan_mask = np.isnan(mask_np)
        if np.any(nan_mask):
            print("NaN values detected in the array")
            print(nan_mask)
        # # Step 3: Apply threshold to binarize the mask
        mask = (np.where(mask_np > 0, 1, 0)*255).astype(np.uint8)
        # binarized_mask_np = (mask)
        # binarized_mask_np = (mask_np > 0).astype(np.uint8)
        # binarized_mask_np = ((mask_np > 0).astype(np.uint8)*255).astype(np.uint8)
        #
        mask = Image.fromarray(mask)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)

        return image,mask

    def __len__(self):
        """Return the number of images."""
        return len(self.image_masks_paths)


def return_MVTecAD_loader(image_dir, batch_size=256, image_size=224,train=True):
    """Build and return a data loader."""
    transform = []
    # mean = [0.5, 0.5, 0.5]  # (assuming grayscale or RGB)
    # std = [0.5, 0.5, 0.5]  # (assuming grayscale or RGB)
    mean = [0.0622145,  0.0864737,  0.07538847]
    std = [0.09039213, 0.08525948, 0.09549119]
    # Desired output size of the crop
    crop_size = (400, 1936)  # Desired crop size
    #CenterCrop(crop_size)
    #transform.append(CenterCrop(crop_size))
    transform.append(T.Resize((image_size, image_size)))

    #transform.append(T.RandomCrop((128,128)))
    #transform.append(T.RandomHorizontalFlip(p=0.5))
    #transform.append(T.RandomVerticalFlip(p=0.5))
    transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean, std))
    transform = T.Compose(transform)

    dataset = MVTecAD(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=train,
                                  drop_last=True,
                                  num_workers=8,
                                  pin_memory=True)
    return data_loader


def return_MVTecAD_loader_test_GN(image_dir, batch_size=256,image_size=224):
    """Build and return a data loader."""
    transform = []
    # mean = [0.5, 0.5, 0.5]  # (assuming grayscale or RGB)
    # std = [0.5, 0.5, 0.5]  # (assuming grayscale or RGB)
    mean = [0.0622145,  0.0864737,  0.07538847]
    std = [0.09039213, 0.08525948, 0.09549119]
    # Desired output size of the crop
    crop_size = (400, 1936)  # Desired crop size
    #CenterCrop(crop_size)
    #transform.append(CenterCrop(crop_size))
    transform.append(T.Resize((image_size, image_size)))

    #transform.append(T.RandomCrop((128,128)))
    #transform.append(T.RandomHorizontalFlip(p=0.5))
    #transform.append(T.RandomVerticalFlip(p=0.5))
    transform.append(T.ToTensor())
    # transform.append(T.Normalize(mean, std))
    transform = T.Compose(transform)

    dataset = MVTecAD_test_set_with_masks(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  drop_last=False,
                                  num_workers=8,
                                  pin_memory=True)
    return data_loader
