import json
import numpy as np
import torch

from PIL import Image
from torchvision import transforms
from torchvision import datasets as dset
import torchvision
from torch.utils.data import Dataset, DataLoader
from .aptos import APTOS2019

def get_transform(transform_type='default', image_size=224, args=None):

    if transform_type == 'default':
        IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomVerticalFlip(p=0.5),
            # transforms.ColorJitter(),
            transforms.RandomCrop(size=(image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    return train_transform, test_transform

