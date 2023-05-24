import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import copy as cp
import numpy as np


class NYUDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert('RGB')
        mask_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        mask = Image.open(mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        return image, mask


data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.GaussianBlur(kernel_size=(7,13), sigma=(0.1, 2.0)),
        transforms.RandomHorizontalFlip(p=0.5),
    ])

visual_mask_fn = lambda x: np.array(x) / 10000
