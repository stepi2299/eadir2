import os
import pandas as pd
from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import imageio.v3 as iio
import cv2 as cv
import numpy as np
from torchvision import transforms
import torch
import math


transform_degrade = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((1024, 2048)),
        transforms.GaussianBlur(kernel_size=(13, 15), sigma=(0.4, 4.0)),
    ]
)

transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((1024, 2048)),
    ]
)


class DegradationModel:
    def __init__(self, deg):
        self.deg = deg

    def process(self, img):
        degradation = self.deg(img)
        return degradation + math.sqrt(0.04) * torch.randn(degradation.shape)


class SRDataset(Dataset):
    def __init__(self, img_dir, transform=None, degradation_model=None):
        self.img_dir = img_dir
        self.degradation_model = degradation_model
        self.img_names = os.listdir(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        image = iio.imread(img_path)
        degraded_image = image.copy()

        if self.degradation_model:
            degraded_image = self.degradation_model.process(degraded_image)
        if self.transform:
            image = self.transform(image)
        return image, degraded_image

    def visualize_data(self, idx=None):
        if idx is None:
            idx = 0
        img, deg = self[idx]
        img = self._visual_img(img)
        deg = self._visual_img(deg)
        fig, ax = plt.subplots(1, 2, figsize=(10, 10))
        fig.suptitle("Hight Resolution and Low Resolution")
        ax[0].imshow(img)
        ax[1].imshow(deg)

    def _visual_img(self, img):
        img = img.detach().cpu().numpy()
        r = img[0]
        g = img[1]
        b = img[2]
        img = cv.merge([r, g, b]) * 255
        return img.astype("uint8")


deg_model = DegradationModel(transform_degrade)
sr_dataset = SRDataset("../../data/sr_data", transform, deg_model)

sr_dataset.visualize_data(0)
plt.show()
