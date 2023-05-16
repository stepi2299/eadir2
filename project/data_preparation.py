import os
import pandas as pd
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import copy as cp
import numpy as np


def visualize_depth_map(data, index=None):
    if index is not None:
        imgs = data[index]
        img, depth = visualize_img(imgs)
        figure, axs = plt.subplots(1, 2, figsize=(10, 5))
        figure.suptitle(f'Image with depth map, index {index}')
        axs[0].set_title("Base image")
        axs[0].imshow(img)
        axs[1].set_title("Depth Map")
        axs[1].imshow(depth, cmap="plasma")
    else:
        print("Show 20 first image pairs")
        figure, axs = plt.subplots(20, 2, figsize=(10, 20 * 5))
        for i, imgs in enumerate(data):
            if i == 20:
                break
            img, depth = visualize_img(imgs)
            axs[i, 0].set_title(f"Base image with index: {i}")
            axs[i, 0].imshow(img)
            axs[i, 1].set_title("Depth Map")
            axs[i, 1].imshow(depth, cmap="plasma")
    plt.show()


def visualize_img(data):
    img, depth = cp.deepcopy(data)
    img = img.detach().cpu().numpy()
    r = img[0]
    g = img[1]
    b = img[2]
    img = (cv.merge([r, g, b]) + 1) * 125
    img = img.astype("uint8")
    return img, depth


data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.ColorJitter(brightness=0, contrast=0, saturation=0, hue=0),
        transforms.GaussianBlur(kernel_size=(7,13), sigma=(0.1, 2.0)),
    ])

visual_mask_fn = lambda x: np.array(x) / 10000


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


annotation_file = "/home/stepi2299/repo/eadir2/data/nyu_samples/nyu2_test.csv"
img_folder = "/home/stepi2299/repo/eadir2/data/nyu_samples/nyu2_test"
dataset = NYUDataset(annotation_file, img_folder, data_transform, visual_mask_fn)

visualize_depth_map(dataset)
