from torchvision.io import read_image
from torchvision import transforms
from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np


class Dataset(Dataset):
  def __init__(self, img_path, transform=None):
    self.base_img_path = img_path
    self.images_path = os.listdir(img_path)
    self.transform = transform

  def __len__(self):
    return len(os.listdir(self.images_path))

  def __getitem__(self, idx):
    if idx is None:
      idx = 0
    img = Image.open(os.path.join(self.base_img_path, self.images_path[idx]))
    width, height = img.size
    left_box = (0, 0, width // 2, height)
    right_box = (width // 2, 0, width, height)
    image = img.crop(left_box)
    seg_map = img.crop(right_box)
    if self.transform:
      image = self.transform(image)
    return image, seg_map
