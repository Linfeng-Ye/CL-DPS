import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from util import Transpose0312
import random
import math
from PIL import Image

class FFHQ(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, Resize=None, JRC=None, transform=None,distortion=None, 
                 normalize = None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.distortion = distortion
        self.normalize = normalize
        self.train = train
        self.transpose = Transpose0312()
        self.ToTensor = transforms.ToTensor()
        self.Resize = Resize
        self.JRC = JRC
    def __len__(self):
        if self.train:
            return 69000
        else:
            return 1000
    def __getitem__(self, idx):
        if not self.train:
            idx = idx + 69000
        img_name = os.path.join(self.root_dir, str(idx).zfill(5)+".png")
        image = Image.open(img_name) # PIL image
        
        image = self.Resize(image)  # PIL image # 256 X 256 X 3
        image = self.transform(image)  # PIL image # 256 X 256 X 3

        Dimage = self.distortion(image) # PIL image # 256 X 256 X 3

        image, Dimage = self.JRC(image, Dimage)  # PIL image # 64 X 64 X 3

        image = self.normalize(self.ToTensor(image))

        Dimage = self.normalize(self.ToTensor(Dimage))
        
        return Dimage, image, idx