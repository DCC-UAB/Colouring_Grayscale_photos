import os
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from Preprocessing.ColorProcessing import *

class DataClass(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        all_imgs = os.listdir(image_path)
        self.total_imgs = all_imgs
        
    def __getitem__(self, idx):
        img_loc = os.path.join(self.image_path, self.total_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        return tensor_image
    
    def __len__(self):
        return len(self.total_imgs)
    
    def shuffle(self):
        random.shuffle(self.total_imgs)

        
class LabImage():
    def __init__(self, dataclass):
        self.dataclass = dataclass
        self.lab_images = list()
        self.L = list()
        self.ab = list()
        for i in range(len(self.dataclass)):
            image = TransformToLAB(self.dataclass[i])
            self.lab_images.append(image)
        self.split_channels()

    def __getitem__(self, idx):
        return (self.L[idx], self.ab[idx])
    
    def __len__(self):
        return len(self.lab_images)
    
    def split_channels(self):
        self.L = list()
        self.ab = list()
        for image in self.lab_images:
            self.L.append(torch.unsqueeze(image[0],0))
            self.ab.append(image[1:3])
    
    def shuffle(self):
        random.shuffle(self.lab_images)
        self.split_channels()

    def get_grey(self, idx):
        return self.L[idx]
    
    def get_ab(self, idx):
        return self.ab[idx]
    
    def get_LAB_image(self, idx):
        return self.lab_images[idx]
