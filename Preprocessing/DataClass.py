import os
import random
import torch
from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset
from Preprocessing.ColorProcessing import *

class DataClass(Dataset):
    def __init__(self, image_path, transform=None):
        self.image_path = image_path
        self.transform = transform
        self.total_imgs = os.listdir(image_path)
        self.lab_list = list()
        
        #Square the image and transform to lab
        for i in range(len(self.total_imgs)):
            print(i)
            img_loc = os.path.join(self.image_path, self.total_imgs[i])
            image = Image.open(img_loc).convert("RGB")
            tensor_image = self.transform(image)
            h = tensor_image.shape[1]
            trans2 = transforms.Compose([
                transforms.CenterCrop(h),
                transforms.Resize((128,128))
            ])
            rgb_image = trans2(tensor_image)
            lab_image = TransformToLAB(rgb_image)
            self.lab_list.append(lab_image)
        
    def __getitem__(self, idx):
        lab_image = self.lab_list[idx]
        return lab_image
    
    def __len__(self):
        return len(self.total_imgs)
    
    def shuffle(self):
        random.shuffle(self.lab_list)