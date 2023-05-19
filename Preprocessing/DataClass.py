import os
import random
from PIL import Image
from torch.utils.data import Dataset

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
