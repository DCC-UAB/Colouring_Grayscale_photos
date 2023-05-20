import torch
from skimage.color import rgb2lab, lab2rgb

def TransformToLAB(image):
    lab_img = rgb2lab(image.permute(1,2,0))
    return torch.from_numpy(lab_img).permute(2,0,1)

def TransformToRGB(image):
    rgb_img = lab2rgb(image.permute(1,2,0))
    return torch.from_numpy(rgb_img).permute(2,0,1)