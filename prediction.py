import torch
import torchvision
import matplotlib.pyplot as plt
from Preprocessing.ColorProcessing import *
from Preprocessing.DataClass import *
from Preprocessing.LoaderClass import *
from plots import *

def predict(model, image):
    grey = image[0]
    pred = model(grey.to(device)).to('cpu')
    colored_image_Lab = torch.cat([grey*100, pred*128], dim=0).detach()
    original_image_Lab = torch.cat([grey*100, image[1]*128], dim=0).detach()
    colored_RGB = TransformToRGB(colored_image_Lab)
    original_RGB = TransformToRGB(original_image_Lab)

    # print images
    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    axs[0].imshow(original_RGB.permute(1,2,0))
    axs[0].axis('off')
    axs[1].imshow(colored_RGB.permute(1,2,0))
    axs[1].axis('off')
    plt.show()
