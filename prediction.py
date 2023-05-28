import torch
import torchvision
import matplotlib.pyplot as plt
from Preprocessing.ColorProcessing import *
from Preprocessing.DataClass import *
from Preprocessing.LoaderClass import *
from plots import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def predict_sample(model, image, path):
    grey = torch.unsqueeze(image[0], dim=0)
    pred = model(grey.to(device)).to('cpu')
    colored_image_Lab = torch.cat([grey, pred*128], dim=0).detach()
    original_image_Lab = torch.cat([grey, torch.unsqueeze(image[1], dim=0), torch.unsqueeze(image[2], dim=0)], dim=0).detach()
    colored_RGB = TransformToRGB(colored_image_Lab)
    original_RGB = TransformToRGB(original_image_Lab)

    #save images
    fig, axs = plt.subplots(1,2, figsize=(16, 8))
    axs[0].imshow(original_RGB.permute(1,2,0))
    axs[0].axis('off')
    axs[1].imshow(colored_RGB.permute(1,2,0))
    axs[1].axis('off')
    plt.savefig(path)

def prediction(dataset, model, folder, num_samples=1):
    for i in range(num_samples):
        image = dataset[i]
        predict_sample(model, image, folder+str(i))
