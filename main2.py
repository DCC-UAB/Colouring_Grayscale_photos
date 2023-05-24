import wandb
import torch
import torchvision
import numpy as np
from torch import nn
from Preprocessing.DataClass import *
from Preprocessing.LoaderClass import *
from Preprocessing.ColorProcessing import *
from models.modelAutoencoder1 import *
from train import *
from plots import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    #wandb.init
    trans = torchvision.transforms.ToTensor()
    faces_path = './LandscapeDataset'
    dataset = DataClass(faces_path, transform=trans)
    lab_data = LabImage(dataset)

    #  lab_data[0][0].shape => [128, 128]
    #  lab_data[0][1].shape => [2, 128, 128]

    dataloader = LoaderClass(lab_data, 64)
    model = modelAutoencoder1(input_size = 225).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train(model, dataloader, criterion, optimizer, 30)

    #PREDICT 1 IMAGE
    image = lab_data[65]
    grey = image[0]
    pred = model(grey.to(device)).to('cpu')*128
    print(pred)
    colored_image_Lab = torch.cat([grey, pred], dim=0).detach()
    colored_RGB = TransformToRGB(colored_image_Lab)
    showImage(colored_RGB)
