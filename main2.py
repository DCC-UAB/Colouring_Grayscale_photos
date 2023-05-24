import wandb
import torch
import torchvision
import numpy as np
from torch import nn
from Preprocessing.DataClass import *
from Preprocessing.LoaderClass import *
from Preprocessing.ColorProcessing import *
from Models.Model1 import *
from Models.Model2 import *
from plots import *

if __name__ == '__main__':
    #wandb.init
    trans = torchvision.transforms.ToTensor()
    faces_path = 'C:\\Users\\abril\\Documents\\Practiques MatCAD\\Xarxes Neuronals\\DeepColorization\\Faces'
    dataset = DataClass(faces_path, transform=trans)
    lab_data = LabImage(dataset)
    dataloader = LoaderClass(lab_data, 128)
    model = modelAutoencoder1(input_size = 128)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    train(model, dataloader, criterion, optimizer, 5)

    #PREDICT 1 IMAGE
    image = lab_data[65]
    grey = image[0]
    pred = model(grey)
    colored_image_Lab = torch.cat([grey, pred], dim=0).detach()
    colored_RGB = TransformToRGB(colored_image_Lab)
    showImage(colored_RGB)
