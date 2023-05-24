import wandb
import torch
import torchvision
import numpy as np
from torch import nn
from Preprocessing.DataClass import *
from Preprocessing.LoaderClass import *
from Preprocessing.ColorProcessing import *
from models.modelAutoencoder1 import *
from models.modelAutoencoder import *
from train import *
from plots import *

def init_parameters(model):
  for name, w in model.named_parameters():
    if "weight" in name:
      nn.init.xavier_normal_(w)
    if "bias" in name:
      nn.init.zeros_(w)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    #wandb.init
    trans = torchvision.transforms.ToTensor()
    trans2 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4336, 0.4392, 0.4503], [0.2467, 0.2137, 0.2168])
    ])
        
    faces_path = './LandscapeDataset'
    dataset = DataClass(faces_path, transform=trans)
    print("Dataset made")
    lab_data = LabImage(dataset)
    print("Lab made")

    #  lab_data[0][0].shape => [128, 128]
    #  lab_data[0][1].shape => [2, 128, 128]

    dataloader = LoaderClass(lab_data, 64)
    print("Dataloader done")
    model = modelAutoencoder1(input_size = 128).to(device)
    init_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay = 0.005)
    criterion = nn.MSELoss()
    train(model, dataloader, criterion, optimizer, 50)

    #PREDICT 1 IMAGE
    image = lab_data[65]
    grey = image[0]
    pred = model(grey.to(device)).to('cpu')
    print(pred)
    print(image[1])
    colored_image_Lab = torch.cat([grey*100, pred*128], dim=0).detach()
    colored_RGB = TransformToRGB(colored_image_Lab)
    showImage(colored_RGB)
