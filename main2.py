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
from models.ConvAE2 import *
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

    dataloader = LoaderClass(dataset, 128)
    print("Dataloader done")
    model = ConvAE2().to(device)
    init_parameters(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.0)
    criterion = nn.MSELoss()
    train(model, dataloader, criterion, optimizer, 20)


    #PREDICT 1 IMAGE
    image = dataset[0]
    grey = image[0]
    pred = model(grey.to(device)).to('cpu')
    print(pred)
    print(image[1])
    colored_image_Lab = torch.cat([grey, pred], dim=0).detach()
    original_image_Lab = torch.cat([grey, image[1]], dim=0)
    colored_RGB = TransformToRGB(colored_image_Lab)
    original_RGB = TransformToRGB(original_image_Lab)
    showImage(colored_RGB,'./imatgesProva/pred')
    showImage(original_RGB,'./imatgesProva/original')
