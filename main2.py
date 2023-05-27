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

#Function to initialize the weights using Xavier initialization
def init_parameters(model):
  for name, w in model.named_parameters():
    if "weight" in name:
      nn.init.xavier_normal_(w)
    if "bias" in name:
      nn.init.zeros_(w)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    #wandb.init
    #Transformation aplied to the images
    trans = torchvision.transforms.ToTensor()
        
    train_path = '.TrainingLandscape' #Training images path
    val_path = '.ValidationLandscape' #Validation images path
    dataset = DataClass(train_path, transform=trans) #Initialization of the dataset
    
    dataloader = LoaderClass(dataset, 128) #Initialization of the dataloader

    model = ConvAE2().to(device) #Initialization of the model
    init_parameters(model) #Xavier initialization of the weights
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.0) #Optimizer initialization
    criterion = nn.MSELoss() #Criterion initialization
    
    train(model, dataloader, criterion, optimizer, 20) #Training of the model
