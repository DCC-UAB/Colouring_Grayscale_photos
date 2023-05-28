import wandb
import torch
import torchvision
import numpy as np
from torch import nn
from Preprocessing.DataClass import *
from Preprocessing.LoaderClass import *
from Preprocessing.ColorProcessing import *
from models.StartingPoint_ConvAE import *
from models.Simple_ConvAE import *
from models.Colorization_ConvAE import *
from train import *
from plots import *
from prediction import *

#Function to initialize the weights using Xavier initialization
def init_parameters(model):
  for name, w in model.named_parameters():
    if "weight" in name:
      nn.init.xavier_normal_(w)
    if "bias" in name:
      nn.init.zeros_(w)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    print('start')
    
    #wandb.init
    #Transformation aplied to the images
    trans = torchvision.transforms.ToTensor()
   
    train_path = 'TrainingLandscape' #Training images path
    train_dataset = DataClass(train_path, transform=trans) #Initialization of the dataset
    print('Train dataset started')
    
    dataloader = LoaderClass(train_dataset, 64) #Initialization of the dataloader

    model = Colorization_ConvAE().to(device) #Initialization of the model
    init_parameters(model) #Xavier initialization of the weights
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay = 0.0) #Optimizer initialization
    criterion = nn.MSELoss() #Criterion initialization
    
    #Print loss model
    losses = train(model, dataloader, criterion, optimizer, 500) #Training of the model
    showLoss(losses, './LossEvaluation/total')
    losses.pop(0)
    showLoss(losses, './LossEvaluation/evalutaion')

    #Save the model so there is no need to train it again
    torch.save(model.state_dict(), 'TrainedModel') 

    #Load the model trained
    '''
    model = Colorization_ConvAE()
    model.load_state_dict(torch.load('TrainedModel'))
    model.to(device)
    '''

    #Validate the model
    val_path = 'ValidationLandscape' #Validation images path
    val_dataset = DataClass(val_path, transform=trans)  #Initialization of the dataset
    print('Val dataset started')

    prediction(val_dataset, model, './PredictedImages_Validation/', 10) #Prediction of some images
