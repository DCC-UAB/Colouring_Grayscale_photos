import wandb
import torch
import torchvision
import numpy as np
from torch import nn
from Preprocessing.DataClass import *
from Preprocessing.LoaderClass import *
from Preprocessing.ColorProcessing import *
from models.modelAutoencoder1 import *
from plots import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, loader, criterion, optimizer, num_epochs):
    #wandb.init
    #wandb.watch(model, criterion, log='all', log_freq=10)
    model.train()
    train_loss = 0
    losses = []
    for epoch in range(num_epochs):
        #Shuffle and batching of the dataloader
        loader.shuffle_data()
        loader.batch_sampler()
        total_loss = 0
        
        for num_batch in range(len(loader)):
            batch = loader[num_batch]
            optimizer.zero_grad()
            x = batch[0]
            y = batch[1]
            output = model(x.to(device)) #Prediction of the batch
            train_loss = criterion(output.to(device), y.to(device)) #Loss calculus
            total_loss += train_loss
            train_loss.backward()
            optimizer.step()
        print ("Epoch: " + str(epoch) + " | Loss: " + str(total_loss.item()/len(loader)))
        losses.append(total_loss.item()/len(loader))

        #Plot an image of the training dataset with predicted colors
        image = loader[0][0]
        grey = image[0]
        pred = model(grey.to(device)).to('cpu')*128
        showImageTraining(grey, pred, epoch)
        showSpectrum(pred, epoch)

    return losses