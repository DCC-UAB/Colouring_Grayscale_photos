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
    example_ct = 0
    batch_ct = 0
    for epoch in range(num_epochs):
        #Shuffle and batching of the dataloader
        loader.shuffle_data()
        loader.batch_sampler()
        total_loss = 0
        i = 0
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
            example_ct += len(batch)
            batch_ct += 1
            i += 1

            #if ((batch_ct + 1) % 25) == 0:
            #    train_log(train_loss, example_ct, epoch)
        print ("Epoch: " + str(epoch) + " | Loss: " + str(total_loss/len(loader)))
        
        #Plot an image of the training dataset with predicted colors
        image = loader[0][0]
        grey = image[0]
        pred = model(grey.to(device)).to('cpu')*128
        colored_image_Lab = torch.cat([grey, pred], dim=0).detach()
        colored_RGB = TransformToRGB(colored_image_Lab)
        showImage(colored_RGB,'./imatgesProva/'+str(epoch))
        fig2, axs2 = plt.subplots(1,2, figsize=(15, 6))
        axs2[0].imshow(torch.unsqueeze(pred[0].detach(), 2), cmap='Greys')
        axs2[1].imshow(torch.unsqueeze(pred[1].detach(), 2), cmap='Greys')
        plt.savefig('./imatgesProva/ab'+str(epoch))



def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
