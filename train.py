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
    train_loss = 0
    model.train()
    example_ct = 0
    batch_ct = 0
    for epoch in range(num_epochs):
        loader.shuffle_data()
        loader.batch_sampler()
        i = 0
        for batch in loader:
            print(i)
            optimizer.zero_grad()
            greys = loader.get_batch_greys(i)
            label = loader.get_batch_labels(i)[0]
            outs = model(greys.to(device))[0]
            train_loss = criterion(outs.to(device), label.to(device))
            train_loss.backward()
            optimizer.step()
            example_ct += len(batch)
            batch_ct += 1
            i += 1

            #if ((batch_ct + 1) % 25) == 0:
            #    train_log(train_loss, example_ct, epoch)
        print ("Epoch: " + str(epoch) + " | Loss: " + str(train_loss))
        image = loader[0][0]
        grey = image[0]
        pred = model(grey.to(device)).to('cpu')
        colored_image_Lab = torch.cat([grey*100, pred*128], dim=0).detach()
        colored_RGB = TransformToRGB(colored_image_Lab)
        showImage(colored_RGB,'./imatgesProva/'+str(epoch))
        fig2, axs2 = plt.subplots(1,2, figsize=(15, 6))
        axs2[0].imshow(torch.unsqueeze(pred[0].detach(), 2), cmap='Greys')
        axs2[1].imshow(torch.unsqueeze(pred[1].detach(), 2), cmap='Greys')
        plt.savefig('./imatgesProva/ab'+str(epoch))



def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")
