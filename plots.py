import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap
from Preprocessing.ColorProcessing import *
import torch

#Show one image which is passed in RGB format
def showImage(image, path):
    fig, axs = plt.subplots(1,1, figsize=(4, 4))
    axs.imshow(image.permute(1,2,0)) #Changes the order of the dimensions to (128, 128, 1)
    axs.axis('off')
    plt.savefig(path)
    plt.close(fig)

#Show the three channels of an RGB image
def showChannels_RGB(image):
    fig3, axs = plt.subplots(1,3, figsize=(14, 26))
    axs[0].imshow(image[0], cmap='Reds') #Red channel
    axs[1].imshow(image[1], cmap='Greens') #Green channel
    axs[2].imshow(image[2], cmap='Blues') #Blue channel
    plt.show()
    plt.close(fig3)

#Show the three channels of a Lab image
def showChannels_LAB(image):
    green_red = LinearSegmentedColormap.from_list('gr',["g", "w", "r"], N=256)
    yellow_blue = LinearSegmentedColormap.from_list('yb',["b", "w", "y"], N=256)
    fig4, axs = plt.subplots(1,3, figsize=(14, 26))
    axs[0].imshow(image.permute(2,0,1)[0], cmap='Greys_r') #L channel, black and white image
    axs[1].imshow(image.permute(2,0,1)[1], cmap=green_red) #a channel, green and red spectrum
    axs[2].imshow(image.permute(2,0,1)[2], cmap=yellow_blue) #b channel, yellow and blue spectrum
    plt.show()
    plt.close(fig4)

#Show the loss evolution
def showLoss(losses, path):
    fig5, axs = plt.subplots(1,1)
    axs.plot(losses)
    plt.title('model loss')
    plt.savefig(path)
    plt.close(fig5)

#Show the spectrum of the image while training the model
def showSpectrum(pred, epoch):
    fig2, axs2 = plt.subplots(1,2, figsize=(15, 6))
    axs2[0].imshow(torch.unsqueeze(pred[0].detach(), 2), cmap='Greys')
    axs2[0].axis('off')
    axs2[1].imshow(torch.unsqueeze(pred[1].detach(), 2), cmap='Greys')
    axs2[1].axis('off')
    plt.savefig('./PredictedImages_Training/SpectrumEpoch'+str(epoch))
    plt.close(fig2)

#Show the image RGB while training the model
def showImageTraining(grey, pred, epoch):
    colored_image_Lab = torch.cat([grey, pred], dim=0).detach()
    colored_RGB = TransformToRGB(colored_image_Lab)
    showImage(colored_RGB, './PredictedImages_Training/ColoredEpoch'+str(epoch))