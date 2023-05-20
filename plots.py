import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap


def showImage(image):
    fig, axs = plt.subplots(1,1, figsize=(4, 4))
    axs.imshow(image.permute(1,2,0))
    plt.show()


def showChannels_RGB(image):
    fig, axs = plt.subplots(1,3, figsize=(14, 26))
    axs[0].imshow(image[0], cmap='Reds')
    axs[1].imshow(image[1], cmap='Greens')
    axs[2].imshow(image[2], cmap='Blues')
    plt.show()


def showChannels_LAB(image):
    green_red = LinearSegmentedColormap.from_list('gr',["g", "w", "r"], N=256)
    yellow_blue = LinearSegmentedColormap.from_list('yb',["b", "w", "y"], N=256)
    fig, axs = plt.subplots(1,3, figsize=(14, 26))
    axs[0].imshow(image.permute(2,0,1)[0], cmap='Greys_r')
    axs[1].imshow(image.permute(2,0,1)[1], cmap=green_red)
    axs[2].imshow(image.permute(2,0,1)[2], cmap=yellow_blue)
    plt.show()