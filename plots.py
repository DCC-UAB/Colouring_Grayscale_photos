import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

#Show one image which is passed in RGB format
def showImage(image, path):
    fig, axs = plt.subplots(1,1, figsize=(4, 4))
    axs.imshow(image.permute(1,2,0)) #Changes the order of the dimensions to (128, 128, 1)
    plt.savefig(path)

#Show the three channels of an RGB image
def showChannels_RGB(image):
    fig, axs = plt.subplots(1,3, figsize=(14, 26))
    axs[0].imshow(image[0], cmap='Reds') #Red channel
    axs[1].imshow(image[1], cmap='Greens') #Green channel
    axs[2].imshow(image[2], cmap='Blues') #Blue channel
    plt.show()

#Show the three channels of a Lab image
def showChannels_LAB(image):
    green_red = LinearSegmentedColormap.from_list('gr',["g", "w", "r"], N=256)
    yellow_blue = LinearSegmentedColormap.from_list('yb',["b", "w", "y"], N=256)
    fig, axs = plt.subplots(1,3, figsize=(14, 26))
    axs[0].imshow(image.permute(2,0,1)[0], cmap='Greys_r') #L channel, black and white image
    axs[1].imshow(image.permute(2,0,1)[1], cmap=green_red) #a channel, green and red spectrum
    axs[2].imshow(image.permute(2,0,1)[2], cmap=yellow_blue) #b channel, yellow and blue spectrum
    plt.show()
