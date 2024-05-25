# Preprocessing
This folder contains essentials files for images preprocessing designed for colorization.

## ColorProcessing
This file provides two functions that guarantee a consistent image transformation from RGB format to Lab coding and vice versa. Both functions get an input tensor of dimensions $(3, 128, 128)$ and returns another tensor of same dimensions.

## DataClass
This file defines the `DataClass` class for handling image datasets. This one is a subclass of `torch.utils.data.Dataset` and provides methods for loading and processing images. The initialization will only ask for the path where the images are stored, will crop and resize the images so they have dimensions $(3, 128, 128)$, and finally will transform them to Lab encoding. This process may take a while if the dataset is large or if the images are large and complex. When returning a sample of the dataset, the image in Lab encoding will be obtained in tensor form, but note that the grey channel will be in scale [0,100] and the other two channels will have range [-1,1].

## LoaderClass
This file contains the `LoaderClass` class for creating data loaders to iterate throught batches of data. The initialization of one instance will ask for the dataset where the data is stored, the size of each batch of data, and a boolean saving if the data should be shuffled in each epoch.
