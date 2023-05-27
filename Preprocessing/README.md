# Preprocessing
This folder contains essentials files for images preprocessing designed for colorization.

## ColorProcessing
This file provides two functions for tansforming images to LAB and RGB. Both functions have as inputs and outpus an image in tensor format. 
### `TransformToLAB(image)`
This function takes an input image in the RGB color space and transforms it into the LAB color space.
### `TransformToRGB(image)`
This function takes an input image in the LAB color space and transforms it into the RGB color space.

## DataClass
This file defines the `DataClass` and `LabImage` classes for handling image datasets.
### `DataClass`
This class is a subclass of `torch.utils.data.Dataset` and provides methods for loading and processing images.
### `LabImage`
This class is a collection of the LAB color space images crated from the `DataClass`dataset.

## LoaderClass
This file contains the `LoaderClass` class for creating data loaders to iterate throught batches of data.
### `LoaderClass`
This class combines a dataset and a sampler, and provides an iterable over the given dataset.
