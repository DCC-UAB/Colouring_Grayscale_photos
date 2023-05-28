[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11118782&assignment_repo_type=AssignmentRepo)
# Colouring grayscale photos
In this project, we aim to develop a program that can automatically colorize black and white images using deep neural networks. By leveraging the power of deep learning, we can teach our model to understand the context and semantics of images, enabling it to generate plausible and aesthetically pleasing colorizations. The primary objective of this project is to design and train a deep neural network model capable of accurately predicting and assigning appropriate color values to grayscale images. This involves teaching the model to understand various visual features, textures, and patterns present in colored images to produce realistic and visually coherent colorizations. Ultimately, our goal is to develop an efficient and effective system that can save significant time and effort for digital artists and photographers.

## Starting Point
Some papers have helped us to understand the underlying logic of the automatic colorization problem, as well as providing as a starting point architecture for our models. The most relevant ones are [Coloring with neuran networks]([https://github.com/saeed-anwar/ColorSurvey](https://emilwallner.medium.com/colorize-b-w-photos-with-a-100-line-neural-network-53d9b4449f8d)) and [Image Colorization Basics](https://www.kaggle.com/code/basu369victor/image-colorization-basic-implementation-with-cnn).

## Code structure
This repository is structured as shown below:
```
|--- Preprocessing
    |--- DataClass.py
    |--- LoaderClass.py
    |--- ColorProcessing.py
|--- models
    |--- Colorization_ConvAE.py
    |--- Simple_ConvAE.py
|--- PredictedImages_Training
|--- PredictedImages_Validation
|--- QuantitativeAnalysis
|--- LandscapeDataset
|--- train.py
|--- prediction.py
|--- plots.py
|--- main.py
```
The repository contains all the necessary scripts to succesfully train and validate the deep colorization models. Two scripts folders can be found: **Preprocessing**, which contains classes and functions to store and manipulate the images, and **models**, that contains the two models used in this project. There are also four more folders: **PredictedImages_**, that will store the pictures colored by the model during the training and the validation respectively; **QuantitativeAnalysis**, intended to save the plots of the loss and other metrics; and **TrainingLandscape** and **ValidationLandscape**, which contain the images for the training and the validation process. Finally,  we can see four scripts: **train.py** and **prediction.py** containing the training and validation functions, **plots.py**, consisting of a set of functions that allows the plot of the images, and the **main.py** script, which imports the datset, trains the model and make the validation.

## Database
For this deep colorization project, we utilized two distinct datasets: one consisting of face images and the other comprising fruit images with a white background. These datasets were individually chosen to ensure diversity within each domain and enable the model to learn specific colorization patterns for faces and fruits.

### Landscape Dataset
The landscape dataset was sourced from !!!!!!!!!!!!!!!!!!!!!!, [ColorSurvey](https://github.com/saeed-anwar/ColorSurvey). These images include a large variety of colorfull images, some of them with a strong color contrast and other with color gradient. The original images are of diferent sizes and resolutions, so their preprocessing is compulsory; our project contains the needed functions to make this proces. First, the vertical images should be eliminated from the dataset; then the images are cut to make them  squared, and finally are resized so that the final images are of size ```[128,128]```.

## Image representation
Models for colorization will obviously work on images, therefore understanding how images are represented is crucial. The most common numeric representation for images, and the one used for the images of our datasets, is the **RGB**. This name comes from *Red-Green-Blue*, because the image is divided on three channels, one encoding color red, the second encoding green and the last one, blue. Each of these channels indicate the intensity of the corresponding color in the pixel in a scale ```[0,1]```.

However, RGB is not the best representation for Deep Colorization: **Lab** representation is more fitting for the problem. Lab also encode the image in three channels: **L**, in a scale ```[0,100]```, encodes the lightness and corresponds to the black and white image; and **a** and **b** stands for the color spectra green–red and blue–yellow and their values are in range ```[-128,128]```. This encoding is really usefull for the present problem, the L channel can be the input of the model and channels a and b are the ones that the model should learn to predict. 

## Running example
First of all, the document **Requirements.txt** should be executed to install all the necessary dependencies. After that, an example code can be executed by writing the following command in the terminal:
```
python main.py
```
This will train the model ```ConvAE``` with the Landscape dataset in 500 epochs, and will generate 10 images of validation that will be saved in the **PredictedImages_Validation** directory along with the original colored picture. Moreover, a training loss evolution graphic will be saved in the **QuantitativeAnalysis** folder. 



## Contributors
* Laia Escursell Rof - laia.escursellr@autonoma.cat
* Abril Pérez Martí - abril.perezm@autonoma.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Computational Mathematics & Data analyitics, 
UAB, 2023
