[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/sPgOnVC9)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=11118782&assignment_repo_type=AssignmentRepo)
# Colouring grayscale photos
In this project, we aim to develop a program that can automatically colorize black and white images using deep neural networks. By leveraging the power of deep learning, we can teach our model to understand the context and semantics of images, enabling it to generate plausible and aesthetically pleasing colorizations. The primary objective of this project is to design and train a deep neural network model capable of accurately predicting and assigning appropriate color values to grayscale images. This involves teaching the model to understand various visual features, textures, and patterns present in colored images to produce realistic and visually coherent colorizations. Ultimately, our goal is to develop an efficient and effective system that can save significant time and effort for digital artists and photographers.

## Code structure
You must create as many folders as you consider. You can use the proposed structure or replace it by the one in the base code that you use as starting point. Do not forget to add Markdown files as needed to explain well the code and how to use it.

## Database
For this deep colorization project, we utilized two distinct datasets: one consisting of face images and the other comprising fruit images with a white background. These datasets were individually chosen to ensure diversity within each domain and enable the model to learn specific colorization patterns for faces and fruits.

### Face Dataset
The face dataset used in this project was sourced from a dedicated GitHub repository, [DeepColorization](https://github.com/2014mchidamb/DeepColorization/tree/master/face_images). This repository provides a comprehensive collection of face images, carefully curated to encompass a wide range of facial features, expressions, and lighting conditions.

### Landscape Dataset
The fruit dataset was sourced from another GitHub repository, [ColorSurvey](https://github.com/saeed-anwar/ColorSurvey). These images include a large variety of colorfull images, some of them with a strong color contrast and other with color gradient. The original images are of diferent sizes and resolutions, so their preprocessing is compulsory; our project contains the needed functions to make this proces. First, the vertical images should be eliminated from the dataset; then the images are cut to make them  squared, and finally are resized so that the final images are of size ```[128,128]```.

## Image representation
Models for colorization will obviously work on images, therefore understanding how images are represented is crucial. The most common numeric representation for images, and the one used for the images of our datasets, is the **RGB**. This name comes from *Red-Green-Blue*, because the image is divided on three channels, one encoding color red, the second encoding green and the last one, blue. Each of these channels indicate the intensity of the corresponding color in the pixel in a scale ```[0,1]```.

However, RGB is not the best representation for Deep Colorization: **Lab** representation is more fitting for the problem. Lab also encode the image in three channels: **L**, in a scale ```[0,100]```, encodes the lightness and corresponds to the black and white image; and **a** and **b** stands for the color spectra green–red and blue–yellow and their values are in range ```[-128,128]```. This encoding is really usefull for the present problem, the L channel can be the input of the model and channels a and b are the ones that the model should learn to predict. 

## Running the code
Our code runs in ```pytorch```, so this and other packages should be downloaded before running the code. The needed packages are listed below:
* numpy
* torch
* torchvision
* pillow
* matplotlib
* skimage

These can be downloaded in windows using ```pip```:
```
pip install numpy
```
or in ```anaconda``` with the command:
```
conda install numpy
```

Now, the code can be runned in the machine. The project contains a ```main``` script that trains the model ```MODEL1``` with the dataset ```DATASET1``` in ```NUM_EPOCHS``` epochs and generate 10 images after that. This images are printed alongside with the original image so that the model can be qualitatively evaluated. To run this script the following command should be runned in the command line of the prefered terminal:
```
python main.py
```



## Contributors
Laia Escursell Rof - laia.escursellr@autonoma.cat
Abril Pérez Martí - abril.perezm@autonoma.cat

Xarxes Neuronals i Aprenentatge Profund
Grau de Computational Mathematics & Data analyitics, 
UAB, 2023
