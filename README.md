# Master's Thesis
## "A Deep Convolutional Neural Network Based on U-Net to Predict Annual Luminance Maps"

**Mohammad Ali Qorbani, Farhad Dalirani, Mohammad Rahmati, Mohammad Reza Hafezi**

[**Link to the paper**](https://www.tandfonline.com/doi/full/10.1080/19401493.2021.2004229)

Currently, we are working on an application with a user-friendly UI to make this method available to architects, which will be released at https://github.com/maqorbani/DCNU_Lighting. However, in the meanwhile, you can access our code and dataset in this repository.

## Abstract

Studying annual luminance maps during the design process provides architects with insight into the space's spatial quality and occupants' visual comfort. Simulating annual luminance maps is computationally expensive, especially if the objective is to render the scene for multiple viewpoints. This repository is a method based on deep learning that accelerates these simulations by predicting the annual luminance maps using only a limited number of rendered high-dynamic-range images. The proposed model predicts HDR images that are comparable to the rendered ones. Using the transfer learning approach, this model can robustly predict HDR images from other viewpoints in the space with less rendered images (up to one-tenth) and less training required. This method was evaluated using various evaluation metrics, such as MSE, RER, PSNR, SSIM, and runtime duration and it shows improvements in all metrics compared to the previous [work](https://arxiv.org/abs/2009.09928), especially 33% better MSE loss, 48% more accurate DGP values, and 50% faster runtime.

## How it works?

In a standard image simulation procedure, the rendering program simulates the luminance-based HDR image by first acquiring the desired scene's information as input, i.e., the scene geometry, materials, and light sources. It starts ray tracing to determine how the emitted light rays illuminate each object in the scene by striking them, bouncing off, and eventually reaching the camera's sensor. However, in this method, a deep neural network is trained to predict luminance maps using only a limited number of Radiance simulated synthetic HDR images and then predicts the rest of the annual luminance maps without ray-tracing by providing each hour's corresponding lighting condition and the scene information.

## Steps for this method:
1.	Scene modeling
2.	Annual lighting conditions extraction
3.	Sparse samples selection
4.	Data set generation
5.	Training the neural network

An overview of our proposed method is presented in the following figure.
![Architecture-05](https://user-images.githubusercontent.com/47574645/140614071-b022bf5c-920e-4b72-b90a-8154f1703805.png)

### 1. Scene modeling

Modeling the scene could take place in any 3D CAD software. We recommend Rhinoceros since using the [Honeybee](https://www.ladybug.tools/honeybee.html) you can easily assign Radiance materials to the geometry and export a Radiance scene description. A Radiance scene desription with assigned materials is all you need for this step.

In this repository, we used the room presented in the following figure as our scene.
![image](https://user-images.githubusercontent.com/47574645/140609425-d34c3422-d81b-4032-a18d-8736af509391.png)


### 2.	Annual lighting conditions extraction

In this step, annual sky (daylighting) condition is needed. You should extract sun's altitude and azimuth and direct irradiance, and the sky's diffuse irradiance from your desired region climate (epw) file. We also recommend using the [Ladybug](https://www.ladybug.tools/honeybee.html) for this step. Then, a Radiance sky description should be created using Radiance GENDAYLIT command out of the extracted parameters.

### 3.	Sparse samples selection
Selecting sparse samples throughout the year could be carried out using the k-means.py script provided in the K-means directory.

After this step, the files are ready to be rendered using Radiance. Though, you could use this method on images created by any synthetic image rendering.

### 4.	Data set generation
Creating the data set for the neural network training process is done using the TensorMaker.py script provided in V2DataAnalysis directory given the corresponding scene references which should be put in the SceneRefrences directory in their corresponding name. 

### 5.	Training the neural network
The PyTorchConvReg.py script in V2DataAnalysis directory does the neural network training.

The following figure depicts the neural network's architecture used in this method.
![Architecture2 0_Artboard 1](https://user-images.githubusercontent.com/47574645/140615136-cb565395-6cc1-47c7-9423-5fdab97caad5.png)

## Dataset 
You can download the dataset [here](https://www.dropbox.com/sh/c5f0kdwsfnv95z7/AAAch4qmi2DISGWJyKTy5PM0a?dl=0)

Additionally, you can create your own dataset to employ this method in order to predict annual luminance maps. The scripts provided in [TheRender folder](https://github.com/maqorbani/Thesis/tree/master/TheRender) will help you achieve this goal. Moreover, using [RadianceEquirectangular](https://github.com/maqorbani/RadianceEquirectangular) repository, you can render equirectangular images using Radiance.
