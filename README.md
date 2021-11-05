# Master's Thesis
## "A Deep Convolutional Neural Network Based on U-Net to Predict Annual Luminance Maps"

**Mohammad Ali Qorbani, Farhad Dalirani, Mohammad Rahmati, Mohammad Reza Hafezi**

**Accepted for Publication in the Jounral of Building Performance Simulation.**

Studying annual luminance maps during the design process provides architects with insight 
into the space's spatial quality and occupants' visual comfort. Simulating annual luminance maps 
is computationally expensive, especially if the objective is to render the scene for multiple viewpoints. 
This repository is a method based on deep learning that accelerates these simulations by predicting 
the annual luminance maps using only a limited number of rendered high-dynamic-range images. 
The proposed model predicts HDR images that are comparable to the rendered ones.
Using the transfer learning approach, this model can robustly predict HDR images from other 
viewpoints in the space with less rendered images (up to one-tenth) and less training required.
This method was evaluated using various evaluation metrics, such as MSE, RER, PSNR, SSIM, 
and runtime duration and it shows improvements in all metrics compared to the previous 
[work](https://arxiv.org/abs/2009.09928), 
especially 33% better MSE loss, 48% more accurate DGP values, and 50% faster runtime.

