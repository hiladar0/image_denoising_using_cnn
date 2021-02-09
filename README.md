# image_denoising_using_cnn
This exercise deals with neural networks and their application to image restora tion. In this exercise you will develop a general workflow for training networks to restore corrupted ima ges, and then apply this workflow on two different tasks: (i) image denoising, and (ii) image deblurring . 

Collect “clean” images, apply simulated random corruptions, and extract smal l patches.
Train a neural network to map from corrupted patches to clean patches.
Given a corrupted image, use the trained network to restore the complete ima ge by restoring each patch separately, by applying the “ConvNet Trick” for approximating this proces s as learned in class.
