# image_denoising_using_cnn
This exercise deals with neural networks and their application to image restora tion. In this exercise you will develop a general workflow for training networks to restore corrupted ima ges, and then apply this workflow on two different tasks: (i) image denoising, and (ii) image deblurring . 2 Background Before you start working on the exercise it is recommended that you review the lecture slides covering neural networks, and how to implement them using the Keras framework. To recap the relevant part of the lecture: there are many possible ways to use neural networks for image rest oration. The method you will implement consists of the following three steps:

Collect “clean” images, apply simulated random corruptions, and extract smal l patches.
Train a neural network to map from corrupted patches to clean patches.
Given a corrupted image, use the trained network to restore the complete ima ge by restoring each patch separately, by applying the “ConvNet Trick” for approximating this proces s as learned in class.
