import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
from skimage.color import rgb2gray
from scipy.ndimage.filters import convolve
import denoise_helper
from tensorflow.keras.layers import Input, Conv2D, Activation, Add
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

PIXEL_INTENSITIES = 256
GRAY_MODE = 1

# NEURAL-NETWORK IMPLEMENTATION
SUBTRACTION_FACTOR = 0.5
KERNEL_SIZE = (3, 3)
LOSS = 'mean_squared_error'
VALIDATION_RATIO = 0.2
TRAIN_RATIO = 1 - VALIDATION_RATIO

# IMAGE DENOISING TASK
PATCH_WIDTH_DNS = 24
PATCH_HEIGHT_DNS = 24
NUM_CHANELS_DNS = 48
MIN_SIGMA = 0
MAX_SIGMA = 0.2
# image denoising task regular mode
BATCH_SIZE_DNS = 100
STEPS_PER_EPOCH_DNS = 100
NUM_EPOCHS_DNS = 5
NUM_VALID_SAMPLES_DNS = 1000
# image denoising task quick mode
BATCH_SIZE_DNS_qm = 10
STEPS_PER_EPOCH_DNS_qm = 3
NUM_EPOCHS_DNS_qm = 2
NUM_VALID_SAMPLES_DNS_qm = 30

# IMAGE DEBLURING TASK
PATCH_WIDTH_BLR = 16
PATCH_HEIGHT_BLR = 16
NUM_CHANELS_BLR = 32
# image denoising task regular mode
BATCH_SIZE_BLR = 100
STEPS_PER_EPOCH_BLR = 100
NUM_EPOCHS_BLR = 10
NUM_VALID_SAMPLES_BLR = 1000
# image denoising task quick mode
BATCH_SIZE_BLR_qm = 10
STEPS_PER_EPOCH_BLR_qm = 3
NUM_EPOCHS_BLR_qm = 2
NUM_VALID_SAMPLES_BLR_qm = 30
MAX_ANGLE = np.pi


def read_image(filename, representation):
    """
    reads an image file and converts it into a given representation.
    :param filename:        string containing the image filename to read.
    :param representation:  representation code, either 1 or 2 defining whether
                            the output should be a greyscale image (1) or an
                            RGB image (2).
    :return:                returns an image represented by a matrix of type
                            .float64 with intensities normalized to the
                            range [0,1]
    """
    im = imread(filename)
    im_float = im.astype(np.float64)
    if (representation == GRAY_MODE):
        im_float = rgb2gray(im_float)
    return im_float / PIXEL_INTENSITIES


def data_normalize(im):
    """
    Args:
        im: grayscale image (np.array) in the [0,1] range , type float
    Returns: np.array in the [-SUBTRACTION_FACTOR,1-SUBTRACTION_FACTOR] range, type float

    """
    return im.astype(np.float) - SUBTRACTION_FACTOR

def data_unnormalize(im):
    """
    Args:
        im: grayscale image (np.array) in the [-SUBTRACTION_FACTOR,1-SUBTRACTION_FACTOR] range , type float
    Returns: np.array in the [0,1] range, type float

    """
    return im.astype(np.float) + SUBTRACTION_FACTOR


def crop_image(im, corrupt_im, crop_size):
    # Randomly choosing the location of a patch the size of crop_size
    height, width = im.shape
    patch_height, patch_width = crop_size
    patch_x = np.random.randint(0, height - patch_height)
    patch_y = np.random.randint(0, width - patch_width)

    # Restricting the image to the chosen patch
    patch_im = im[patch_x:patch_x + patch_height,
               patch_y: patch_y + patch_width]
    patch_corrupt_im = corrupt_im[patch_x:patch_x + patch_height,
                       patch_y: patch_y + patch_width]
    return patch_corrupt_im,patch_im


def load_dataset(filenames, batch_size, corruption_func, crop_size):
    """
    A Generator of Training-Data Batches for the NN.
    Args:
        filenames: A list of filenames of clean images
        batch_size: The size of the batch of images for each iteration of Stochastic Gradient Descent
        corruption_func: A function receiving a numpy’s array representation of an image as a single argument,
        and returns a randomly corrupted version of the input image.
        crop_size: A tuple (height, width) specifying the crop size of the patches to extract.

    Returns: a Python’s generator object which outputs random tuples of the form
            (source_batch, target_batch), where each output variable is an array of
            shape (batch_size, height,width, 1)
    """
    cache = {}
    patch_height, patch_width = crop_size
    while True:
        size = (batch_size, patch_height, patch_width, 1)
        source_batch = np.zeros(size).astype(np.float64)
        target_batch = np.zeros(size).astype(np.float64)
        for i in range(batch_size):
            # Randomly picks an image and make a corrupted version of  it
            rand_fname = np.random.choice(filenames)
            if rand_fname not in cache:
                im = read_image(rand_fname, GRAY_MODE)
                cache[rand_fname] = im
            im = cache[rand_fname]
            corrupt_im = corruption_func(im)
            source_patch, target_patch = crop_image(im, corrupt_im, crop_size)
            source_batch[i, :, :, :] = data_normalize(source_patch)[..., np.newaxis]
            target_batch[i, :, :, :] = data_normalize(target_patch)[..., np.newaxis]
        yield (source_batch, target_batch)

def resblock(input_tensor, num_channels):
    """
    Creating a residual block of the ResNet Neural Network for image restoration
    Args:
        input_tensor: A symbolic input tensor
        num_channels: The number of channels for each of its convolutional layers.
    Returns: The symbolic output tensor of the residual block.

    """
    b = Conv2D(num_channels, KERNEL_SIZE, padding='same')(input_tensor)
    b = Activation('relu')(b)
    b = Conv2D(num_channels, KERNEL_SIZE, padding='same')(b)
    add = Add()([input_tensor, b])
    return Activation('relu')(add)


def build_nn_model(height, width, num_channels, num_res_blocks):
    """
    Creating a ResNet Neural Network model for image restoration
    Args:
        height,width: The dimensions of the input image.
        num_channels: number of output channels for all  the convolutional layers
        num_res_blocks: Number of residual block

    Returns: a Neural Network

    """
    input = Input(shape=(height, width, GRAY_MODE))
    out = Conv2D(num_channels, KERNEL_SIZE, padding='same')(input)
    out = Activation('relu')(out)
    resblk = out
    for i in range(num_res_blocks):
        resblk = resblock(resblk, num_channels)

    out = Conv2D(1, KERNEL_SIZE, padding='same')(resblk)
    out = Add()([input, out])
    return Model(inputs=input, outputs=out)


def train_model(model, images, corruption_func, batch_size, steps_per_epoch, num_epochs, num_valid_samples):
    """
    Train the Neural Network on a given training set.
    Args:
        model: a general neural network model for image restoration
        images: a list of file paths pointing to image files
        corruption_func: a function receiving a numpy’s array representation of an image as a single argument,
        and returns a randomly corrupted version of the input image
        batch_size: the size of the batch of examples for each iteration of SGD
        steps_per_epoch: the number of update steps in each epoch
        num_epochs: the number of epochs for which the optimization will run
        num_valid_samples: the number of samples in the validation set to test on after every epoch

    """
    # Divide the images into a training set and validation set
    train_index = int((TRAIN_RATIO) * len(images))
    train_set = images[:train_index]
    validation_set = images[train_index:]

    # Creates train data generator
    crop_size = (model.input_shape[1], model.input_shape[2])
    train_data_generator = load_dataset(train_set, batch_size, corruption_func, crop_size)
    # Creates validation data generator
    validation_generator = load_dataset(validation_set, batch_size, corruption_func, crop_size)

    model.compile(loss=LOSS,
                  optimizer=Adam(beta_2=0.9))
    history = model.fit_generator(generator=train_data_generator,
                                  validation_data=validation_generator, epochs=num_epochs,
                                  steps_per_epoch=steps_per_epoch, validation_steps=num_valid_samples,
                                  use_multiprocessing=True)
    return history


def restore_image(corrupted_image, base_model):
    """
    Restores corrupted image using Res-Net neural network
    Args:
        corrupted_image: a grayscale image of shape (height, width) and with values in
        the [0, 1] range of type float64
        base_model: a neural network trained to restore small patches

    Returns: restored image, with values in the [0, 1] range of type float64

    """
    # adjust the base model to the dimensions of the corrupted image
    input_tensor = Input(corrupted_image[..., np.newaxis].shape)
    model = base_model(input_tensor)
    new_model = Model(inputs=input_tensor, outputs=model)
    # use the model to restore the corrupted image.
    # The input and output of the network are images with values in the [−0.5, 0.5]
    # range, so we need to preprocess the image and also match it's shape  
    X = data_normalize(corrupted_image)[np.newaxis, :, :, np.newaxis]
    restored_image = new_model.predict(X)
    restored_image = restored_image.reshape(corrupted_image.shape)
    return np.clip(data_unnormalize(restored_image), 0, 1)


def normalize_image(image):
    """
    round the value of each pixel to the nearest fraction i/255 and then clip to [0, 1]
    Args:
        image: a grayscale image, in type float
    Returns: a grayscale image, in type float, in range [0,1]

    """
    im = np.round(image * 255) / 255
    return np.clip(im, 0, 1).astype(np.float64)


def add_gaussian_noise(image, min_sigma, max_sigma):
    """
    Add random noise to a given image
    Args:
        image: a grayscale image with values in the [0, 1] range of type float64
        min_sigma, max_sigma: a non-negative scalar values representing the minimal/maximal
         variance of the gaussian distribution
    Returns: a noisy  grayscale image, in the [0, 1] range of type float64

    """
    # randomly sample a value of sigma
    sigma = np.random.uniform(min_sigma, max_sigma)
    # ay adding to every pixel of the input image a zero-mean gaussian
    # random variable with standard deviation equal to sigma
    noisy_img = image + np.random.normal(0.0, sigma, size=image.shape)
    return normalize_image(noisy_img)


def learn_denoising_model(num_res_blocks=5, quick_mode=False):
    """
    Args:
        num_res_blocks: Number of residual block, For the Neural Network model
        quick_mode: A boolean variable, True if we want the network no train faster.

    Returns: A trained denoising Neural Network model
    """
    min_sig, max_sig = MIN_SIGMA, MAX_SIGMA
    patch_height, patch_width, channels = PATCH_HEIGHT_DNS, PATCH_WIDTH_DNS, NUM_CHANELS_DNS
    batch_size, steps_per_epochs, num_epochs, num_valid_samples = BATCH_SIZE_DNS, STEPS_PER_EPOCH_DNS, NUM_EPOCHS_DNS, NUM_VALID_SAMPLES_DNS
    if quick_mode:
        batch_size, steps_per_epochs, num_epochs, num_valid_samples = BATCH_SIZE_DNS_qm, STEPS_PER_EPOCH_DNS_qm, NUM_EPOCHS_DNS_qm, NUM_VALID_SAMPLES_DNS_qm

    filenames = sol5_utils.images_for_denoising()
    img_denoising_model = build_nn_model(patch_height, patch_width, channels, num_res_blocks)
    history = train_model(img_denoising_model, filenames, lambda img: add_gaussian_noise(img, min_sig, max_sig),
                          batch_size, steps_per_epochs, num_epochs, num_valid_samples)
    #return history, img_denoising_model
    return img_denoising_model


def add_motion_blur(image, kernel_size, angle):
    """
     simulate motion blur on a given image, using convolution with a square kernel.
    Args:
        image: a grayscale image with values in the [0, 1] range of type float64
        kernel_size: non zero integer representing the size of the kernel
        angle: angle in radians, measured relative to the positive horizontal axis
    Returns: blurry grayscale image, in the [0, 1] range of type float64

    """
    return convolve(image, sol5_utils.motion_blur_kernel(kernel_size, angle))


def random_motion_blur(image, list_of_kernel_sizes):
    """
    randomly blur a given image
    Args:
        image: a grayscale image with values in the [0, 1] range of type float64
        list_of_kernel_sizes: blurry grayscale image, in the [0, 1] range of type float64
    Returns: blurry grayscale image, in the [0, 1] range of type float64

    """
    # randomly choose a kernel size from the given list
    kernel_size = np.random.choice(list_of_kernel_sizes)
    # randomly chosoe an angle
    angle = np.random.uniform(0, MAX_ANGLE)
    return normalize_image(add_motion_blur(image, kernel_size, angle))


def learn_deblurring_model(num_res_blocks=5, quick_mode=False):
    """
    Args:
        num_res_blocks: Number of residual block, For the Neural Network model
        quick_mode: A boolean variable, True if we want the network no train faster.

    Returns: A trained deblurring Neural Network model
    """
    kernel_sizes = [7]
    patch_height, patch_width, channels = PATCH_HEIGHT_BLR, PATCH_WIDTH_BLR, NUM_CHANELS_BLR
    batch_size, steps_per_epochs, num_epochs, num_valid_samples = BATCH_SIZE_BLR, STEPS_PER_EPOCH_BLR, NUM_EPOCHS_BLR, NUM_VALID_SAMPLES_BLR
    if quick_mode:
        batch_size, steps_per_epochs, num_epochs, num_valid_samples = BATCH_SIZE_BLR_qm, STEPS_PER_EPOCH_BLR_qm, NUM_EPOCHS_BLR_qm, NUM_VALID_SAMPLES_BLR_qm

    filenames = sol5_utils.images_for_deblurring()
    img_deblurring_model = build_nn_model(patch_height, patch_width, channels, num_res_blocks)
    history = train_model(img_deblurring_model, filenames, lambda img: random_motion_blur(img, kernel_sizes),
                          batch_size,
                          steps_per_epochs, num_epochs, num_valid_samples)

    #return history, img_deblurring_model
    return img_deblurring_model


# def effect_of_depth_denoise(quick_mode):
#     # get model errors for all res_blocks
#     res_blocks = np.arange(1, 6)
#     validation_errors = []
#     for num_res_blocks in res_blocks:
#         history, _ = learn_denoising_model(num_res_blocks=num_res_blocks, quick_mode=quick_mode)
#         validation_errors.append(history.history['val_loss'][-1])
#     plt.figure()
#     plt.plot(res_blocks, validation_errors)
#     plt.xticks(res_blocks)
#     plt.xlabel('num_res_blocks')
#     plt.ylabel('validation error for denoise model')
#     plt.savefig('depth_plot_denoise.png')
# 
# 
# def effect_of_depth_deblur(quick_mode):
#     # get model errors for all res_blocks
#     res_blocks = np.arange(1, 6)
#     validation_errors = []
#     for num_res_blocks in res_blocks:
#         history, _ = learn_deblurring_model(num_res_blocks=num_res_blocks, quick_mode=quick_mode)
#         validation_errors.append(history.history['val_loss'][-1])
#     plt.figure()
#     plt.plot(res_blocks, validation_errors)
#     plt.xticks(res_blocks)
#     plt.xlabel('num_res_blocks')
#     plt.ylabel('validation error for deblur model')
#     plt.savefig('depth_plot_deblur.png')
# 
# 
# if __name__ == "__main__":
#     effect_of_depth_denoise(False)
#     effect_of_depth_deblur(False)
# 
