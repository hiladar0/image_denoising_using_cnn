import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.signal
import skimage.color
from imageio import imread

RGB_SHAPE = 3
NORMALIZED = 'float64'
MAX_VAL = 1
GRAY_SCALE = 1
MAX_SHADE_NUM = 255.0
BASE_VEC = [1, 1]
ROW_INDEX = 0
COL_INDEX = 1
MINIMAL_DIMENSION = 16
G1_INDEX = 1
LAST_GAUSSIAN_INDEX = -1
ORIGINAL_IMG_INDEX = 0
SAMPELING_RATIO = 2
EMPTY_PYRAMID = 0
MIN_VAL = 0
DEFAULT_COEFF = [1]
GRAY_MODE = 'gray'
PYRAMID_INDEX = 0
RGB = 2
IM1_EX1 = "externals/slugf2.jpg"
IM2_EX1 = "externals/mudf2.jpg"
MASK_EX1 = "externals/maskf2.jpg"
MAX_LEVELS_EX1 = 5
FILTER_SIZE_EX1 = 13
IM1_EX2 = "externals/whale22.jpg"
IM2_EX2 = "externals/mef22.jpg"
MASK_EX2 = "externals/mask22.jpg"
MAX_LEVELS_EX2 = 31
FILTER_SIZE_EX2 = 13


def is_rgb(image):
    """
    :param img: image as a np matrix
    :return: true if it is a RGB image
    """
    return len(image.shape) == RGB_SHAPE


def is_need_representation_change(image, representation):
    """
    checks if a representation change is needed
    :param image: image
    :param representation: 1 if grayScale 2 if RGB
    :return: true if a representation change is needed, otherwise returns false
    """
    return is_rgb(image) and representation == GRAY_SCALE


def normalize_img(image):
    """
    :param image: not normalized image of dtype np.uint8
    :return: normalized image with values [0:1] of dtype np.float64
    """
    image = image.astype(np.float64)
    return image / MAX_SHADE_NUM


def read_image(filename, representation):
    """
    reads image from given file
    :param filename: file name of image
    :param representation: 1 if grayScale 2 if RGB
    :return: a normalized picture dtype float64, in given representation
    """
    image = imread(filename)
    if (is_need_representation_change(image, representation)):
        return skimage.color.rgb2gray(image)
    if (image.dtype == NORMALIZED and np.max(image) <= MAX_VAL):
        return image
    return normalize_img(image)


############################################################################

def build_filter_vec(filter_size):
    '''
    builds filter vector in the given size
    :param filter_size: the size of the vector
    :return: the normalized filter vector
    '''
    if (filter_size == 1):
        im_filter = scipy.signal.convolve(BASE_VEC, BASE_VEC, mode='valid')
    else:
        im_filter = BASE_VEC
        for i in range(filter_size - 2):
            im_filter = scipy.signal.convolve(im_filter, BASE_VEC)
    return np.asarray([im_filter / sum(im_filter)])


def convolve_img(im, filter_vec):
    '''
    calculates the convolution of the given image with the given vector.
    :param im: a grayscale image
    :param filter_vec: the filter vector
    :return: the image after convolution
    '''
    im = scipy.ndimage.filters.convolve(im, filter_vec)
    return scipy.ndimage.filters.convolve(im,
                                          filter_vec.transpose())


def reduce(im, filter_vec):
    '''
    reduces the size of given image
    :param im: the image to be reduced
    :param filter_vec: a filter vector to be applied
    :return: the reduced image
    '''
    return convolve_img(im, filter_vec)[::SAMPELING_RATIO, ::SAMPELING_RATIO]


def build_gaussian_pyramid(im, max_levels, filter_size):
    '''
    builds a gaussian pyramid from given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    :return: pyr- as a standard python array where each element of the array is
    a grayscale image, filter_vec- row vector of shape (1, filter_size) used
    for the pyramid construction
    '''
    filter_vec = build_filter_vec(filter_size)
    pyr = []
    lower_dim = ROW_INDEX if im.shape[ROW_INDEX] < im.shape[COL_INDEX] else \
        COL_INDEX
    if max_levels == EMPTY_PYRAMID or im.shape[lower_dim] < MINIMAL_DIMENSION:
        return pyr, filter_vec
    pyr.append(im)
    for i in range(max_levels - 1):
        im = reduce(im, filter_vec)
        if im.shape[lower_dim] < MINIMAL_DIMENSION:
            break
        pyr.append(im)
    return pyr, filter_vec


def expand(im, filter_vec):
    '''
    expands the size of given image
    :param im: the image to be expanded
    :param filter_vec: a filter vector to be applied
    :return: the expanded image
    '''
    new_im = np.zeros((im.shape[ROW_INDEX] * SAMPELING_RATIO,
                       im.shape[COL_INDEX] * SAMPELING_RATIO))
    new_im[::SAMPELING_RATIO, ::SAMPELING_RATIO] = im
    return convolve_img(new_im, filter_vec * SAMPELING_RATIO)


def build_laplacian_pyramid(im, max_levels, filter_size):
    '''
    builds a laplacian pyramid from given image.
    :param im: a grayscale image with double values in [0, 1]
    :param max_levels: the maximal number of levels in the resulting pyramid
    :param filter_size: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) to be used in constructing the pyramid filter
    :return: pyr- as a standard python array where each element of the array is
    a grayscale image, filter_vec- row vector of shape (1, filter_size) used
    for the pyramid construction
    '''
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(pyr) - 1):
        pyr[i] = pyr[i] - expand(pyr[i + 1], filter_vec)
    return pyr, filter_vec


def laplacian_to_image(lpyr, filter_vec, coeff):
    '''
    builds the image from its laplacian pyramid
    :param lpyr: laplacian pyramid
    :param filter_vec: the vector was used for the building of the pyramid
    :param coeff: a python list of the coefficients
    :return: the original image filtered according to the coefficients
    '''
    for i in range(len(coeff)):
        lpyr[i] = lpyr[i] * coeff[i]
    for i in range(len(lpyr) - 1, 0, -1):
        lpyr[i] = expand(lpyr[i], filter_vec)
        lpyr[i - 1] = lpyr[i] + lpyr[i - 1]
    return lpyr[ORIGINAL_IMG_INDEX]


def render_pyramid(pyr, levels):
    '''
        res[:heights[i], offset:widths[i] + offset] = pyr[i]
        offset += heights[i]
    return res


def displayPyramid(pyr, levels):
    '''
    shows the rendered pyramid
    :param pyr: a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result ≤ max_levels
    '''
    res = render_pyramid(pyr, levels)
    plt.figure()
    plt.imshow(res, GRAY_MODE)
    plt.show()


def pyramidBlending(im1, im2, mask, maxLevels, filterSizeIm, filterSizeMask):
    '''
    blends in1 and im2
    :param im1:first grayscale images to be blended
    :param im2:second grayscale images to be blended
    :param mask: a binary mask containing 1’s and 0’s representing which
    parts of im1 and im2 should appear in the resulting imBlend.
    :param maxLevels:the maxLevels parameter using for generating
    the Gaussian and Laplacian pyramids.
    :param filterSizeIm: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) which defining the filter used in the
    construction of the Laplacian pyramids of im1 and im2.
    :param filterSizeMask:  the size of the Gaussian filter(an odd scalar that
    represents a squared filter) which defining the filter used in the
    construction of the Gaussian pyramid of mask.
    :return: the blended image
    '''
    l1, filter_vec = build_laplacian_pyramid(im1, maxLevels, filterSizeIm)
    l2 = build_laplacian_pyramid(im2, maxLevels, filterSizeIm)[PYRAMID_INDEX]
    gmask = \
        build_gaussian_pyramid(mask.astype(np.float64), maxLevels,
                               filterSizeMask)[
            PYRAMID_INDEX]
    lout = [EMPTY_PYRAMID] * len(l1)
    for i in range(len(l1)):
        lout[i] = (gmask[i] * l1[i]) + ((1 - gmask[i]) * l2[i])
    blended_img = laplacian_to_image(lout, filter_vec, DEFAULT_COEFF * len(l1))
    return np.clip(blended_img, MIN_VAL, MAX_VAL)
    renders a pyramid to display the number of levels of it in one image
    :param pyr: a Gaussian or Laplacian pyramid
    :param levels: the number of levels to present in the result ≤ max_levels
    :return: single black image in which the pyramid levels of the given
    pyramid pyr are stacked horizontally (after stretching the values to [0, 1])
    '''
    pyr = pyr[:levels]
    widths, heights = zip(*(i.shape for i in pyr))

    total_width = sum(widths)
    height = heights[ORIGINAL_IMG_INDEX]
    res = np.zeros((height, total_width))
    for i in range(len(pyr)):
        pyr[i] = (pyr[i] - np.amin(pyr[i])) / (
        np.amax(pyr[i]) - np.amin(pyr[i]))
    offset = 0
    for i in range(len(pyr)):


def relpath(filename):
    '''
    finds the relative path of file
    :param filename: file
    :return: the relative path of file
    '''
    return os.path.join(os.path.dirname(__file__), filename)


def blending_ex_helper(im1_directory, im2_directory, mask_directory,
                       max_levels_num, filterSizeIm, filterSizeMask):
    '''
    blends im1 and im2
    :param im1_directory:first RGB images to be blended
    :param im2_directory:second RGB images to be blended
    :param mask_directory: a binary mask containing 1’s and 0’s representing which
    parts of im1 and im2 should appear in the resulting imBlend.
    :param max_levels_num:the maxLevels parameter using for generating
    the Gaussian and Laplacian pyramids.
    :param filterSizeIm: the size of the Gaussian filter (an odd scalar that
    represents a squared filter) which defining the filter used in the
    construction of the Laplacian pyramids of im1 and im2.
    :param filterSizeMask:  the size of the Gaussian filter(an odd scalar that
    represents a squared filter) which defining the filter used in the
    construction of the Gaussian pyramid of mask.
    :return: im1, im2, mask, blended image
    '''
    im1 = read_image(relpath(im1_directory), RGB)
    im2 = read_image(relpath(im2_directory), RGB)
    mask = read_image(relpath(mask_directory), RGB)
    output = np.zeros(im1.shape)
    for i in range(RGB_SHAPE):
        output[:, :, i] = pyramidBlending(im2[:, :, i], im1[:, :, i],
                                          mask[:, :, i],
                                          max_levels_num, filterSizeIm,
                                          filterSizeMask)

    # fig, (ax1,ax2,ax3,ax4) = plt.subplots(3,1)
    # ax1.plot(im1)
    # ax2.plot(im2)
    # ax3.plot(output)
    # ax4.plot(mask)

    return im1, im2, mask, output


def blending_example1():
    '''
    displays the first example of blending images
    :return: im1, im2, mask, blended image
    '''
    return blending_ex_helper(IM1_EX1, IM2_EX1, MASK_EX1, MAX_LEVELS_EX1,
                              FILTER_SIZE_EX1, FILTER_SIZE_EX1)


def blending_example2():
    '''
    displays the second example of blending images
    :return: im1, im2, mask, blended image
    '''
    return blending_ex_helper(IM1_EX2, IM2_EX2, MASK_EX2, MAX_LEVELS_EX2,
                              FILTER_SIZE_EX2, FILTER_SIZE_EX2)
blending_example1()