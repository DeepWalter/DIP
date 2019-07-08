import numpy as np
import cv2 as cv


def equalize_histogram(image):
    """Generate an new image with a flat histogram from the given image.

    Note: currently, it only supports transformations for grayscale images.

    # TODO: Add support for normalization.

    Paramters
    ---------
    image: 2-dim ndarray of dtype uint8
        The input grayscale image.
    Returns
    -------
    2-dim ndarray of dtype uint8
        The histogram equalization of input image with the same shape
        and dtype.
    """
    L = 256  # number of intensity levels
    coefficient = (L - 1) / np.prod(image.shape)

    hist = np.histogram(image, bins=range(L + 1))[0]
    hist_cumsum = coefficient * np.cumsum(hist)
    equalization_map = np.rint(hist_cumsum).astype(np.uint8)

    return equalization_map[image]


def match_histogram(image, target):
    """Generate an new image with the specified histogram from the given image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.
    target : np.ndarray
        The specified histogram of shape (256,).
    """
    L = 256
    image_hist = np.histogram(image, bins=range(L + 1))[0]  # (256,)
    image_cumhist = np.expand_dims(np.cumsum(image_hist), axis=1)  # (256, 1)
    target_cumhist = np.expand_dims(np.cumsum(target), axis=0)  # (1, 256)

    abs_diff = np.abs(image_cumhist - target_cumhist)  # (256, 256)

    matching_map = np.argmin(abs_diff, axis=1)  # (256,)

    return matching_map[image]


def gamma_correct(image, gamma=1.0):
    """Perform the gamma correction on image.

    Note: currently, it only supports transformations for grayscale
    images.

    Parameters
    ----------
    image: 2-dim ndarray
        The input grayscale image.
    gamma: nonnegative float
        Gamma value.
    Returns
    -------
    2-dim ndarray
        Gamma corrected image with the same shape and dtype as input
        image.
    """
    # Sanity checks.
    if gamma < 0.0:
        raise ValueError('Gamma value should be non-negative')
    if gamma == 1.0:
        return image

    # Gamma transformation lookup talbe.
    L = 256
    lookup_table = np.arange(L, dtype=np.float)
    np.power(lookup_table / (L - 1), gamma, out=lookup_table)
    np.rint(lookup_table * (L - 1), out=lookup_table)
    lookup_table = lookup_table.astype(np.uint8)

    return lookup_table[image]


def bit_planes(image):
    """Return the bit planes of an image.

    Parameters
    ----------
    image : np.ndarray
        The input grayscale image.

    Returns
    -------
    np.ndarray
        8 bit planes. The first dimension indicates the order of bits
        with zero corresponding to the least significant bit.
    """
    bit_mask = 1
    planes = np.empty((8, *image.shape), dtype=np.uint8)

    for i in range(8):
        cv.bitwise_and(image, bit_mask, planes[i])
        planes[i] //= bit_mask
        bit_mask <<= 1

    return planes
