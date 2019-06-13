import numpy as np


def equalize_histogram(image):
    """Compute the histogram equalization of the input image.

    Note: currently, it only supports transformations for grayscale
    images.
    #TODO: Add support for normalization.

    Paramters
    ---------
    image: ndarray of dimension 2
        The input grayscale image.
    Returns
    -------
    ndarray of dimension 2
        The histogram equalization of the input image.
    """
    L = 256  # number of intensity levels
    coefficient = (L - 1) / np.product(image.shape)

    hist = np.histogram(image, bins=range(L + 1))[0]
    hist_cumsum = coefficient * np.cumsum(hist)
    equalization_map = np.rint(hist_cumsum).astype(np.uint8)

    return equalization_map[image]
