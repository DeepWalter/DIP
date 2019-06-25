import numpy as np


def equalize_histogram(image):
    """Compute the histogram equalization of the input image.

    Note: currently, it only supports transformations for grayscale
    images.

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


def gamma_correct(image, gamma=1.0):
    """Perform the gamma correction on image.

    Note: currently, it only supports transformations for grayscale
    images.

    Parameters
    ----------
    image: 2-dim ndarray
        The input grayscale image.
    gamma: float in [0, 1]
        Gamma value.
    Returns
    -------
    2-dim ndarray
        Gamma corrected image with the same shape as input image.
    """
    # Gamma transformation lookup talbe.
    L = 256
    lookup_table = np.arange(L, dtype=np.float)
    np.power(lookup_table / (L - 1), gamma, out=lookup_table)
    np.rint(lookup_table * (L - 1), out=lookup_table)
    lookup_table.astype(np.uint8, copy=False)

    return lookup_table[image]
