import numpy as np


def bimage2set(image, origin=None):
    """Convert a binary image into a set of coordinates.

    Only the coordinates of those non-zero pixels are collected. And
    their coordinates are computed w.r.t. the given origin. That is, if
    a non-zero pixel has indices `(i, j)` in the image, then its
    coordinates w.r.t. origin is `(i, j) - origin` where `origin` is
    the numeric form of origin.

    Parameters
    ----------
    image : np.ndarray
        Input binary image.
    origin : (int, int), optional
        Coordinates of the origin (default to None, which means (0, 0)).

    Returns
    -------
    np.ndarray
        The set of all non-zero pixels' coordinates, stored as a ndarray
        of shape (*, 2).

    See Also
    --------
    set2bimage: convert a set of coordinates to an image.

    Note
    ----
    May add support for collecting zero pixels.
    """
    # TODO: Add more options for origin, e.g. 'center'
    if origin is None:
        origin = (0, 0)

    return np.transpose(np.nonzero(image)) - origin


def set2bimage(set):
    """Convert a set of coordinates into an image.

    This is the inverse function of `bimage2set()`. The resulting image
    is the smallest rectangular image in which all coordinates in the
    set fit according to their relative positions. The coordinates
    correspond to the non-zero pixels in the resulting image. The
    indices of the origin is also returned so that the original set of
    coordinates can be recovered from them and the resulting image.

    Parameters
    ----------
    set : np.ndarray
        The set of coordinates with shape (*, 2).

    Returns
    -------
    image : np.ndarray
        The corresponding image.
    origin : (int, int)
        Indices of the origin in the resulting image.

    See Also
    --------
    bimage2set : Inverse function.
    """
    ymin, xmin = np.amin(set, axis=0)
    ymax, xmax = np.amax(set, axis=0)

    origin = (-ymin, -xmin)
    image = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    x, y = np.hsplit(set + origin, 2)
    image[x, y] = 1

    return image, origin


if __name__ == '__main__':
    pass
