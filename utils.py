# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt


def imshow(image, position=None, title=None, axis='off', cmap='gray',
           interpolation=None):
    """Plot the image.

    Parameters
    ----------
    image: ndarray
        The image to be plotted.
    position: int or tuple of ints, optional
        The position of the subplot. See `pyplot.subplot()` for more
        details. If not None, create a subplot according to position and
        plot on it. (default to None)
    title: str, optional
        The title of the plot. (default to None, no title)
    axis: str, optional
        Axis options. See `pyplot.imshow()` function for more details.
    cmap: str, optional
        Color map options. (default to 'gray', grayscale colormap)
    interpolation: str, optional
        Interpolation options. (default to None, no interpolation is
        applied)
    """
    if position is not None:
        if isinstance(position, tuple):
            plt.subplot(*position)
        elif isinstance(position, int):
            plt.subplot(position)
        else:
            raise ValueError('position should be an int or tuple of ints')
    if axis is not None:
        plt.axis(axis)
    if title is not None:
        plt.title(title)

    kwargs = {'cmap': cmap}
    if interpolation is not None:
        kwargs['interpolation'] = interpolation
    plt.imshow(image, **kwargs)
