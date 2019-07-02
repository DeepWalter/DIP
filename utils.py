# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


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


def bimshow(bimage, ticklabels='on', color='darkgray', figsize=None):
    """Plot the binary image on the grid.

    Parameters
    ----------
    bimage : np.ndarray
        The binary image to be plotted.
    ticklabels : {'on', 'off'}, optional
        Whether to show the tick labels (default to 'on', which shows
        tick labels on both axes).
    color : str, optional
        The color used to paint nonzero squares (default to 'darkgray').
    figsize : (float, float), optional
        Width, height in inches of the figure (default to None). For
        more detail, see plt.figure().
    """
    height, width = bimage.shape

    # Customize the color map.
    cmap = colors.ListedColormap(['white', color])
    bounds = [0, 1, 255]
    norm = colors.BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=figsize)

    ax.imshow(bimage, cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(-0.5, width, 1))  # set the major ticks
    ax.set_xticklabels('')  # hide the major ticklabels
    ax.set_yticks(np.arange(-0.5, height, 1))
    ax.set_yticklabels('')
    ax.grid(which='major', axis='both', linestyle='-', color='k', lw=0.8)

    assert ticklabels in ('on', 'off')
    if ticklabels == 'on':
        ax.xaxis.tick_top()
        xloc = np.arange(width)
        ax.set_xticks(xloc, minor=True)  # set the minor ticks
        ax.set_xticklabels(xloc, minor=True)  # set the minor ticklabels

        yloc = np.arange(height)
        ax.set_yticks(yloc, minor=True)
        ax.set_yticklabels(yloc, minor=True)


if __name__ == '__main__':
    image = np.zeros((20, 20), dtype=np.uint8)
    for i in range(20):
        for j in range(20):
            if (i - 10)**2 + (j - 10)**2 <= 25:
                image[i, j] = 1

    bimshow(image, figsize=(8, 8))
    plt.show()
