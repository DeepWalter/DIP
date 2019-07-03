import numpy as np


class StructuringElement:
    """Structuring element.

    Parameters
    ----------
    data : np.ndarray
        Image representation or coordinates representation of the
        structuring element.
    origin : (int, int), optional
        Coordinates of the origin.
    coordinates : bool, optional
        Whether the data is an image representation or coordinates
        representation of the structuring element.

    Attributes
    ----------
    origin : (int, int)
        Coordinates of the origin.
    image : np.ndarray
        Image representation of this structuring element.
    elems : np.ndarray of shape (*, 2)
        Coordinates representation of this structuring element.
    """

    def __init__(self, data, origin=None, coordinates=False):
        if coordinates:
            self.elems = data
        else:
            self._origin = origin
            self.image = data

    @property
    def image(self):
        return self._image

    @image.setter
    def image(self, data):
        self._image = data
        self._elems = np.transpose(np.nonzero(data)) - self._origin

    @property
    def elems(self):
        return self._elems

    @elems.setter
    def elems(self, data):
        self._elems = data
        xmin, xmax = np.amin(data[:, 1]), np.amax(data[:, 1])
        ymin, ymax = np.amin(data[:, 0]), np.amax(data[:, 0])
        print(xmin, xmax)
        self._origin = (-ymin, -xmin)
        self._image = np.zeros((xmax-xmin+1, ymax-ymin+1), dtype=np.uint8)
        x = (data + self._origin)[:, 0]
        y = (data + self._origin)[:, 1]
        self._image[x, y] = 1

    def to_image(self):
        """Convert the structuring element into an image."""
        pass

    def show(self):
        """Plot the structuring element."""
        pass


def bimage2set(image, origin=None):
    """Convert a binary image to the set of coordinates."""
    pass


def set2bimage(set):
    pass


if __name__ == '__main__':
    # data = np.zeros((3, 3), dtype=np.uint8)
    # data[:, 1] = 1
    # data[1, :] = 1
    # print(data)

    data = np.array([[-1, 0],
                     [0, -1],
                     [0, 0],
                     [1, 0],
                     [0, 1]])

    mask = StructuringElement(data, origin=(1, 1), coordinates=True)
    print(mask.image)
