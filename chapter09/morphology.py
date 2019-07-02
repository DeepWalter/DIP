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
            self.image = data

        self._origin = origin

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
            xmin, xmax = np.amin(data, axis=1), np.amax(data, axis=1)
            ymin, ymax = np.amin(data, axis=0), np.amax(data, axis=0)
            self._origin = (-ymin, -xmin)
            self._image = np.zeros((xmax-xmin, ymax-ymin), dtype=np.uint8)
            self._image[data + self.origin] = 1

    def to_image(self):
        """Convert the structuring element into an image."""
        pass

    def show(self):
        """Plot the structuring element."""
        pass


class Test:
    def __init__(self, value, first=True):
        self.x = x
        self.y = y

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value ** 2
        self._y =


if __name__ == '__main__':
    data = np.zeros((3, 3), dtype=np.uint8)
    data[:, 1] = 1
    data[1, :] = 1
    print(data)

    # mask = StructuringElement(data, origin=(1, 1))
    # print(mask.elems)

    t = Test(9, 6)
    print(t.x, t.y)
