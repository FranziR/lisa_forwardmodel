'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''

import numpy as np
from numpy import sin, cos, pi


def get_Euler(a, b, g):
    """
    Creates an rotation matrix from three given angles.

    :param a: (float)  first angle
    :param b: (float)  second angle
    :param g: (float)  third angle

    :returns:
        E (array):  3x3 rotation matrix
    """
    dim = a.shape[0]
    E = np.zeros((dim, 3, 3))
    E[:, 0, 0] = sin(a) * cos(b) - cos(a) * sin(g) * sin(b)
    E[:, 1, 0] = -cos(a) * cos(b) - sin(a) * sin(g) * sin(b)
    E[:, 2, 0] = cos(g) * sin(b)

    E[:, 0, 1] = -sin(a) * sin(b) - cos(a) * sin(g) * cos(b)
    E[:, 1, 1] = cos(a) * sin(b) - sin(a) * sin(g) * cos(b)
    E[:, 2, 1] = cos(g) * cos(b)

    E[:, 0, 2] = -cos(a) * cos(g)
    E[:, 1, 2] = -sin(a) * cos(g)
    E[:, 2, 2] = -sin(g)
    return E


if __name__ == '__main__':
    a = pi * np.linspace(0, 10, 11)
    b = -pi * np.linspace(0, 10, 11)
    c = 0 * np.linspace(0, 10, 11)
    A = get_Euler(a, b, c)
    print(A)
