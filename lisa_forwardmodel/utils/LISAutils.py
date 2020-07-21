'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''


def checkLink(link):
    """
    LISA links can only take certain values. This function checks, whether a given link index is valid.

    :param link: (int) link to be checked

    :returns:
        indicator (bool):   True, if link is valid; False otherwise
    """
    try:
        list((-3, -2, -1, 1, 2, 3)).index(link)
        return True
    except ValueError:
        return False


def checkCraft(craft):
    """
    LISA spacecraft indices can only take certain values.
    This function checks, whether a given craft index is valid.

    :param craft: (int)    spacecraft index to be checked

    :returns:
        indicator (bool):   True, if spacecraft index is valid; False otherwise
    """
    try:
        list((1, 2, 3)).index(craft)
        return True
    except ValueError:
        return False


def get_link(detector=None, recv=None, send=None):
    """
    Interspacecraft measurments can be derived from six combinations of
    sending/receiving spacecrafts and the associated transmitting link.
    For given sending and receiving crafts, this function returns the belonging link.

    :param detector:  (object)  SpaceInterferometer object
    :param recv:  (int)         index of receiving craft
    :param send: (int)          index of sending craft

    :returns:
        link (int):         link index of receiving/sending spacecraft pair
    """
    try:
        slr = detector.get_constellation()
    except AttributeError:
        slr = [[3, 1, 2], [1, 2, 3], [2, 3, 1], [2, -1, 3], [3, -2, 1], [1, -3, 2]]

    if recv is not None and send is not None:
        while len(slr) != 0:
            temp = slr.pop(0)
            if temp[0] == send and temp[-1] == recv:
                return temp[1]

        return ValueError
