'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''

import numpy as np

from lisa_forwardmodel.utils.checkInput import checkInput


def delete_zeros(list_ref, list_pred):
    """
    From two lists, indices of zero entries in list_ref are found and deleted from both lists.
    Note:
        input --> len(list_ref) == len(list_pred)
        output --> len(ref) == len(pred) BUT len(ref) =< len(list_ref) and len(pred) =< len(list_pred)

    :param list_ref: (list)     numerical list, from which all entries that are equal to zero should be deleted
    :param list_pred: (list)    numerical list, from which all entries are deleted which are in the same position
                                as the zero entries in list_ref

    :returns:
        ref (array):        modified input list list_ref (does not contain any zeros)
        pred (array):       modified input list list_pred (entries, that have been deleted from list_ref are also
                            deleted from list_pred)
    """
    # Check input data type and adapt if necessary
    ind1, temp1 = checkInput(list_ref, list, np.ndarray)
    if ind1:
        ref = temp1.copy()
    else:
        raise TypeError

    ind2, temp2 = checkInput(list_pred, list, np.ndarray)
    if ind2:
        prediction = temp2.copy()
    else:
        raise TypeError

    # find zero entries in reference list and remove them from both lists
    indices = [i for i, x in enumerate(ref) if x == 0]
    while len(indices) > 0:
        i = indices.pop(-1)
        ref.pop(i)
        prediction.pop(i)
    return np.array(ref), np.array(prediction)


def MAPE(list_ref, list_pre):
    """
    Computes the mean absolute percentage error between two lists.
    Output is a number.

    :param list_ref: (list)    list with reference values
    :param list_pre: (list)    list with values whose deviation wrt reference is to be computed

    :returns:
        MAPE (float):       mean absolute percentage error
    """
    ind1, temp1 = checkInput(list_ref, list, np.ndarray)
    if ind1:
        list_ref = temp1.copy()
    else:
        raise TypeError

    ind2, temp2 = checkInput(list_pre, list, np.ndarray)
    if ind2:
        list_pred = temp2.copy()
    else:
        raise TypeError

    ref, prediction = delete_zeros(list_ref, list_pred)
    if len(ref.shape) == 1:
        n = len(ref)
    else:
        n = ref.shape[1]
    return (1 / n) * np.sum(np.abs((prediction - ref) / ref))


def MSE(list1, list2):
    """
    Computes the mean squared error between two lists.
    Output is a number.

    :param list1: (list)   list with reference values
    :param list2: (list)   list with values whose deviation wrt reference is to be computed

    :returns:
        MSE (float):    mean squared error
    """
    ind1, temp1 = checkInput(list1, np.ndarray, list)
    if ind1:
        ref = temp1.copy()
    else:
        raise TypeError

    ind2, temp2 = checkInput(list2, np.ndarray, list)
    if ind2:
        prediction = temp2.copy()
    else:
        raise TypeError

    if len(ref.shape) == 1:
        n = len(ref)
    else:
        n = ref.shape[1]
    return np.sum((prediction - ref) ** 2) / n
