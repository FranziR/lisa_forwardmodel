import numpy as np
from lisa_forwardmodel.utils.checkInput import checkInput


def delete_zeros(list_ref, list_pred):
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

    indices = [i for i, x in enumerate(ref) if x == 0]
    while len(indices) > 0:
        i = indices.pop(-1)
        ref.pop(i)
        prediction.pop(i)
    return np.array(ref), np.array(prediction)


def MAPE(list_ref, list_pre):
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
