'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''

import numpy as np

def checkInput(value, dest_type, opt_type):
    """
    This function checks the data type of a given item.
    The item is allowed to be of further data types (opt_type). If this is the case, the function automatically
    converts the item into the desired data type.

    :param value:      item whose data type has to be checked and adapted if necessary
    :param dest_type:  python data type, which the item should have
    :param opt_type:   list of optional data types which the item could possibly have; if so, the item is converted to dest_type

    :returns:
        indicator (bool):   indicates whether returned item has correct data type
        value (dest_type):  item of correct data type (if check/conversion was successful), 0 if not
    """
    if value is not None:
        if isinstance(value, dest_type):
            return (True, value)
        elif isinstance(value, opt_type):
            if dest_type == int:
                return (True, int(value))
            elif dest_type == float:
                return (True, float(value))
            elif dest_type == bool:
                return (True, bool(value))
            elif dest_type == str:
                return (True, str(value))
            elif dest_type == list:
                if isinstance(value, (int, float)):
                    return (True, list([value]))
                else:
                    return (True, list(value))
            elif dest_type == tuple:
                return (True, tuple(value))
            elif dest_type == np.ndarray:
                return (True, np.array(value))
        else:
            return (False, 0)
    else:
        return (False, 0)

if __name__ == '__main__':
    #t = (1,2,3)
    t = None
    indicator, t_new = checkInput(t, np.ndarray, (list, tuple))
    print(type(t_new))
    print(t_new)