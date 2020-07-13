import numpy as np

def checkInput(value, dest_type, opt_type):
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