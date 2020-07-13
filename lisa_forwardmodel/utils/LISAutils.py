
def checkLink(link):
    try:
        list((-3, -2, -1, 1, 2, 3)).index(link)
        return True
    except ValueError:
        return False

def checkCraft(craft):
    try:
        list((1,2,3)).index(craft)
        return True
    except ValueError:
        return False

def get_link(detector = None, recv = None, send = None):
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
