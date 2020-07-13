import os
import sys
import h5py

import numpy as np
import math
from math import pi as pi

def read_HDF5_func(filepath):
    if not os.path.isfile(filepath):
        print("File path {} does not exist. Exiting...".format(filepath))
        sys.exit()

    with h5py.File(filepath) as fp:
        data = {}
        GWSources = list(fp[list(fp.keys())[0]]['GWSources'])
        for i in range(len(GWSources)):
            src_temp = list(fp[list(fp.keys())[0]]['GWSources'])[i]
            data_temp = list(fp[list(fp.keys())[0]]['GWSources'][src_temp])
            data[src_temp] = dict()
            for j in list(data_temp):
                #key_temp = data_temp[j]
                if j.find("hphc") > -1:
                    if src_temp.find('Gal') > -1:
                        hphc = np.array(list(fp[list(fp.keys())[0]]['GWSources'][src_temp][j]))
                        data[src_temp]['t'] = hphc[0,0,:]
                        data[src_temp]['hp'] = hphc[1, 0, :]
                        data[src_temp]['hc'] = hphc[2, 0, :]
                    elif src_temp.find('MBHB') > -1:
                        hphc = np.array(fp[list(fp.keys())[0]]['GWSources'][src_temp][j])
                        data[src_temp]['t'] = hphc[:,0]
                        data[src_temp]['hp'] = hphc[:,1]
                        data[src_temp]['hc'] = hphc[:,2]
                else:
                    try:
                        data[src_temp][j] = list(fp[list(fp.keys())[0]]['GWSources'][src_temp][j])[:]
                    except TypeError:
                        data[src_temp][j] = fp[list(fp.keys())[0]]['GWSources'][src_temp][j].value
        return data

if __name__ == '__main__':
    dir_path = os.path.dirname(os.path.realpath(__file__))
    print(dir_path)

    data_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                             'test','Param_MBHB.hdf5')

    data = read_HDF5_func(data_filepath)
    print(data)
