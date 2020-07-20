'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''

import os
import sys

myPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, myPath + '/../')
import numpy as np

from lisa_forwardmodel.objects.Simulation import Simulation
from lisa_forwardmodel.objects.Waveforms import WaveformSimpleBinaries
from lisa_forwardmodel.utils.readParameters import read_parameter_fun
from lisa_forwardmodel.utils.test_utils import MAPE


def test_GalBin():
    """
    Checks accuracy of the polarization components of gravitational waves, emitted by Galactic Binaries.
    For this, it compares the mean of the relative deviation of the lisa_forwardmodel simulation with the
    sytheticlisa results.
    --> verifies the implementation of the WaveformSimpleBinaries class implementation

    :return:
    """
    print('START GalBin TEST')

    # Loading reference data
    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test', 'resources',
                            'Waveform')
    hp_ref = np.load(os.path.join(ref_path, 'GalBin_hp.npy'))
    hc_ref = np.load(os.path.join(ref_path, 'GalBin_hc.npy'))
    h_ref = np.load(os.path.join(ref_path, 'GalBin_h.npy'))
    t_ref = np.load(os.path.join(ref_path, 'GalBin_t.npy'))

    # Simulation object
    sim = Simulation(t=t_ref)

    # Wave object (galactic binary)
    GalBin_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                   'GalBin_parameters.txt')
    GalBin_parameters = read_parameter_fun(GalBin_filepath)
    GalBin = WaveformSimpleBinaries(parameters=GalBin_parameters, sim=sim)

    # Computing plus and cross polarizations of gravitational wave
    hphc_GalBin = GalBin.h_plus_cross()
    h_GalBin = GalBin.h()
    h = np.zeros((h_GalBin.shape[-1], h_GalBin.shape[1], h_GalBin.shape[0]))

    # Asserting accuracy
    deviation_p = MAPE(hp_ref, hphc_GalBin['h_plus'])
    deviation_c = MAPE(hc_ref, hphc_GalBin['h_cross'])

    assert deviation_p < 1e-3, "Too strong deviation in hp component of Galactic Binary."
    print("MAPE h_p: %e" % deviation_p)
    assert deviation_c < 1e-3, "Too strong deviation in hc component of Galactic Binary."
    print("MAPE h_c: %e" % deviation_c)
    for i in range(0, 3):
        for j in range(0, 3):
            deviation_h = MAPE(h_ref[:, i, j], h_GalBin[i, j, :])
            assert deviation_h < 1e-3, "Too strong deviation in h(%d,%d) component of Galactic Binary." % (i, j)
            print("MAPE h(%d,%d): %e" % (i, j, deviation_h))
    print('GalBin TEST SUCCESS')
    return 0


if __name__ == '__main__':
    test_GalBin()
