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
    print('START GalBin TEST')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test', 'resources',
                            'Waveform')
    hp_ref = np.load(os.path.join(ref_path, 'GalBin_hp.npy'))
    hc_ref = np.load(os.path.join(ref_path, 'GalBin_hc.npy'))
    h_ref = np.load(os.path.join(ref_path, 'GalBin_h.npy'))
    t_ref = np.load(os.path.join(ref_path, 'GalBin_t.npy'))
    sim = Simulation(t=t_ref)

    GalBin_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                   'GalBin_parameters.txt')
    GalBin_parameters = read_parameter_fun(GalBin_filepath)
    GalBin = WaveformSimpleBinaries(parameters=GalBin_parameters, sim=sim)

    hphc_GalBin = GalBin.h_plus_cross()
    h_GalBin = GalBin.h()
    h = np.zeros((h_GalBin.shape[-1], h_GalBin.shape[1], h_GalBin.shape[0]))

    deviation_p = MAPE(hp_ref, hphc_GalBin['h_plus'])
    deviation_c = MAPE(hc_ref, hphc_GalBin['h_cross'])

    # import matplotlib.pyplot as plt
    # t_start = 260285
    # t_end = 260295
    # plt.close('all')
    # fig1 = plt.figure(1)
    # plt.subplot(2, 1, 1)
    # plt.plot(t_ref[t_start:t_end], hp_ref[t_start:t_end], 'red', Linewidth=1, label='hp_ref')
    # plt.plot(t_ref[t_start:t_end], hphc_GalBin['h_plus'][t_start:t_end], 'black', Linewidth=1, Linestyle='--', label='hp')
    # # plt.xlabel('time in sec')
    # plt.ylabel('hp')
    # plt.grid()
    #
    # plt.subplot(2, 1, 2)
    # plt.plot(t_ref[t_start:t_end], hc_ref[t_start:t_end], 'red', Linewidth=1, label='hc_ref')
    # plt.plot(t_ref[t_start:t_end], hphc_GalBin['h_cross'][t_start:t_end], 'black', Linewidth=1, Linestyle='--', label='hc')
    # plt.xlabel('time in sec')
    # plt.ylabel('hc')
    # plt.grid()

    # plt.show()

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
