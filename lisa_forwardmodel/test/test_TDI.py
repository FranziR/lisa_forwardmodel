import os, sys
myPath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, myPath + '/../')
import numpy as np
# import matplotlib.pyplot as plt

import lisa_forwardmodel.objects.Waveforms as WF
import lisa_forwardmodel.objects.LISA as DET
from lisa_forwardmodel.objects.Simulation import Simulation
from lisa_forwardmodel.objects.TDI import InterSCDopplerObservable, TDI
from lisa_forwardmodel.utils.readParameters import read_parameter_fun
from lisa_forwardmodel.utils.test_utils import MAPE


def test_InterSCObs_y():
    print('START INTERSPACECRAFT OBS. TEST')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test',
                            'resources', 'InterSCDopplerObservables')
    y_ref = dict()
    y_ref['ecc'] = dict()
    y_ref['ecc']['123'] = np.load(os.path.join(ref_path, 'Eccentric','LISAEcc_GalBin_y123.npy'))
    y_ref['ecc']['231'] = np.load(os.path.join(ref_path, 'Eccentric','LISAEcc_GalBin_y231.npy'))
    y_ref['ecc']['312'] = np.load(os.path.join(ref_path, 'Eccentric','LISAEcc_GalBin_y312.npy'))
    y_ref['ecc']['3-21'] = np.load(os.path.join(ref_path, 'Eccentric','LISAEcc_GalBin_y3neg21.npy'))
    y_ref['ecc']['2-13'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y2neg13.npy'))
    y_ref['ecc']['1-32'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y1neg32.npy'))

    y_ref['cir'] = dict()
    y_ref['cir']['123'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y123.npy'))
    y_ref['cir']['231'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y231.npy'))
    y_ref['cir']['312'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y312.npy'))
    y_ref['cir']['3-21'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y3neg21.npy'))
    y_ref['cir']['2-13'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y2neg13.npy'))
    y_ref['cir']['1-32'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y1neg32.npy'))
    t_ref = np.load(os.path.join(ref_path, 'Circular','LISACir_GalBin_time.npy'))
    t_ref_ecc = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_time.npy'))

    sim_par = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                              'resources', 'Simulation_parameters.txt'))
    const = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                            'resources', 'Constants.txt'))
    sim = Simulation(simulation_parameters=sim_par,
                     constant=const,
                     t_res=150,
                     t_max=31557585,
                     t=t_ref)

    sim_ecc = Simulation(simulation_parameters=sim_par,
                         constant=const,
                         t_res=150,
                         t_max=31557585,
                         t=t_ref_ecc)

    LISA_file_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                  'resources', 'LISA_parameters.txt')
    EccLISA = DET.EccentricLISA(filepath=LISA_file_path, sim=sim_ecc)
    CirLISA = DET.CircularLISA(filepath=LISA_file_path, sim=sim)

    GalBin_parameters = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                                        'resources', 'GalBin_parameters.txt'))
    GalBin = WF.WaveformSimpleBinaries(parameters=GalBin_parameters, sim=sim)

    CirDoppler = InterSCDopplerObservable(detector=CirLISA, wave=GalBin)
    EccDoppler = InterSCDopplerObservable(detector=EccLISA, wave=GalBin)

    y = dict()
    y['cir'] = dict()
    y['cir']['123'] = CirDoppler.y(s=1, r=3)
    y['cir']['231'] = CirDoppler.y(s=2, r=1)
    y['cir']['3-21'] = CirDoppler.y(s=3, r=1)
    y['cir']['312'] = CirDoppler.y(s=3, r=2)
    y['cir']['2-13'] = CirDoppler.y(s=2, r=3)
    y['cir']['1-32'] = CirDoppler.y(s=1, r=2)

    y['ecc'] = dict()
    y['ecc']['123'] = EccDoppler.y(s=1, r=3)
    y['ecc']['231'] = EccDoppler.y(s=2, r=1)
    y['ecc']['3-21'] = EccDoppler.y(s=3, r=1)
    y['ecc']['312'] = EccDoppler.y(s=3, r=2)
    y['ecc']['2-13'] = EccDoppler.y(s=2, r=3)
    y['ecc']['1-32'] = EccDoppler.y(s=1, r=2)

    for i in list(y.keys()):
        for j in list(y[i].keys()):
            deviation = MAPE(y_ref[i][j], y[i][j])
            assert deviation < 1e-4, \
                "Too strong deviation in interspacecraft Doppler Observables %s, %s" %(j,i)
            print("MAPE y %s %s: %e" %(i, j, deviation))

    plt.show()
    print('INTERSPACECRAFT OBS. TEST SUCCESS')
    return 0

def test_InterSCObs_retard():
    print('START RETARDATION TEST')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test', 'resources', 'TDI')

    y_ref = dict()
    y_ref['cir'] = dict()
    y_ref['cir']['y31'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y31.npy'))
    y_ref['cir']['y13_neg2'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y13_neg2.npy'))
    y_ref['cir']['y21_2neg2'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y21_2neg2.npy'))
    y_ref['cir']['y12_32neg2'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y12_32neg2.npy'))
    y_ref['cir']['y21_neg332neg2'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y21_neg332neg2.npy'))
    y_ref['cir']['y31_neg33neg332neg2'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y31_neg33neg332neg2.npy'))
    y_ref['cir']['y13_neg2neg33neg332neg2'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y13_neg2neg33neg332neg2.npy'))
    y_ref['cir']['y21'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y21.npy'))
    y_ref['cir']['y12_3'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y12_3.npy'))
    y_ref['cir']['y31_neg33'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y31_neg33.npy'))
    y_ref['cir']['y13_neg2neg33'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y13_neg2neg33.npy'))
    y_ref['cir']['y31_2neg2neg33'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_y31_2neg2neg33.npy'))
    y_ref['cir']['y13_neg22neg2neg33'] = np.load(os.path.join(ref_path, 'Circular',
                                                              'LISACir_GalBin_y13_neg22neg2neg33.npy'))
    y_ref['cir']['y21_2neg22neg2neg33'] = np.load(os.path.join(ref_path, 'Circular',
                                                               'LISACir_GalBin_y21_2neg22neg2neg33.npy'))
    y_ref['cir']['y12_32neg22neg2neg33'] = np.load(os.path.join(ref_path, 'Circular',
                                                               'LISACir_GalBin_y12_32neg22neg2neg33.npy'))

    y_ref['ecc'] = dict()
    y_ref['ecc']['y31'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y31.npy'))
    y_ref['ecc']['y13_neg2'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y13_neg2.npy'))
    y_ref['ecc']['y21_2neg2'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y21_2neg2.npy'))
    y_ref['ecc']['y12_32neg2'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y12_32neg2.npy'))
    y_ref['ecc']['y21_neg332neg2'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y21_neg332neg2.npy'))
    y_ref['ecc']['y12_32neg22neg2neg33'] = np.load(os.path.join(ref_path, 'Eccentric',
                                                                'LISAEcc_GalBin_y12_32neg22neg2neg33.npy'))
    y_ref['ecc']['y31_neg33neg332neg2'] = np.load(
        os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y31_neg33neg332neg2.npy'))
    y_ref['ecc']['y13_neg2neg33neg332neg2'] = np.load(
        os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y13_neg2neg33neg332neg2.npy'))
    y_ref['ecc']['y21'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y21.npy'))
    y_ref['ecc']['y12_3'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y12_3.npy'))
    y_ref['ecc']['y31_neg33'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y31_neg33.npy'))
    y_ref['ecc']['y13_neg2neg33'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y13_neg2neg33.npy'))
    y_ref['ecc']['y31_2neg2neg33'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_y31_2neg2neg33.npy'))
    y_ref['ecc']['y13_neg22neg2neg33'] = np.load(os.path.join(ref_path, 'Eccentric',
                                                              'LISAEcc_GalBin_y13_neg22neg2neg33.npy'))
    y_ref['ecc']['y21_2neg22neg2neg33'] = np.load(os.path.join(ref_path, 'Eccentric',
                                                               'LISAEcc_GalBin_y21_2neg22neg2neg33.npy'))

    t_ref = np.load(os.path.join(ref_path, 'Eccentric','LISAEcc_GalBin_time.npy'))

    sim_par = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                              'resources', 'Simulation_parameters.txt'))
    const = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                            'resources', 'Constants.txt'))
    sim = Simulation(simulation_parameters=sim_par,
                     constant=const,
                     t_res=150,
                     t_max=31557585,
                     t=t_ref)

    sim = Simulation(t = t_ref)

    LISA_file_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                  'resources', 'LISA_parameters.txt')
    EccLISA = DET.EccentricLISA(filepath=LISA_file_path, sim=sim)
    CirLISA = DET.CircularLISA(filepath=LISA_file_path, sim=sim)

    GalBin_parameters = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                                        'resources', 'GalBin_parameters.txt'))
    GalBin = WF.WaveformSimpleBinaries(parameters=GalBin_parameters, sim=sim)

    CirDoppler = InterSCDopplerObservable(detector=CirLISA, wave=GalBin)
    EccDoppler = InterSCDopplerObservable(detector=EccLISA, wave=GalBin)

    y = dict()
    y['cir'] = dict()
    y['cir']['y31'] = CirDoppler.y(t=t_ref, s=3, r=1)
    y['cir']['y13_neg2'] = CirDoppler.y(t=t_ref, s=1, r=3, retard=-2)
    y['cir']['y21_2neg2'] = CirDoppler.y(t=t_ref, s=2, r=1, retard=[2, -2])
    y['cir']['y12_32neg2'] = CirDoppler.y(t=t_ref, s=1, r=2, retard=[3, 2, -2])
    y['cir']['y21_neg332neg2'] = CirDoppler.y(t=t_ref, s=2, r=1, retard=[-3, 3, 2, -2])
    y['cir']['y31_neg33neg332neg2'] = CirDoppler.y(t=t_ref, s=3, r=1, retard=[-3, 3, -3, 3, 2, -2])
    y['cir']['y13_neg2neg33neg332neg2'] = CirDoppler.y(t=t_ref, s=1, r=3, retard=[-2, -3, 3, -3, 3, 2, -2])
    y['cir']['y21'] = CirDoppler.y(t=t_ref, s=2, r=1)
    y['cir']['y12_3'] = CirDoppler.y(t=t_ref, s=1, r=2, retard=3)
    y['cir']['y12_32neg22neg2neg33'] = CirDoppler.y(t=t_ref, s=1, r=2, retard=[3, 2, -2, 2, -2, -3, 3])
    y['cir']['y31_neg33'] = CirDoppler.y(t=t_ref, s=3, r=1, retard=[-3, 3])
    y['cir']['y13_neg2neg33'] = CirDoppler.y(t=t_ref, s=1,  r=3, retard=[-2, -3, 3])
    y['cir']['y31_2neg2neg33'] = CirDoppler.y(t=t_ref, s=3, r=1, retard=[2, -2, -3, 3])
    y['cir']['y13_neg22neg2neg33'] = CirDoppler.y(t=t_ref, s=1, r=3, retard=[-2, 2, -2, -3, 3])
    y['cir']['y21_2neg22neg2neg33'] = CirDoppler.y(t=t_ref, s=2, r=1, retard=[2, -2, 2, -2, -3, 3])

    y['ecc'] = dict()
    y['ecc']['y31'] = EccDoppler.y(t=t_ref, s=3, r=1)
    y['ecc']['y13_neg2'] = EccDoppler.y(t=t_ref, s=1, r=3, retard=-2)
    y['ecc']['y21_2neg2'] = EccDoppler.y(t=t_ref, s=2, r=1, retard=[2, -2])
    y['ecc']['y12_32neg2'] = EccDoppler.y(t=t_ref, s=1, r=2, retard=[3, 2, -2])
    y['ecc']['y21_neg332neg2'] = EccDoppler.y(t=t_ref, s=2, r=1, retard=[-3, 3, 2, -2])
    y['ecc']['y31_neg33neg332neg2'] = EccDoppler.y(t=t_ref, s=3, r=1, retard=[-3, 3, -3, 3, 2, -2])
    y['ecc']['y13_neg2neg33neg332neg2'] = EccDoppler.y(t=t_ref, s=1, r=3, retard=[-2, -3, 3, -3, 3, 2, -2])
    y['ecc']['y21'] = EccDoppler.y(t=t_ref, s=2, r=1)
    y['ecc']['y12_3'] = EccDoppler.y(t=t_ref, s=1, r=2, retard=3)
    y['ecc']['y12_32neg22neg2neg33'] = EccDoppler.y(t=t_ref, s=1, r=2, retard=[3, 2, -2, 2, -2, -3, 3])
    y['ecc']['y31_neg33'] = EccDoppler.y(t=t_ref, s=3, r=1, retard=[-3, 3])
    y['ecc']['y13_neg2neg33'] = EccDoppler.y(t=t_ref, s=1, r=3, retard=[-2, -3, 3])
    y['ecc']['y31_2neg2neg33'] = EccDoppler.y(t=t_ref, s=3, r=1, retard=[2, -2, -3, 3])
    y['ecc']['y13_neg22neg2neg33'] = EccDoppler.y(t=t_ref, s=1, r=3, retard=[-2, 2, -2, -3, 3])
    y['ecc']['y21_2neg22neg2neg33'] = EccDoppler.y(t=t_ref, s=2, r=1, retard=[2, -2, 2, -2, -3, 3])

    for i in list(y.keys()):
        for j in list(y[i].keys()):
            deviation = MAPE(y_ref[i][j], y[i][j])
            assert deviation < 1e-4, \
                "Too strong deviation in interspacecraft Doppler Observables %s, %s" % (j, i)
            print("MAPE y %s %s: %e" % (i, j, deviation))

    print('RETARDATION TEST SUCCESS')
    return 0

def test_TDI_X():
    print('START X TEST')
    ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test', 'resources', 'TDI')

    X_ref = dict()
    X_ref['cir'] = dict()
    X_ref['cir']['1'] = np.load(os.path.join(ref_path, 'Circular','LISACir_GalBin_X1.npy'))
    X_ref['cir']['2'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_X2.npy'))
    X_ref['cir']['3'] = np.load(os.path.join(ref_path, 'Circular', 'LISACir_GalBin_X3.npy'))

    X_ref['ecc'] = dict()
    X_ref['ecc']['1'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_X1.npy'))
    X_ref['ecc']['2'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_X2.npy'))
    X_ref['ecc']['3'] = np.load(os.path.join(ref_path, 'Eccentric', 'LISAEcc_GalBin_X3.npy'))

    t_ref = np.load(os.path.join(ref_path, 'Circular','LISACir_GalBin_time.npy'))

    sim_par = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                              'resources', 'Simulation_parameters.txt'))
    const = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                            'resources', 'Constants.txt'))
    sim = Simulation(simulation_parameters=sim_par,
                     constant=const,
                     t_res=150,
                     t_max=31557585,
                     t=t_ref)

    sim = Simulation(t = t_ref)

    LISA_file_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                  'resources', 'LISA_parameters.txt')
    EccLISA = DET.EccentricLISA(filepath=LISA_file_path, sim=sim)
    CirLISA = DET.CircularLISA(filepath=LISA_file_path, sim=sim)

    GalBin_parameters = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                                        'resources', 'GalBin_parameters.txt'))
    GalBin = WF.WaveformSimpleBinaries(parameters=GalBin_parameters, sim=sim)

    CirTDI = TDI(detector=CirLISA, wave=GalBin)
    EccTDI = TDI(detector=EccLISA, wave=GalBin)

    X = dict()
    X['cir'] = dict()
    X['cir']['1'] = CirTDI.X1()
    X['cir']['2'] = CirTDI.X2()
    X['cir']['3'] = CirTDI.X3()

    X['ecc'] = dict()
    X['ecc']['1'] = EccTDI.X1()
    X['ecc']['2'] = EccTDI.X2()
    X['ecc']['3'] = EccTDI.X3()

    for i in list(X.keys()):
        for j in list(X[i].keys()):
            deviation = MAPE(X_ref[i][j], X[i][j])
            assert deviation < 1e-4, \
                "Too strong deviation in interspacecraft Doppler Observables %s, %s" % (j, i)
            print("MAPE X %s %s: %e" % (i, j, deviation))

    print('RETARDATION TEST SUCCESS')
    return 0

if __name__ == '__main__':
    # test_InterSCObs_y()
    # test_InterSCObs_retard()
    test_TDI_X()