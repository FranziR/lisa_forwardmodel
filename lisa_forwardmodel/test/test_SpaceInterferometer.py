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

from lisa_forwardmodel.objects.LISA import EccentricLISA, CircularLISA
from lisa_forwardmodel.objects.Simulation import Simulation
import numpy as np

from lisa_forwardmodel.utils.readParameters import read_parameter_fun
from lisa_forwardmodel.utils.test_utils import MAPE


def test_orbit():
    """
    Checks the accuracy of the spacecraft orbits by comparing the lisa_forwardmodel simulations to the sytheticlisa
    results. MAPE (mean absolute percentage error) is chosen as a error measure.
    --> verifies parts of the SpaceInterferometer class implementation

    :return:
    """
    print('\nSTART LISA ORBIT TEST')

    # Loading reference data
    cur_path = os.path.dirname(os.path.abspath(__file__))
    t_ref = np.loadtxt(os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Circular/pCIR_1yr_time.out'))
    t_ref_ecc = np.loadtxt(os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Eccentric/pEcc_1yr_time.out'))

    p_ref = dict()
    p_ref['cir'] = dict()
    p_ref['cir']['1'] = np.loadtxt(os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Circular/pCIR1_1yr.out'))
    p_ref['cir']['2'] = np.loadtxt(os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Circular/pCIR2_1yr.out'))
    p_ref['cir']['3'] = np.loadtxt(os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Circular/pCIR3_1yr.out'))

    p_ref['ecc'] = dict()
    p_ref['ecc']['1'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Eccentric/pEcc1_1yr.out'))
    p_ref['ecc']['2'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Eccentric/pEcc2_1yr.out'))
    p_ref['ecc']['3'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/Orbit/Eccentric/pEcc3_1yr.out'))

    # Creating data
    sim_par = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                              'resources', 'Simulation_parameters.txt'))
    const = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                            'resources', 'Constants.txt'))
    # Simulation object
    sim_cir = Simulation(simulation_parameters=sim_par,
                         constant=const,
                         t_res=150,
                         t_max=31557585,
                         t=t_ref)
    sim_ecc = Simulation(simulation_parameters=sim_par,
                         constant=const,
                         t_res=150,
                         t_max=31557585,
                         t=t_ref_ecc)

    # Detector object (circular and eccentric LISA)
    LISA_file_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                  'resources', 'LISA_parameters.txt')
    parameters = read_parameter_fun(LISA_file_path)
    CirLISA = CircularLISA(parameters=parameters, sim=sim_cir)
    EccLISA = EccentricLISA(parameters=parameters, sim=sim_ecc)

    # Computing the orbits
    p = dict()
    p['cir'] = dict()
    p['cir'] = CirLISA.get_p()

    p['ecc'] = dict()
    p['ecc'] = EccLISA.get_p()

    # Asserting the accuracy
    dim = ['x', 'y', 'z']
    for i in list(p.keys()):
        for j in list(p[i].keys()):
            for k in range(0, 3):
                deviation = MAPE(p_ref[i][j][k], p[i][j][k])
                assert deviation < 1e-5, \
                    "Too strong deviation (%e) in LISA orbit: SC %s, LISA %s dimension %s" % (deviation, j, i, dim[k])
                print("MAPE p %s %s %s: %e" % (i, j, dim[k], deviation))

    print('LISA ORBIT TEST SUCCESS')
    return 0


def test_light_propagation():
    """
    Checks the accuracy of the light propagation direction and time of the laser beam between two spacecrafts,
    by comparing the lisa_forwardmodel simulations to the sytheticlisa results.
    MAPE (mean absolute percentage error) is chosen as a error measure.
    --> verifies parts of the SpaceInterferometer class implementation

    :return:
    """
    print('\nSTART LISA LIGHT PROPAGATION TEST')

    # Loading reference data
    cur_path = os.path.dirname(os.path.abspath(__file__))
    t_ref = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/nCIR_1yr_time.out'))
    t_ref_ecc = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/nEcc_1yr_time.out'))

    n_ref = dict()
    n_ref['cir'] = dict()
    n_ref['ecc'] = dict()

    L_ref = dict()
    L_ref['cir'] = dict()
    L_ref['ecc'] = dict()

    n_ref['cir']['1'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/nCIR1_1yr.out'))
    n_ref['cir']['2'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/nCIR2_1yr.out'))
    n_ref['cir']['3'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/nCIR3_1yr.out'))
    n_ref['cir']['-1'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/nCIRrev1_1yr.out'))
    n_ref['cir']['-2'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/nCIRrev2_1yr.out'))
    n_ref['cir']['-3'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/nCIRrev3_1yr.out'))

    L_ref['cir']['1'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/LCIR1_1yr.npy'))
    L_ref['cir']['2'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/LCIR2_1yr.npy'))
    L_ref['cir']['3'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/LCIR3_1yr.npy'))
    L_ref['cir']['-1'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/LCIRrev1_1yr.npy'))
    L_ref['cir']['-2'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/LCIRrev2_1yr.npy'))
    L_ref['cir']['-3'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Circular/LCIRrev3_1yr.npy'))

    n_ref['ecc']['1'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/nEcc1_1yr.out'))
    n_ref['ecc']['2'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/nEcc2_1yr.out'))
    n_ref['ecc']['3'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/nEcc3_1yr.out'))
    n_ref['ecc']['-1'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/nEccrev1_1yr.out'))
    n_ref['ecc']['-2'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/nEccrev2_1yr.out'))
    n_ref['ecc']['-3'] = np.loadtxt(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/nEccrev3_1yr.out'))

    L_ref['ecc']['1'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/LECC1_1yr.npy'))
    L_ref['ecc']['2'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/LECC2_1yr.npy'))
    L_ref['ecc']['3'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/LECC3_1yr.npy'))
    L_ref['ecc']['-1'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/LECCrev1_1yr.npy'))
    L_ref['ecc']['-2'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/LECCrev2_1yr.npy'))
    L_ref['ecc']['-3'] = np.load(
        os.path.join(cur_path, 'resources/SpaceInterferometer/LightPropagation/Eccentric/LECCrev3_1yr.npy'))

    # Simulation objects
    sim_par = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                              'resources', 'Simulation_parameters.txt'))
    const = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                            'resources', 'Constants.txt'))
    sim_cir = Simulation(simulation_parameters=sim_par,
                         constant=const,
                         t_res=150,
                         t_max=31557585,
                         t=t_ref)
    sim_ecc = Simulation(simulation_parameters=sim_par,
                         constant=const,
                         t_res=150,
                         t_max=31557585,
                         t=t_ref_ecc)

    # Detector object (circular and eccentric)
    LISA_file_path = os.path.join(os.path.join(os.path.dirname(os.path.realpath(__file__))),
                                  'resources', 'LISA_parameters.txt')
    parameters = read_parameter_fun(LISA_file_path)
    CirLISA = CircularLISA(parameters=parameters, sim=sim_cir)
    EccLISA = EccentricLISA(parameters=parameters, sim=sim_ecc)

    # Computing ligzt propagation characteristics
    n = dict()
    n['cir'] = dict()
    n['cir'] = CirLISA.get_n()

    n['ecc'] = dict()
    n['ecc'] = EccLISA.get_n()

    L = dict()
    L['cir'] = dict()
    L['cir'] = CirLISA.L_light()

    L['ecc'] = dict()
    L['ecc'] = EccLISA.L_light()

    # Asserting accuracy (light propagation direction)
    dim = ['x', 'y', 'z']
    for i in list(n.keys()):
        for j in list(n[i].keys()):
            for k in range(0, 3):
                deviation = MAPE(n_ref[i][j][k], n[i][j][k])
                assert deviation < 1e-2, \
                    "Too strong deviation (%e) in LISA laser propagation direction" \
                    ": SC %s, LISA %s dimension %s" % (deviation, j, i, dim[k])
                print("MAPE n %s %s %s: %e" % (i, j, dim[k], deviation))

    # Asserting accuracy (light propagation duration)
    for i in list(L.keys()):
        for j in list(L[i].keys()):
            deviation = MAPE(L_ref[i][j], L[i][j])
            assert deviation < 1e-5, \
                "Too strong deviation (%e) in LISA laser propagation time" \
                ": SC %s, LISA %s" % (deviation, j, i)
            print("MAPE L %s %s: %e" % (i, j, deviation))

    print('LISA LIGHT PROPAGATION TEST SUCCESS')
    return 0


if __name__ == "__main__":
    test_orbit()
    test_light_propagation()
