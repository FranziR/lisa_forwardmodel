'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''

import inspect
import os

import matplotlib.pyplot as plt
import numpy as np

import lisa_forwardmodel.objects.Waveforms as WF
from lisa_forwardmodel.objects.Simulation import Simulation
from lisa_forwardmodel.utils.LISAutils import checkLink, checkCraft, get_link
from lisa_forwardmodel.utils.checkInput import checkInput
from lisa_forwardmodel.utils.readParameters import read_parameter_fun

WFobjects = tuple(
    [eval('WF.' + m[0]) for m in inspect.getmembers(WF, inspect.isclass) if m[1].__module__ == WF.__name__])

import lisa_forwardmodel.objects.LISA as DET

DETobjects = tuple(
    [eval('DET.' + m[0]) for m in inspect.getmembers(DET, inspect.isclass) if m[1].__module__ == DET.__name__])


# NOIobjects = tuple(
#     [eval('NOI.' + m[0]) for m in inspect.getmembers(NOI, inspect.isclass) if m[1].__module__ == NOI.__name__])


# TODO: waveform class is responsible for computing the antenna beam patterns, LISA response and TDI

class InterSCDopplerObservable():
    """
    Class that computes the interspacecraft Doppler observables between two spacecrafts for a passing gravitational wave.
    No noise is considered

    ATTRIBUTES:
        wave: (WaveformClass) wave object
        detector: (SpaceInterferometer) detector object

    METHODS:
        retard:     retards the basic Doppler measurements along a given link
        y:          computes the interspacecraft Doppler measurements for a passing gravitational wave
    """

    def __init__(self,
                 detector=None,
                 wave=None):

        assert isinstance(detector, DETobjects), \
            "Detector object in class Doppler must be one of the classes in src.objects.LISA."

        assert isinstance(wave, WFobjects), \
            "Wave object in class Doppler must be one of the waveform classes in src.objects.Waveforms."

        assert (detector.sim is not None and wave.sim is not None), \
            "Wave and detector have to be defined for same time interval."

        assert (detector.sim.__eq__(wave.sim)), \
            "Detector and Wave object should have same simulation parameters."

        self.detector = detector
        self.wave = wave
        self.sim = wave.sim

    @property
    def wave(self):
        return self.__wave

    @wave.setter
    def wave(self, obj):
        if obj is not None:
            assert isinstance(obj, WFobjects), \
                "Wave object in class Doppler must be one of the waveform classes in src.objects.Waveforms."
            self.__wave = obj

    @property
    def detector(self):
        return self.__detector

    @detector.setter
    def detector(self, obj):
        if obj is not None:
            assert isinstance(obj, DETobjects), \
                "Detector object in class Doppler must be one of the classes in src.objects.LISA."
            self.__detector = obj

    def retard(self, i=None, t=None):
        """
        Retards the a given time array by the light propagation duration along a certain arm.

        :param i: (int) link along which to retard
        :param t: (int/float/list/array)   time array to be retarded
        :return: retarded time array
        """
        ind_t, temp_t = checkInput(t, np.ndarray, (int, float, list))
        if ind_t:
            t_initial = temp_t
        else:
            t_initial = self.sim.t

        ind_i, temp_i = checkInput(i, list, (int, float, np.ndarray))
        if ind_i:
            l = temp_i.copy()
        else:
            raise TypeError

        t_ret = t_initial
        while len(l) != 0:
            l_ret = l.pop(-1)
            assert checkLink(l_ret)
            t_ret = t_ret - self.detector.L_light(t=t_ret, link=l_ret)[str(l_ret)]
        return t_ret

    def y(self, t=None, s=None, r=None, retard=None):
        """
        Computes the interspacecraft Doppler measurements between two crafts.

        :param t: (int/float/list/array)   time during which Doppler observables are to be simulated
        :param s: (int) sending spacecraft
        :param r: (int) receiving spacecraft
        :param retard: (int/list) list of indices or index along which the time array has to be retarded
        :return: interspacecraft Doppler observable y
        """
        assert checkCraft(s), "Wrong sending spacecraft."
        assert checkCraft(r), "Wrong receiving spacecraft."

        if t is not None:
            if isinstance(t, np.ndarray):
                t_temp = t
            else:
                t_temp = np.array(t)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        if retard is not None:
            t_rec = self.retard(i=retard, t=t_temp)
        else:
            t_rec = t_temp

        try:
            dim_t = len(t_temp)
        except TypeError:
            dim_t = 1

        k = self.wave.get_k()
        l = get_link(recv=r, send=s)

        # 1. reception time: set t_r = t_temp
        # 2. emission time: retard reception time by armlength
        travel_times = self.detector.L_light(t=t_rec, link=l)
        t_send = t_rec - travel_times[str(l)]

        # 3. compute photon propagation direction n_l(t_r) = (p_r(t_r) - p_s(t_s))/|p_r(t_r) - p_s(t_s)|
        p_r = self.detector.get_p(t=t_rec)[str(r)]
        p_s = self.detector.get_p(t=t_send)[str(s)]
        n_sr = (p_r - p_s) / np.sqrt(np.sum((p_r - p_s) ** 2, axis=0))

        # 4. retardation along k*p_s(t_s) and k*p_r(t_r)
        retard_r = t_rec - np.dot(k, p_r)
        retard_s = t_send - np.dot(k, p_s)

        # 5. gravitational wave tensors at retarded times
        h_r = self.wave.h(t=retard_r)
        h_s = self.wave.h(t=retard_s)

        # 6. computing psi values (projected GW responses)
        psi_r = np.zeros((dim_t,))
        psi_s = np.zeros((dim_t,))
        for j in range(0, dim_t):
            psi_r[j] = 0.5 * np.dot(np.dot(n_sr[:, j].transpose(), h_r[:, :, j]), n_sr[:, j])
            psi_s[j] = 0.5 * np.dot(np.dot(n_sr[:, j].transpose(), h_s[:, :, j]), n_sr[:, j])

        norm = 1 - np.dot(k, n_sr)
        return (psi_s - psi_r) / norm


class TDI(InterSCDopplerObservable):
    """
    Class TDI simulates a set of different TDI observables. Again, no noise is taken into account.

    ATTRIBUTES:
        wave: (WaveformClass) wave object
        detector: (SpaceInterferometer) detector object

    METHODS:
        X1, X2, X3: (array) each of these compute on of the equally named unequal arm Michelson TDI observable
        AET: (dict)         one function to compute all of the three optimal TDI observables
    """

    def __init__(self,
                 detector=None,
                 wave=None):

        super().__init__(detector,
                         wave)

        # self.noise = noise

    # @property
    # def noise(self):
    #     return self.__noise
    #
    # @noise.setter
    # def noise(self, obj):
    #     if obj is not None:
    #         assert isinstance(obj, NOIobjects), \
    #             "Noise object in class TDI must be one of the classes in src.objects.Noise."
    #         self.__noise = obj

    def X1(self, t=None):
        """
        Computes unequal arm Michelson TDI observable X1.

        :param t: (int/float/list/array)   time at which X1 is to be computed
        :return: X1 during time t
        """
        ind_t, temp_t = checkInput(t, np.ndarray, (int, float, list))
        if ind_t:
            t_initial = temp_t
        else:
            t_initial = self.sim.t

        y31 = self.y(t=t_initial, s=3, r=1)
        y13_neg2 = self.y(t=t_initial, s=1, r=3, retard=-2)
        y21_2neg2 = self.y(t=t_initial, s=2, r=1, retard=[2, -2])
        y12_32neg2 = self.y(t=t_initial, s=1, r=2, retard=[3, 2, -2])
        y21_neg332neg2 = self.y(t=t_initial, s=2, r=1, retard=[-3, 3, 2, -2])
        y12_3neg332neg2 = self.y(t=t_initial, s=1, r=2, retard=[3, -3, 3, 2, -2])
        y31_neg33neg332neg2 = self.y(t=t_initial, s=3, r=1, retard=[-3, 3, -3, 3, 2, -2])
        y13_neg2neg33neg332neg2 = self.y(t=t_initial, s=1, r=3, retard=[-2, -3, 3, -3, 3, 2, -2])
        y21 = self.y(t=t_initial, s=2, r=1)
        y12_3 = self.y(t=t_initial, s=1, r=2, retard=3)
        y31_neg33 = self.y(t=t_initial, s=3, r=1, retard=[-3, 3])
        y13_neg2neg33 = self.y(t=t_initial, s=1, r=3, retard=[-2, -3, 3])
        y31_2neg2neg33 = self.y(t=t_initial, s=3, r=1, retard=[2, -2, -3, 3])
        y13_neg22neg2neg33 = self.y(t=t_initial, s=1, r=3, retard=[-2, 2, -2, -3, 3])
        y21_2neg22neg2neg33 = self.y(t=t_initial, s=2, r=1, retard=[2, -2, 2, -2, -3, 3])
        y12_32neg22neg2neg33 = self.y(t=t_initial, s=1, r=2, retard=[3, 2, -2, 2, -2, -3, 3])

        return (y12_32neg22neg2neg33 - y13_neg2neg33neg332neg2 + y21_2neg22neg2neg33 - y31_neg33neg332neg2
                + y13_neg22neg2neg33 - y12_3neg332neg2 + y31_2neg2neg33 - y21_neg332neg2 + y13_neg2neg33
                - y12_32neg2 + y31_neg33 - y21_2neg2 + y12_3 - y13_neg2 + y21 - y31)

    def X2(self, t=None):
        """
        Computes unequal arm Michelson TDI observable X2.

        :param t: (int/float/list/array)   time at which X2 is to be computed
        :return: X2 during time t
        """
        ind_t, temp_t = checkInput(t, np.ndarray, (int, float, list))
        if ind_t:
            t_initial = temp_t
        else:
            t_initial = self.sim.t

        y23_13neg33neg3neg11 = self.y(t=t_initial, s=2, r=3, retard=[1, 3, -3, 3, -3, -1, 1])
        y21_neg3neg11neg113neg3 = self.y(t=t_initial, s=2, r=1, retard=[-3, -1, 1, -1, 1, 3, -3])
        y32_3neg33neg3neg11 = self.y(t=t_initial, s=3, r=2, retard=[3, -3, 3, -3, -1, 1])
        y12_neg11neg113neg3 = self.y(t=t_initial, s=1, r=2, retard=[-1, 1, -1, 1, 3, -3])
        y21_neg33neg3neg11 = self.y(t=t_initial, s=2, r=1, retard=[-3, 3, -3, -1, 1])
        y23_1neg113neg3 = self.y(t=t_initial, s=2, r=3, retard=[1, -1, 1, 3, -3])
        y12_3neg3neg11 = self.y(t=t_initial, s=1, r=2, retard=[3, -3, -1, 1])
        y32_neg113neg3 = self.y(t=t_initial, s=3, r=2, retard=[-1, 1, 3, -3])
        y21_neg3neg11 = self.y(t=t_initial, s=2, r=1, retard=[-3, -1, 1])
        y23_13neg3 = self.y(t=t_initial, s=2, r=3, retard=[1, 3, -3])
        y12_neg11 = self.y(t=t_initial, s=1, r=2, retard=[-1, 1])
        y32_3neg3 = self.y(t=t_initial, s=3, r=2, retard=[3, -3])
        y23_1 = self.y(t=t_initial, s=2, r=3, retard=1)
        y21_neg3 = self.y(t=t_initial, s=2, r=1, retard=-3)
        y32 = self.y(t=t_initial, s=3, r=2)
        y12 = self.y(t=t_initial, s=1, r=2)

        return (y23_13neg33neg3neg11 - y21_neg3neg11neg113neg3 \
                + y32_3neg33neg3neg11 - y12_neg11neg113neg3 \
                + y21_neg33neg3neg11 - y23_1neg113neg3 \
                + y12_3neg3neg11 - y32_neg113neg3 \
                + y21_neg3neg11 - y23_13neg3 \
                + y12_neg11 - y32_3neg3 \
                + y23_1 - y21_neg3 \
                + y32 - y12)

    def X3(self, t=None):
        """
        Computes unequal arm Michelson TDI observable X3.

        :param t: (int/float/list/array)   time at which X3 is to be computed
        :return: X3 during time t
        """
        ind_t, temp_t = checkInput(t, np.ndarray, (int, float, list))
        if ind_t:
            t_initial = temp_t
        else:
            t_initial = self.sim.t

        y31_21neg11neg1neg22 = self.y(t=t_initial, s=3, r=1, retard=[2, 1, -1, 1, -1, -2, 2])
        y32_neg1neg22neg221neg1 = self.y(t=t_initial, s=3, r=2, retard=[-1, -2, 2, -2, 2, 1, -1])
        y13_1neg11neg1neg22 = self.y(t=t_initial, s=1, r=3, retard=[1, -1, 1, -1, -2, 2])
        y23_neg22neg221neg1 = self.y(t=t_initial, s=2, r=3, retard=[-2, 2, -2, 2, 1, -1])
        y32_neg11neg1neg22 = self.y(t=t_initial, s=3, r=2, retard=[-1, 1, -1, -2, 2])
        y31_2neg221neg1 = self.y(t=t_initial, s=3, r=1, retard=[2, -2, 2, 1, -1])
        y23_1neg1neg22 = self.y(t=t_initial, s=2, r=3, retard=[1, -1, -2, 2])
        y13_neg221neg1 = self.y(t=t_initial, s=1, r=3, retard=[-2, 2, 1, -1])
        y32_neg1neg22 = self.y(t=t_initial, s=3, r=2, retard=[-1, -2, 2])
        y31_21neg1 = self.y(t=t_initial, s=3, r=1, retard=[2, 1, -1])
        y23_neg22 = self.y(t=t_initial, s=2, r=3, retard=[-2, 2])
        y13_1neg1 = self.y(t=t_initial, s=1, r=3, retard=[1, -1])
        y31_2 = self.y(t=t_initial, s=3, r=1, retard=2)
        y32_neg1 = self.y(t=t_initial, s=3, r=2, retard=-1)
        y13 = self.y(t=t_initial, s=1, r=3)
        y23 = self.y(t=t_initial, s=2, r=3)

        return (y31_21neg11neg1neg22 - y32_neg1neg22neg221neg1 \
                + y13_1neg11neg1neg22 - y23_neg22neg221neg1 \
                + y32_neg11neg1neg22 - y31_2neg221neg1 \
                + y23_1neg1neg22 - y13_neg221neg1 \
                + y32_neg1neg22 - y31_21neg1 \
                + y23_neg22 - y13_1neg1 \
                + y31_2 - y32_neg1 \
                + y13 - y23)

    def AET(self, t=None):
        """
        Computes optimal TDI observable A, E and T.

        :param t: (int/float/list/array)   time at which X3 is to be computed
        :return: A, E and T, stored in a dictionary
        """
        ind_t, temp_t = checkInput(t, np.ndarray, (int, float, list))
        if ind_t:
            t_initial = temp_t
        else:
            t_initial = self.sim.t

        X = self.X1(t=t_initial)
        Y = self.X2(t=t_initial)
        Z = self.X3(t=t_initial)

        AET = dict()
        AET['A'] = 1 / 3 * (2 * X - Y - Z)
        AET['E'] = 1 / np.sqrt(3) * (Z - Y)
        AET['T'] = 1 / 3 * (X + Y + Z)

        return AET


if __name__ == '__main__':
    sim_par = read_parameter_fun(
        os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                     'input_data', 'Simulation_parameters.txt'))

    const = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                                            'input_data', 'Constants.txt'))
    sim = Simulation()

    LISA_file_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))),
                                  'input_data', 'LISA_parameters.txt')
    EccLISA = DET.EccentricLISA(filepath=LISA_file_path, sim=sim)
    CirLISA = DET.CircularLISA(filepath=LISA_file_path, sim=sim)

    GalBin_parameters = read_parameter_fun(os.path.join(os.path.join(os.path.dirname(
        os.path.dirname(os.path.realpath(__file__)))), 'input_data', 'GalBin_parameters.txt'))
    GalBin = WF.WaveformSimpleBinaries(parameters=GalBin_parameters, sim=sim)

    CirTDI = TDI(detector=CirLISA, wave=GalBin)
    EccTDI = TDI(detector=EccLISA, wave=GalBin)
    CirAET = CirTDI.AET()
    EccAET = EccTDI.AET()

    TDIObs = dict()
    TDIObs['cir'] = dict()
    TDIObs['cir']['A'] = CirAET['A']
    TDIObs['cir']['E'] = CirAET['E']
    TDIObs['cir']['T'] = CirAET['T']

    TDIObs['ecc'] = dict()
    TDIObs['ecc']['A'] = EccAET['A']
    TDIObs['ecc']['E'] = EccAET['E']
    TDIObs['ecc']['T'] = EccAET['T']

    plt.close()
    count = 1
    for i in list(TDIObs.keys()):
        plt.figure(i)
        t_start = 0
        t_end = -1
        plt.subplot(3, 1, 1)
        plt.title('Optimal TDI Observables: LISA ' + i)
        plt.plot(sim.t[t_start:t_end], TDIObs[i]['A'], 'red', Linewidth=1, label='A')
        plt.grid()
        plt.ylabel('A')
        plt.subplot(3, 1, 2)
        plt.plot(sim.t[t_start:t_end], TDIObs[i]['E'], 'green', Linewidth=1, label='E')
        plt.grid()
        plt.ylabel('E')
        plt.subplot(3, 1, 2)
        plt.plot(sim.t[t_start:t_end], TDIObs[i]['T'], 'blue', Linewidth=1, label='T')
        plt.grid()
        plt.ylabel('T')
        plt.xlabel('time (sec)')
