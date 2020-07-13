import os
import sys
from math import pi as pi

import matplotlib.pyplot as plt
import numpy as np
from numpy import cos as cos
from numpy import sin as sin

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'utils'))
from src.utils.readParameters import read_parameter_fun
from src.objects.Simulation import Simulation


class WaveformClass:
    def __init__(self,
                 parameters=None,
                 filepath=None,
                 source=None,
                 sim=None,
                 lat=None,
                 long=None,
                 psi=None):
        assert not (parameters is None and filepath is None), \
            "Please give either a path to a file with GW source parameters (input filepath) or a read-to-use " \
            "parameter dictionary. "

        self.parameters = parameters
        self.filepath = filepath
        self.source = source
        self.sim = sim

        self.lat = lat
        self.long = long
        self.psi = psi

    @property
    def parameters(self):
        return self.__parameters

    @parameters.setter
    def parameters(self, value):
        if value is not None:
            self.__parameters = value
        elif self.filepath is not None:
            read_in_data = read_parameter_fun(self.filepath)
            assert isinstance(read_in_data, dict)
            self.__parameters = read_in_data
        else:
            self.__parameters = 0

    @property
    def filepath(self):
        return self.__filepath

    @filepath.setter
    def filepath(self, value):
        if value is not None:
            self.__filepath = value
        else:
            self.__filepath = 0

    @property
    def source(self):
        return self.__source

    @source.setter
    def source(self, value):
        if value is not None and isinstance(value, str):
            self.__source = value
        else:
            self.__source = self.parameters['source']

    @property
    def lat(self):
        return self.__lat

    @lat.setter
    def lat(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__lat = value
        else:
            self.__lat = self.parameters['lat']

    @property
    def long(self):
        return self.__long

    @long.setter
    def long(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__long = value
        else:
            self.__long = self.parameters['long']

    @property
    def psi(self):
        return self.__psi

    @psi.setter
    def psi(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__psi = value
        else:
            self.__psi = self.parameters['psi']

    def get_k(self):
        k_temp = np.zeros((3,))
        k_temp[0] = -cos(self.lat) * cos(self.long)
        k_temp[1] = -cos(self.lat) * sin(self.long)
        k_temp[2] = -sin(self.lat)
        return k_temp

    def get_E(self):
        # Euler rotation sequence, transformation from frame aligned with hp and hc into SSB
        E_temp = np.zeros((3, 3))
        l = self.long
        b = self.lat
        p = self.psi
        E_temp[0, 0] = sin(l) * cos(p) - cos(l) * sin(b) * sin(p)
        E_temp[0, 1] = -sin(l) * sin(p) - cos(l) * sin(b) * cos(p)
        E_temp[0, 2] = -cos(l) * cos(b)

        E_temp[1, 0] = -cos(l) * cos(p) - sin(l) * sin(b) * sin(p)
        E_temp[1, 1] = cos(l) * sin(p) - sin(l) * sin(b) * cos(p)
        E_temp[1, 2] = -sin(l) * cos(b)

        E_temp[2, 0] = cos(b) * sin(p)
        E_temp[2, 1] = cos(b) * cos(p)
        E_temp[2, 2] = -sin(b)
        return E_temp

    def get_ep(self):
        ep_temp = np.zeros((3, 3))
        ep_temp[0, 0] = 1
        ep_temp[1, 1] = -1

        E = self.get_E()
        Et = E.transpose()
        return np.matmul(np.matmul(E, ep_temp), Et)

    def get_ec(self):
        ec_temp = np.zeros((3, 3))
        ec_temp[0, 1] = 1
        ec_temp[1, 0] = 1

        E = self.get_E()
        Et = E.transpose()
        return np.matmul(np.matmul(E, ec_temp), Et)

    def get_h(self,
              h_plus=None,
              h_cross=None):
        assert len(h_cross) == len(h_plus), "Source specific inputs h_cross and h_plus do not have matching lengths."

        ec = self.get_ec()
        ep = self.get_ep()
        h = np.zeros((3, 3, len(h_plus)))

        for i in range(0, len(h_plus)):
            h[:, :, i] = h_plus[i] * ep + h_cross[i] * ec
        return h


class WaveformMBHB(WaveformClass):
    def __init__(self,
                 parameters=None,
                 filepath=None,
                 source=None,
                 sim=None,
                 PN=None,
                 long=None,
                 lat=None,
                 phil_l=None,
                 theta_l=None,
                 psi=None,
                 iota=None,
                 d_l=None,
                 z=None,
                 m1=None,
                 chi1=None,
                 theta_s1=None,
                 phi_s1=None,
                 m2=None,
                 chi2=None,
                 theta_s2=None,
                 phi_s2=None,
                 t_c=None,
                 phi_c=None,
                 spin=None):
        super().__init__(parameters,
                         filepath,
                         source,
                         sim)
        assert not (self.parameters is None and self.filepath is None), \
            "Please give either a path to a file with GW source parameters (input filepath) or a read-to-use " \
            "parameter dictionary. "

        self.spin = spin
        self.PN = PN

        self.long = long
        self.lat = lat
        self.phil_l = phil_l
        self.theta_l = theta_l
        self.psi = psi
        self.iota = iota
        self.d_l = d_l
        self.z = z

        self.m1 = m1
        self.chi1 = chi1
        self.theta_s1 = theta_s1
        self.phi_s1 = phi_s1

        self.m2 = m2
        self.chi2 = chi2
        self.theta_s2 = theta_s2
        self.phi_s2 = phi_s2

        self.t_c = t_c
        self.phi_c = phi_c

    @property
    def long(self):
        return self.__long

    @long.setter
    def long(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__long = value
        else:
            self.__long = self.parameters['long']

    @property
    def lat(self):
        return self.__lat

    @lat.setter
    def lat(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__lat = value
        else:
            self.__lat = self.parameters['lat']

    @property
    def phil_l(self):
        return self.__phil_l

    @phil_l.setter
    def phil_l(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__phil_l = value
        else:
            try:
                self.__phil_l = self.parameters["phi_l"]
            except KeyError:
                print("GW source parameters do not contain PHI_L.")

    @property
    def theta_l(self):
        return self.__theta_l

    @theta_l.setter
    def theta_l(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__theta_l = value
        else:
            try:
                self.__theta_l = self.parameters["theta_l"]
            except KeyError:
                print("GW source parameters do not contain THETA_L.")

    @property
    def psi(self):
        return self.__psi

    @psi.setter
    def psi(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__psi = value
        else:
            try:
                self.__psi = self.parameters["psi"]
            except KeyError:
                print("GW source parameters do not contain PSI.")

    @property
    def iota(self):
        return self.__iota

    @iota.setter
    def iota(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__iota = value
        else:
            try:
                self.__iota = self.parameters["iota"]
            except KeyError:
                print("GW source parameters do not contain IOTA.")

    @property
    def d_l(self):
        return self.__d_l

    @d_l.setter
    def d_l(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__d_l = value
        else:
            try:
                self.__d_l = self.parameters["D_L"] * self.sim.constant["m/parsec"]
            except KeyError:
                print("GW source parameters do not contain D_L.")

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__z = value
        else:
            try:
                self.__z = self.parameters["z"]
            except KeyError:
                print("GW source parameters do not contain Z.")

    @property
    def m1(self):
        return self.__m1

    @m1.setter
    def m1(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__m1 = value
        else:
            try:
                self.__m1 = self.parameters["m1/massSun"] * self.sim.constant['massSun']
            except KeyError:
                print("GW source parameters do not contain M1.")

    @property
    def chi1(self):
        return self.__chi1

    @chi1.setter
    def chi1(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__chi1 = value
        else:
            try:
                self.__chi1 = self.parameters["chi1"]
            except KeyError:
                print("GW source parameters do not contain CHI1.")

    @property
    def theta_s1(self):
        return self.__theta_s1

    @theta_s1.setter
    def theta_s1(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__theta_s1 = value
        else:
            try:
                self.__theta_s1 = self.parameters["theta_s1"]
            except KeyError:
                print("GW source parameters do not contain THETA_S1.")

    @property
    def phi_s1(self):
        return self.__phi_s1

    @phi_s1.setter
    def phi_s1(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__phi_s1 = value
        else:
            try:
                self.__phi_s1 = self.parameters["phi_s1"]
            except KeyError:
                print("GW source parameters do not contain PHI_S1.")

    @property
    def m2(self):
        return self.__m2

    @m2.setter
    def m2(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__m2 = value
        else:
            try:
                self.__m2 = self.parameters["m2/massSun"] * self.sim.constant['massSun']
            except KeyError:
                print("GW source parameters do not contain M2.")

    @property
    def chi2(self):
        return self.__chi2

    @chi2.setter
    def chi2(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__chi2 = value
        else:
            try:
                self.__chi2 = self.parameters["chi2"]
            except KeyError:
                print("GW source parameters do not contain CHI2.")

    @property
    def theta_s2(self):
        return self.__theta_s2

    @theta_s2.setter
    def theta_s2(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__theta_s2 = value
        else:
            try:
                self.__theta_s2 = self.parameters["theta_s2"]
            except KeyError:
                print("GW source parameters do not contain THETA_S2.")

    @property
    def phi_s2(self):
        return self.__phi_s2

    @phi_s2.setter
    def phi_s2(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__phi_s2 = value
        else:
            try:
                self.__phi_s2 = self.parameters["phi_s2"]
            except KeyError:
                print("GW source parameters do not contain PHI_S2.")

    @property
    def t_c(self):
        return self.__t_c

    @t_c.setter
    def t_c(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__t_c = value
        else:
            try:
                self.__t_c = self.parameters["t_c"] * self.sim.constant['secs/year']
            except KeyError:
                print("GW source parameters do not contain T_C.")

    @property
    def phi_c(self):
        return self.__phi_c

    @phi_c.setter
    def phi_c(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__phi_c = value
        else:
            try:
                self.__phi_c = self.parameters["phi_c"]
            except KeyError:
                print("GW source parameters do not contain PHI_C.")

    @property
    def spin(self):
        return self.__spin

    @spin.setter
    def spin(self, value):
        if value is not None:
            if not isinstance(value, bool):
                print("Given value for SPIN can only be ON or OFF. Value from parameter dictionary is taken.")
            else:
                self.__spin = value
        else:
            try:
                self.__spin = self.parameters["spin"]
            except KeyError:
                print("GW source parameters do not contain SPIN.")

    @property
    def PN(self):
        return self.__PN

    @PN.setter
    def PN(self, value):
        if value is not None and isinstance(value, (int, float)):
            self.__PN = value
        else:
            self.__PN = self.parameters['PN']

    def chirp_mass(self, m1_user=None, m2_user=None):
        if (m1_user and m2_user) is not None:
            return (m1_user * m2_user) ** (3 / 5) * (m1_user + m2_user) ** (-1 / 5)
        elif (self.m1 and self.m2) is not None:
            return (self.m1 * self.m2) ** (3 / 5) * (self.m1 + self.m2) ** (-1 / 5)

    def reduced_mass(self, m1_user=None, m2_user=None):
        if (m1_user and m2_user) is not None:
            return (m1_user * m2_user) * (m1_user + m2_user) ** (-2)
        elif (self.m1 and self.m2) is not None:
            return (self.m1 * self.m2) * (self.m1 + self.m2) ** (-2)

    def theta(self, t_user=None):
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        else:
            t_temp = self.sim.t

        eta = self.reduced_mass()
        m = self.m1 + self.m2

        # create new time array which is t_c for time points larger than t_c and t for time points smller than t_c
        # this allows to set theta to zero for time points where t < t_c
        t_new = np.array([self.t_c if t_ > self.t_c else t_ for t_ in t_temp])

        return (self.sim.constant['c'] ** 3 * eta) / (5 * self.sim.constant['G'] * m) * (self.t_c - t_new)

    def phase_PN(self, t_user=None):
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        eta = self.reduced_mass()
        m = self.m1 + self.m2
        theta_temp = self.theta()

        if not self.spin:
            if self.PN == 2:
                return -2 / eta * theta_temp ** (5 / 8) \
                       - 2 * eta ** (-1) * (3715 / 8064 + 55 / 96 * eta) * theta_temp ** (3 / 8) \
                       - 2 * eta ** (-1) * (9275495 / 14450688 + (284875 / 258048) * eta + (
                        1855 / 2048) * eta ** 2) ** theta_temp ** (1 / 8) \
                       - 2 * eta ** (-1) * (-3 * pi / 4) * theta_temp ** (1 / 4)
            elif self.PN == 1:
                return -2 * eta ** (-1) * theta_temp ** (5 / 8) - 2 * eta ** (-1) * (
                        3715 / 8064 + 55 / 96 * eta) * theta_temp ** (3 / 8)
            else:
                return -2 * eta ** (-1) * theta_temp ** (5 / 8)
        elif self.spin:  # TODO
            if self.PN == 2:
                return
            elif self.PN == 1:
                return
            else:
                return

    def phase_orbital(self, t_user=None):
        """
        Function that computes the intrinsic orbital frequency as a sum of the
        phase at coalescence and the Post-Newtonian phase approximation.
        It is phase_orbital = 2*phi_orbital

        Here the values of the PN term depends on whether the spin is on or off.

        :param t_user: optional user input for the time interval

        :return: PHI = 2*phi_orbital = phi_c + phi_PN
        """
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t
        else:
            print('GW computation of orbital phase: Please give a time interval t_user.')

        return self.phi_c + self.phase_PN(t_user=t_temp)

    def phase_delta_pre(self, t_user=None):
        """
        Phase Correction for precession due to Post-Newtonian orbit-orbit and
        orbit-spin coupling.

        :param t_user:
        """
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        if self.spin:
            return 0
        else:
            # TODO: delta spin for spin on
            return

    def phase(self, t_user=None,
              orbital_phase_user=None,
              delta_phase_pre_user=None):
        """
        Function that gives the final value of the phase, copmrising from
        1.  phi_c, the phase of coalescence
        2.  phi_PN, the Post-Newtonian approximation of the phase (summing
                    to the intrinsic phase with 1.)
        3.  delta_phase_pre, the precession correction term

        :param t_user:
        :param orbital_phase_user:
        :param delta_phase_pre_user:
        :return:
        """
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        if orbital_phase_user is not None:
            orbital_phase_temp = orbital_phase_user
        else:
            orbital_phase_temp = self.phase_orbital(t_user=t_temp)

        if self.spin:
            if delta_phase_pre_user is not None:
                delta_phase_pre_temp = delta_phase_pre_user
            else:
                delta_phase_pre_temp = self.phase_delta_pre(t_user=t_temp)

        if not self.spin:
            return 2 * orbital_phase_temp
        else:
            if not delta_phase_pre_temp == 0 and len(orbital_phase_temp) == len(delta_phase_pre_temp):
                return 2 * orbital_phase_temp + delta_phase_pre_temp
            else:
                print(
                    "GW_Source computation of phase: intrinsic orbital phase and delta phase due to precession are not "
                    "of same length")

    def frequency_PN(self, t_user=None):
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        eta = self.reduced_mass()
        m = self.m1 + self.m2
        theta_temp = self.theta()

        if not self.spin:
            if self.PN == 2:
                return self.sim.constant['c'] ** 3 / (8 * self.sim.constant['G'] * m) * \
                       (theta_temp ** (-3 / 8)
                        + (743 / 2688 + 11 / 32 * eta) * theta_temp ** (-5 / 8)
                        + (1855099 / 14450688 + (56975 / 258048) * eta + (371 / 2048) * eta ** 2) ** theta_temp ** (
                                -7 / 8)
                        - (3 * pi / 10) * theta_temp ** (-3 / 4))
            elif self.PN == 1:
                return self.sim.constant['c'] ** 3 / (8 * self.sim.constant['G'] * m) * \
                       (theta_temp ** (-3 / 8)
                        + (743 / 2688 + 11 / 32 * eta) * theta_temp ** (-5 / 8))
            else:
                return self.sim.constant['c'] ** 3 / (8 * self.sim.constant['G'] * m) * theta_temp ** (-3 / 8)
        elif self.spin:  # TODO
            if self.PN == 2:
                return
            elif self.PN == 1:
                return
            else:
                return

    def amplitude(self, t_user=None):
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        eta = self.reduced_mass()
        m = self.m1 + self.m2
        f = self.frequency_PN(t_user=t_temp)

        return (4 * eta * (self.sim.constant['G'] * m) ** (5 / 3)) / (self.sim.constant['c'] ** 4 * self.d_l) * f ** (
                2 / 3)

    def h_0_plus_cross(self, t_user=None):
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        h_0_amplitues = dict()
        h_0_amplitues['h_0_plus'] = 0.5 * self.amplitude() * (1 + np.cos(self.iota) ** 2)
        h_0_amplitues['h_0_cross'] = self.amplitude() * np.cos(self.iota)
        return h_0_amplitues

    def h_plus_cross(self, t_user=None, frame='radiation'):
        if t_user is not None:
            if isinstance(t_user, np.ndarray):
                t_temp = t_user
            else:
                t_temp = np.array(t_user)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        h = dict()
        h['frame'] = frame
        h_plus_temp = self.h_0_plus_cross(t_user=t_temp)['h_0_plus'] * np.cos(self.phase(t_user=t_temp))
        h_cross_temp = self.h_0_plus_cross(t_user=t_temp)['h_0_cross'] * np.sin(self.phase(t_user=t_temp))

        if frame == 'source':
            h['h_plus'] = h_plus_temp
            h['h_cross'] = h_cross_temp
            return h
        elif frame == 'radiation':
            h['h_plus'] = h_plus_temp * np.cos(2 * self.psi) - h_cross_temp * np.sin(2 * self.psi)
            h['h_cross'] = h_plus_temp * np.sin(2 * self.psi) + h_cross_temp * np.cos(2 * self.psi)
            return h


class WaveformSimpleBinaries(WaveformClass):
    def __init__(self,
                 parameters=None,
                 filepath=None,
                 source=None,
                 sim=None,
                 long=None,
                 lat=None,
                 psi=None,
                 iota=None,
                 d_l=None,
                 z=None,
                 phi_0=None,
                 amplitude=None,
                 frequency=None,
                 frequency_dot=None):
        super().__init__(parameters,
                         filepath,
                         source,
                         sim)
        assert not (self.parameters is None and self.filepath is None), \
            "Please give either a path to a file with GW source parameters (input filepath) or a read-to-use " \
            "parameter dictionary. "

        self.long = long
        self.lat = lat
        self.psi = psi
        self.iota = iota
        self.d_l = d_l
        self.z = z
        self.phi_0 = phi_0
        self.frequency = frequency
        self.frequency_dot = frequency_dot
        self.amplitude = amplitude

    @property
    def long(self):
        return self.__long

    @long.setter
    def long(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__long = value
        else:
            self.__long = self.parameters['long']

    @property
    def lat(self):
        return self.__lat

    @lat.setter
    def lat(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__lat = value
        else:
            self.__lat = self.parameters['lat']

    @property
    def psi(self):
        return self.__psi

    @psi.setter
    def psi(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__psi = value
        else:
            try:
                self.__psi = self.parameters["psi"]
            except KeyError:
                print("GW source parameters do not contain PSI.")

    @property
    def iota(self):
        return self.__iota

    @iota.setter
    def iota(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__iota = value
        else:
            try:
                self.__iota = self.parameters["iota"]
            except KeyError:
                print("GW source parameters do not contain IOTA.")

    @property
    def d_l(self):
        return self.__d_l

    @d_l.setter
    def d_l(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__d_l = value
        else:
            try:
                self.__d_l = self.parameters["D_L"] * self.sim.constant["m/parsec"]
            except KeyError:
                print("GW source parameters do not contain D_L.")

    @property
    def z(self):
        return self.__z

    @z.setter
    def z(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__z = value
        else:
            try:
                self.__z = self.parameters["z"]
            except KeyError:
                print("GW source parameters do not contain Z.")

    @property
    def phi_0(self):
        return self.__phi_0

    @phi_0.setter
    def phi_0(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__phi_0 = value
        else:
            try:
                self.__phi_0 = self.parameters["phi_0"]
            except KeyError:
                print("GW source parameters do not contain PHI_0.")

    @property
    def frequency(self):
        return self.__frequency

    @frequency.setter
    def frequency(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__frequency = value
        else:
            try:
                self.__frequency = self.parameters["frequency"]
            except KeyError:
                print("GW source parameters do not contain FREQUENCY.")

    @property
    def frequency_dot(self):
        return self.__frequency_dot

    @frequency_dot.setter
    def frequency_dot(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__frequency_dot = value
        else:
            try:
                self.__frequency_dot = self.parameters["frequency_dot"]
            except KeyError:
                print("GW source parameters do not contain FREQUENCY_DOT.")

    @property
    def amplitude(self):
        return self.__amplitude

    @amplitude.setter
    def amplitude(self, value):
        if value is not None and isinstance(value, (float, int)):
            self.__amplitude = value
        else:
            try:
                self.__amplitude = self.parameters["amplitude"]
            except KeyError:
                m1 = self.parameters['m1'] * self.sim.constant['massSun']
                m2 = self.parameters['m2'] * self.sim.constant['massSun']
                m = m1 + m2
                eta_mass = m1 * m2 / m ** 2
                self.__amplitude = (2 * eta_mass * (self.sim.constant['G'] * m) ** (5 / 3)
                                    * (2 * pi * self.frequency / 2) ** (2 / 3)) \
                                   / (self.parameters['D_L'] * self.sim.constant['m/parsec'] *
                                      self.sim.constant['c'] ** 4)
                print("GW source parameters do not contain AMPLITUDE.")

    def phase(self, t=None,
              f=None,
              f_dot=None,
              phi_init=None):
        """

        """
        if t is not None:
            if isinstance(t, np.ndarray):
                t_temp = t
            else:
                t_temp = np.array(t)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        if f is not None:
            f_temp = f
        else:
            f_temp = self.frequency

        if f_dot is not None:
            f_dot_temp = f_dot
        else:
            f_dot_temp = self.frequency_dot

        if phi_init is not None:
            phi_0_temp = phi_init
        else:
            phi_0_temp = self.phi_0

        # return 2 * pi * (f_temp * t_temp + 0.5 * f_dot_temp * t_temp ** 2) - phi_0_temp
        return 2 * pi * (f_temp * t_temp + 0.5 * f_dot_temp * t_temp ** 2) + phi_0_temp

    def h_0_plus_cross(self, t=None):
        if t is not None:
            if isinstance(t, np.ndarray):
                t_temp = t
            else:
                t_temp = np.array(t)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        h_0_amplitues = dict()
        # h_0_amplitues['h_0_plus'] = -self.amplitude * (1 + np.cos(self.iota) ** 2)
        # h_0_amplitues['h_0_cross'] = -2 * self.amplitude * np.cos(self.iota)
        h_0_amplitues['h_0_plus'] = self.amplitude * (1 + np.cos(self.iota) ** 2)
        h_0_amplitues['h_0_cross'] = 2 * self.amplitude * np.cos(self.iota)
        return h_0_amplitues

    def h_plus_cross(self, t=None, frame='source'):
        if t is not None:
            if isinstance(t, np.ndarray):
                t_temp = t
            else:
                t_temp = np.array(t)
        elif self.sim.t is not None:
            t_temp = self.sim.t

        h = dict()
        h['frame'] = frame
        h_plus_temp = self.h_0_plus_cross(t=t_temp)['h_0_plus'] * np.cos(self.phase(t=t_temp))
        h_cross_temp = self.h_0_plus_cross(t=t_temp)['h_0_cross'] * np.sin(self.phase(t=t_temp))

        if frame == 'source':
            h['h_plus'] = h_plus_temp
            h['h_cross'] = h_cross_temp
            return h
        elif frame == 'radiation':
            h['h_plus'] = h_plus_temp * np.cos(2 * self.psi) - h_cross_temp * np.sin(2 * self.psi)
            h['h_cross'] = h_plus_temp * np.sin(2 * self.psi) + h_cross_temp * np.cos(2 * self.psi)
            return h

    def h(self,
          t=None):
        if t is not None:
            if isinstance(t, np.ndarray):
                t_temp = t
            else:
                t_temp = np.array(t)
        elif self.sim.t is not None:
            t_temp = self.sim.t
        h_temp = self.h_plus_cross(t=t_temp, frame='source')
        return self.get_h(h_plus=h_temp['h_plus'], h_cross=h_temp['h_cross'])


if __name__ == '__main__':
    src = 'GalBin'

    if src == 'MBHB':
        MBHB_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                     'MBHB_parameters.txt')
        MBHB_parameters = read_parameter_fun(MBHB_filepath)
        MBHB = WaveformMBHB(parameters=MBHB_parameters)
        t_obs = MBHB.t

        PN_order = MBHB.PN
        phi_c = MBHB.phi_c
        phase_PN = MBHB.phase_PN()
        phase_orbital = MBHB.phase_orbital()
        theta = MBHB.theta()

        frequency_PN = MBHB.frequency_PN()
        A = MBHB.amplitude()
        hphc = MBHB.h_plus_cross()

        plt.close()
        fig1 = plt.figure(1)
        plt.subplot(211)

        plt.plot(MBHB.t, hphc['h_plus'], label='h_plus')
        plt.grid()
        plt.title('Polarizations MBHB')
        plt.xlabel('time')
        plt.subplot(212)
        plt.plot(MBHB.t, hphc['h_cross'], label='h_cross')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('strain')
        plt.show()

    elif src == 'GalBin':
        ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test',
                                'resources', 'Waveform')
        hpref = np.loadtxt(os.path.join(ref_path, 'GalBin_hp.out'))
        hcref = np.loadtxt(os.path.join(ref_path, 'GalBin_hc.out'))
        tref = np.loadtxt(os.path.join(ref_path, 'GalBin_t.out'))
        sim = Simulation(t=tref)

        GalBin_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                       'GalBin_parameters.txt')
        GalBin_parameters = read_parameter_fun(GalBin_filepath)
        GalBin = WaveformSimpleBinaries(parameters=GalBin_parameters, sim=sim)
        h = GalBin.h()

        hphc_GalBin = GalBin.h_plus_cross()
        hp_diff = hpref - hphc_GalBin['h_plus']
        hc_diff = hcref - hphc_GalBin['h_cross']

        # fig1 = plt.figure(1)
        # plt.subplot(211)
        # plt.plot(tref, hphc_GalBin['h_plus'], 'blue', label='h_plus')
        # plt.plot(tref, hpref, 'black', Linestyle=':', label='h_plus_ref')
        # plt.grid()
        # plt.title('Polarizations GalBin')
        # plt.ylabel('strain hp (-)')
        # plt.legend()
        # # plt.xlabel('time')
        # plt.subplot(212)
        # plt.plot(tref, hphc_GalBin['h_cross'], 'blue', label='h_cross')
        # plt.plot(tref, hcref, 'black', Linestyle=':', label='h_cross_ref')
        # plt.grid()
        # plt.xlabel('time')
        # plt.ylabel('strain hc (-)')
        # plt.show()

        fig2 = plt.figure(2)
        plt.subplot(211)
        plt.plot(tref[2000:2500], hphc_GalBin['h_plus'][2000:2500], 'blue', label='h_plus (Forward Model)')
        plt.plot(tref[2000:2500], hpref[2000:2500], 'black', Linestyle=':', label='h_plus (SyntheticLISA)')
        plt.grid()
        plt.title('Polarizations GalBin 256000s - 320000s')
        plt.ylabel('strain hp (-)')
        plt.legend(loc='upper right')
        # plt.xlabel('time')
        plt.subplot(212)
        plt.plot(tref[2000:2500], hphc_GalBin['h_cross'][2000:2500], 'red', label='h_cross (Forward Model)')
        plt.plot(tref[2000:2500], hcref[2000:2500], 'black', Linestyle=':', label='h_cross (SyntheticLISA)')
        plt.grid()
        plt.xlabel('time')
        plt.ylabel('strain hc (-)')
        plt.show()

        # fig2 = plt.figure(2)
        # plt.subplot(211)
        # plt.plot(GalBin.sim.t, hphc_GalBin['h_plus'], label='h_plus')
        # plt.grid()
        # plt.title('Polarizations GalBin')
        # plt.xlabel('time')
        # plt.subplot(212)
        # plt.plot(GalBin.sim.t, hphc_GalBin['h_cross'], label='h_cross')
        # plt.grid()
        # plt.xlabel('time')
        # plt.ylabel('strain')
        # plt.show()
