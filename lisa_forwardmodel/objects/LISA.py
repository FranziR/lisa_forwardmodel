'''
Date: 20.07.2020
Author: Franziska Riegger
Revision Date:
Revision Author:
'''

import abc
import os
from math import pi as pi

import matplotlib.pyplot as plt
import numpy as np

from lisa_forwardmodel.objects.Simulation import Simulation
from lisa_forwardmodel.utils.LISAutils import checkLink
from lisa_forwardmodel.utils.checkInput import checkInput
from lisa_forwardmodel.utils.readParameters import read_parameter_fun
from lisa_forwardmodel.utils.utils import get_Euler


class SpaceInterferometer(abc.ABC):
    def __init__(self,
                 parameters=None,
                 filepath=None,
                 sim=None):
        assert not (parameters is None and filepath is None), \
            "Please give either a path to a file with LISA parameters (input filepath) or a read-to-use parameter dictionary."

        self.filepath = filepath
        self.parameters = parameters
        self.sim = sim

    @property
    def sim(self):
        return self.__sim

    @sim.setter
    def sim(self, value):
        if value is not None:
            self.__sim = value
        else:
            self.__sim = 0

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
    @abc.abstractmethod
    def L(self):
        return self.__L

    @L.setter
    @abc.abstractmethod
    def L(self, value):
        pass

    @abc.abstractmethod
    def get_p(self):
        pass

    @abc.abstractmethod
    def get_n(self):
        pass

    @abc.abstractmethod
    def get_constellation(self):
        pass

    @abc.abstractmethod
    def L_light(self):
        pass

class EccentricLISA(SpaceInterferometer):
    """A LISA class with eccentric orbits.

    ATTRIBUTES:
        L       (array):        LISA arm lengths in sec
        omega (float):          rotation velocity of LISA guiding centre around Sun in rad/sec
        eta_zero (float):       initial position of rotation of LISA guiding centre around Sun in rad
        xi_zero (float):        initial position of rotation of spacecrafts around LISA guiding centre in rad
        ec (array):             eccentricity of orbits

    METHODS:
        sigma (float):          computes initial angular position of spacecrafts wrt LISA guiding centre
        alpha (array):          computes angular position of LISA guiding centre wrt Sun in rad at time t
        beta (array):           angular anomaly of spacecraft over time
        get_ploc (array):       returns spacecraft position in local LISA frame (position wrt to LISA barycenter) in sec
        get_p (array):          returns spacecraft position in SSB frame in sec
        get_constellation (list): returns possible combinations of sending and receiving crafts with transmitting
                                arm index for Doppler measurement
        get_n (array):          computes light propagation direction (normal unit vectors of LISA arms)
        L_light (array):        gives the evolution of the light propagation time between two crafts
                                (due to direction of propagation: co- and counterclockwise)
    """

    def __init__(self,
                 parameters=None,
                 filepath=None,
                 L=None,
                 omega=None,
                 eta_zero=None,
                 xi_zero=None,
                 ec=None,
                 sim=None,
                 swap=None):
        """
        Initialises EccentricLISA class.

        :param parameters: (dict) LISA simulation parameters
        :param filepath: (str) path from where to read in .txt file that contains LISA parameters
        :param L: (array) LISA armlengths in sec
        :param omega: (float) angular velocity of LISA guiding center rotation around SUN
        :param eta_zero: (float) initial position of LISA
        :param xi_zero: (float) initial position of LISA
        :param ec: (array) eccentricity of LISA spacecraft orbits
        :param sim: (simulation object) simulation object that contains temporal information
        :param swap:
        """
        super().__init__(parameters,
                         filepath,
                         sim)

        assert not (self.parameters is None and self.filepath is None), \
            "Please give either a path to a file with LISA parameters (input filepath) or a read-to-use parameter dictionary."

        self.swap = swap
        self.L = L
        self.ec = ec
        self.omega = omega
        self.xi_zero = xi_zero
        self.eta_zero = eta_zero

    @property
    def swap(self):
        return self.__swap

    @swap.setter
    def swap(self, value):
        if value is not None and isinstance(value, bool):
            self.__swap = value
        else:
            self.__swap = self.parameters['swap']

    @property
    def L(self):
        return self.__L

    @L.setter
    def L(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                self.__L = value
            else:
                self.__L = np.array(value)
        else:
            try:
                self.__L = self.parameters['Lsec']
            except KeyError:
                self.__L = np.array(self.parameters['L']) / self.sim.constant['c']
        assert len(self.__L) == 3, "3D array must be given for the LISA arm length."

    @property
    def omega(self):
        return self.__omega

    @omega.setter
    def omega(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__omega = temp
        else:
            self.__omega = self.parameters['Omega'] / self.sim.constant["secs/year"]

    @property
    def eta_zero(self):
        return self.__eta_zero

    @eta_zero.setter
    def eta_zero(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__eta_zero = temp
        else:
            if self.swap:
                # eta_zero by Cornish and Rubbo:
                # eta_zero = kappa
                # where kappa = 0
                self.__eta_zero = self.parameters['kappa']
            else:
                self.__eta_zero = self.parameters['eta_zero']

    @property
    def xi_zero(self):
        return self.__xi_zero

    @xi_zero.setter
    def xi_zero(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__xi_zero = temp
        else:
            if self.swap:
                # xi_zero by Cornish and Rubbo:
                # xi_zero = 3*pi/2 - kappa + lambda
                # where kappa = 0 and lambda = 3*pi/4
                self.__xi_zero = 3 * pi / 2 - self.parameters['kappa'] + self.parameters['lambda']
            else:
                self.__xi_zero = self.parameters['xi_zero']

    @property
    def ec(self):
        return self.__ec

    @ec.setter
    def ec(self, value):
        if value is not None and isinstance(value, np.ndarray):
            self.__ec = value
        else:
            self.__ec = self.L / (2 * np.sqrt(3) * self.parameters['Rsec'])
        assert len(self.__ec) == 3, "Eccentricity has to be a 3D array."

    def sigma(self, swap=None):
        ind, temp = checkInput(swap, bool, (str, int))
        if ind:
            sw = temp
        else:
            sw = self.swap
        sigma = np.zeros((3,))

        # TODO: check SyntheticLISA!!
        if not sw:
            sigma[0] = 3 * pi / 2  # spacecraft 1
            sigma[1] = 3 * pi / 2 - 4 * pi / 3  # spacecraft 2 (i = 3)
            sigma[2] = 3 * pi / 2 - 2 * pi / 3  # spacecraft 3 (i = 2)
        else:
            for i in range(1, 4):
                sigma[i - 1] = 3 * pi / 2 - 2 * (i - 1) * pi / 3
        return sigma

    def alpha(self, t=None):
        """
        Computes rotational motion of LISA constellation.

        :param t: (int/float/list/array)   time at which alpha is to be computed
        :return:
            alpha (array):  temporal evolution of angular position of LISA
        """
        if (self.omega, self.eta_zero) is not None:
            ind, t_temp = checkInput(t, np.ndarray, (list, tuple, int, float))
            if ind:
                return self.omega * t_temp + self.eta_zero
            elif self.sim.t is not None:
                return self.omega * self.sim.t + self.eta_zero

    def beta(self):
        """
        Computes angular anomaly of each spacecraft.

        :return:
            beta: (array)
        """
        beta_temp = np.zeros((3,))
        sigma_temp = self.sigma()
        # lam = self.eta_zero + self.xi_zero - 3*pi/2
        for i in range(0, 3):
            beta_temp[i] = self.eta_zero + self.xi_zero - sigma_temp[i]
            # beta_temp[i] = lam - sigma_temp[i]
        return beta_temp

    def get_ploc(self, t=None, approx=2):
        """
        Computes spacecraft positions in local frame (wrt LISA guiding center).

        :param t: (int/float/list/array)    time at which p_loc is to be computed
        :param approx: (int)                gives order of accuracy of approximation of orbits
                                            (can only take values 1 and 2)
        :return:
            p_loc: (dict)   dictionary with temporal evolution of position
        """
        ind, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind:
            t_l = self.sim.t

        assert (approx == 1 or approx == 2), \
            "Approximation accuracy can only be of order 1 (approx = 1) or 2 (approx = 2)."

        p_loc = dict()
        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        beta_temp = self.beta()
        if approx == 1:
            for i in range(0, 3):
                p_loc[str(i + 1)] = np.zeros((3, dim_t))
                p_loc[str(i + 1)][0, :] = self.parameters['Rsec'] * self.ec[i] * \
                                          (np.sin(self.alpha(t_l)) * np.cos(self.alpha(t_l)) * np.sin(beta_temp[i]) \
                                           - (1 + np.sin(self.alpha(t_l)) ** 2) * np.cos(beta_temp[i]))
                p_loc[str(i + 1)][1, :] = self.parameters['Rsec'] * self.ec[i] * \
                                          (np.sin(self.alpha(t_l)) * np.cos(self.alpha(t_l)) * np.cos(beta_temp[i]) \
                                           - (1 + np.cos(self.alpha(t_l)) ** 2) * np.sin(beta_temp[i]))
                p_loc[str(i + 1)][2, :] = -self.parameters['Rsec'] * np.sqrt(3) * self.ec[i] * \
                                          np.cos(self.alpha(t_l) - beta_temp[i])
        elif approx == 2:
            for i in range(0, 3):
                p_loc[str(i + 1)] = np.zeros((3, dim_t))
                p_loc[str(i + 1)][0, :] = 0.5 * self.parameters['Rsec'] * self.ec[i] \
                                          * (np.cos(2 * self.alpha(t_l) - beta_temp[i]) - 3 * np.cos(beta_temp[i])) \
                                          + 0.125 * self.parameters['Rsec'] * self.ec[i] ** 2 \
                                          * (3 * np.cos(3 * self.alpha(t_l) - 2 * beta_temp[i])
                                             - 5 * (2 * np.cos(self.alpha(t_l))
                                                    + np.cos(self.alpha(t_l) - 2 *beta_temp[i])))
                p_loc[str(i + 1)][1, :] = 0.5 * self.parameters['Rsec'] * self.ec[i] \
                                          * (np.sin(2 * self.alpha(t_l) - beta_temp[i])
                                             - 3 * np.sin(beta_temp[i])) \
                                          + 0.125 * self.parameters['Rsec'] * self.ec[i] ** 2 \
                                          * (3 * np.sin(3 * self.alpha(t_l) - 2 * beta_temp[i])
                                             - 5 * (2 * np.sin(self.alpha(t_l))
                                                    - np.sin(self.alpha(t_l) - 2 * beta_temp[i])))
                p_loc[str(i + 1)][2, :] = -self.parameters['Rsec'] * np.sqrt(3) * self.ec[i] \
                                          * np.cos(self.alpha(t_l) - beta_temp[i]) \
                                          + self.parameters['Rsec'] * np.sqrt(3) * self.ec[i] ** 2 \
                                          * (np.cos(self.alpha(t_l) - beta_temp[i]) ** 2
                                             + 2.0 * np.sin(self.alpha(t_l) - beta_temp[i]) ** 2)
        return p_loc

    def get_p(self, t=None, approx=2):
        """
        Computes orbits of spacecrafts in SSB frame.

        :param t: (int/float/list/array)   time at which p is to be computed
        :returns:
            p: (dict)   dictionary with orbits of each craft
        """
        assert (approx == 1 or approx == 2), \
            "Approximation accuracy can only be of order 1 (approx = 1) or 2 (approx = 2)."

        ind, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind:
            t_l = self.sim.t

        p = dict()
        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        p_loc = self.get_ploc(t=t_l, approx=2)
        for i in range(0, 3):
            p[str(i + 1)] = np.zeros((3, dim_t))
            p[str(i + 1)][0, :] = self.parameters['Rsec'] * np.cos(self.alpha(t_l)) + p_loc[str(i + 1)][0, :]
            p[str(i + 1)][1, :] = self.parameters['Rsec'] * np.sin(self.alpha(t_l)) + p_loc[str(i + 1)][1, :]
            p[str(i + 1)][2, :] = p_loc[str(i + 1)][2, :]
        return p

    def get_constellation(self, swap=None):
        """
        Returns tuple with triples of sending/receiving crafts and transmitting link.
        :return:
        """
        ind_sw, temp = checkInput(swap, bool, (str, int))
        if ind_sw:
            sw = temp
        else:
            sw = self.swap

        if sw:
            return [[2, 1, 3], [1, 3, 2], [3, 2, 1], [3, -1, 2], [2, -3, 1], [1, -2, 3]]
        else:
            return [[3, 1, 2], [1, 2, 3], [2, 3, 1], [2, -1, 3], [3, -2, 1], [1, -3, 2]]

    def get_n(self, t=None):
        """
        Computes the light propagation direction (unit normal vector of LISA arms).

        :param t: (int/float/list/array)   time at which n is to be computed
        :return:
            n: (dict)   dictionary with unit normal vector
        """
        ind, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind:
            t_l = self.sim.t

        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        n = dict()

        p = self.get_p(t=t_l)
        for i in range(1, 4):
            n[str(i)] = np.zeros((3, dim_t))

        slr = self.get_constellation()
        for i in range(0, len(slr)):
            s = slr[i][0]
            l = slr[i][1]
            r = slr[i][2]

            n_temp = p[str(r)] - p[str(s)]
            norm = np.sqrt(np.sum(n_temp ** 2, axis=0))
            n[str(l)] = n_temp / norm
        return n

    def L_light(self, t=None, link=None, accurate=True):
        """
        Computes the light propagation times between two crafts.

        :param t: (int/float/list/array)   time at which L_light is to be computed
        :param link: (int)  link along which propagation time is to be computed
        :param accurate: (bool) indicator whether accurate comutation (accrate = True) is used or not (accurate = false)
        :return:
            L_light: (dict) dictionary with light propagation times
        """
        ind, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind:
            t_l = self.sim.t
        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        if link == 'all' or link is None:
            slr = self.get_constellation()
        else:
            assert checkLink(link), "Wrong link given."
            slr = [[0, link, 0]]

        L_lightprog = dict()

        if not accurate:
            for i in range(0, len(slr)):
                l = slr[i][1]
                L_lightprog[str(l)] = np.zeros((dim_t,))
                L_lightprog[str(l)] = self.L[np.abs(l) - 1] * np.ones((dim_t,))
            return L_lightprog
        else:
            delta = np.zeros((3,))
            if not self.swap:
                delta[0] = self.xi_zero
                delta[1] = self.xi_zero + 4 * pi / 3
                delta[2] = self.xi_zero + 2 * pi / 3
            else:
                delta[0] = self.xi_zero
                delta[1] = self.xi_zero + 2 * pi / 3
                delta[2] = self.xi_zero + 4 * pi / 3

            for i in range(0, len(slr)):
                l = slr[i][1]
                L_lightprog[str(l)] = np.zeros((dim_t,))
                # if l > 0:
                L_t = self.L[np.abs(l) - 1] + \
                      1 / 32 * self.ec[np.abs(l) - 1] * self.L[np.abs(l) - 1] \
                    * np.sin(3 * self.omega * t_l - 3 * self.xi_zero) \
                    + (np.sign(l) * self.omega * self.parameters['Rsec'] * self.L[np.abs(l) - 1]
                         - 15 / 32 * self.ec[np.abs(l) - 1] * self.L[np.abs(l) - 1]) \
                    * np.sin(self.omega * t_l - delta[np.abs(l) - 1])
                L_lightprog[str(l)] = L_t
            return L_lightprog


class CircularLISA(SpaceInterferometer):
    """A LISA class with circular orbits. Based on paper of Krolak et al.

    ATTRIBUTES:
        zeta    (float):        initial inclination of the LISA plane wrt ecliptic
        L       (array):        LISA arm lengths in sec
        omega_eta (float):      rotation velocity of LISA guiding centre around Sun in rad/sec
        omega_xi (float):       rotation velocity of spacecrafts around LISA guiding centre in rad/sec
        eta_zero (float):       initial position of rotation of LISA guiding centre around Sun in rad
        xi_zero (float):        initial position of rotation of spacecrafts around LISA guiding centre in rad

    METHODS:
        sigma (float):          computes initial angular position of spacecrafts wrt LISA guiding centre
        eta (array):            computes angular position of LISA guiding centre wrt Sun in rad at time t
        xi (array):             computes angular position of spacecrafts wrt LISA guiding centre in rad at time t
        get_ploc (array):       returns spacecraft position in local LISA frame (position wrt to LISA barycenter) in sec
        get_p (array):          returns spacecraft position in SSB frame in sec
        get_constellation (list): returns possible combinations of sending and receiving crafts with transmitting
                                arm index for Doppler measurement
        get_n (array):          computes light propagation direction (normal unit vectors of LISA arms)
        L_light (array):        gives the evolution of the light propagation time between two crafts
                                (due to direction of propagation: co- and counterclockwise)
    """

    def __init__(self,
                 parameters=None,
                 filepath=None,
                 L=None,
                 zeta=None,
                 omega=None,
                 eta_zero=None,
                 xi_zero=None,
                 swap=None,
                 sim=None):
        """
        Initialises CircularLISA class.
        :param parameters: (dict) LISA simulation parameters
        :param filepath: (str) path from where to read in .txt file that contains LISA parameters
        :param L: (array) LISA armlengths in sec
        :param omega: (float) angular velocity of LISA guiding center rotation around SUN
        :param eta_zero: (float) initial position of LISA
        :param xi_zero: (float) initial position of LISA
        :param zeta: (float) inclination of LISA wrt ecliptic
        :param swap:
        :param sim: (Simulation object) contains all information about tempora simualtion
        """
        super().__init__(parameters,
                         filepath,
                         sim)

        assert not (self.parameters is None and self.filepath is None), \
            "Please give either a path to a file with LISA parameters " \
            "(input filepath) or a read-to-use parameter dictionary."

        self.swap = swap
        self.L = L
        self.zeta = zeta  # initial inclination of LISA orbital plane
        self.omega_eta = omega  # rotation of guiding centre around Sun
        self.eta_zero = eta_zero
        try:
            self.omega_xi = -1 * omega  # motion of spacecraft around guiding centre
        except TypeError:
            self.omega_xi = None
        self.xi_zero = xi_zero

    @property
    def swap(self):
        return self.__swap

    @swap.setter
    def swap(self, value):
        if value is not None and isinstance(value, bool):
            self.__swap = value
        else:
            self.__swap = self.parameters['swap']

    @property
    def L(self):
        return self.__L

    @L.setter
    def L(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                self.__L = value
            else:
                self.__L = np.array(value)
        else:
            try:
                self.__L = self.parameters['Lsec']
            except KeyError:
                self.__L = np.array(self.parameters['L']) / self.sim.constant['c']
        assert len(self.__L) == 3, "3D array must be given for the LISA arm length."

    @property
    def zeta(self):
        return self.__zeta

    @zeta.setter
    def zeta(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__zeta = temp
        else:
            self.__zeta = self.parameters['zeta']

    @property
    def omega_eta(self):
        return self.__omega_eta

    @omega_eta.setter
    def omega_eta(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__omega_eta = temp
        else:
            self.__omega_eta = self.parameters['Omega'] / self.sim.constant["secs/year"]

    @property
    def omega_xi(self):
        return self.__omega_xi

    @omega_xi.setter
    def omega_xi(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__omega_xi = temp
        else:
            self.__omega_xi = -self.parameters['Omega'] * self.sim.constant["years/sec"]

    @property
    def eta_zero(self):
        return self.__eta_zero

    @eta_zero.setter
    def eta_zero(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__eta_zero = temp
        else:
            if self.swap:
                # eta_zero by Cornish and Rubbo:
                # eta_zero = kappa
                # where kappa = 0
                self.__eta_zero = self.parameters['kappa']
            else:
                self.__eta_zero = self.parameters['eta_zero']

    @property
    def xi_zero(self):
        return self.__xi_zero

    @xi_zero.setter
    def xi_zero(self, value):
        ind, temp = checkInput(value, float, int)
        if ind:
            self.__xi_zero = temp
        else:
            if self.swap:
                # xi_zero by Cornish and Rubbo:
                # xi_zero = 3*pi/2 - kappa + lambda
                # where kappa = 0 and lambda = 3*pi/4
                self.__xi_zero = 3 * pi / 2 - self.parameters['kappa'] + self.parameters['lambda']
            else:
                self.__xi_zero = self.parameters['xi_zero']

    def sigma(self, swap=None):
        ind, temp = checkInput(swap, bool, (str, int))
        if ind:
            sw = temp
        else:
            sw = self.swap
        sigma = np.zeros((3,))

        if sw:
            sigma[0] = 3 * pi / 2  # spacecraft 1
            sigma[1] = 3 * pi / 2 - 4 * pi / 3  # spacecraft 2 (i = 3)
            sigma[2] = 3 * pi / 2 - 2 * pi / 3  # spacecraft 3 (i = 2)
        else:
            for i in range(1, 4):
                sigma[i - 1] = 3 * pi / 2 - 2 * (i - 1) * pi / 3
        return sigma

    def eta(self, t=None):
        """
        Computes angular position of LISA guiding centre wrt Sun in rad at time t.

        :param t: (int/float/list/array)   time at which eta is to be computed
        :return:
            eta (array):    array (of length len(t)) that contains eta at given times t
        """
        if (self.omega_eta, self.eta_zero) is not None:
            ind, t_temp = checkInput(t, np.ndarray, (list, tuple, int, float))
            if ind:
                return self.omega_eta * t_temp + self.eta_zero
            elif self.sim.t is not None:
                return self.omega_eta * self.sim.t + self.eta_zero

    def xi(self, t=None):
        """
        Computes angular position of spacecrafts wrt LISA guiding centre in rad at time t.

        :param t: (int/float/list/array)   time at which eta is to be computed
        :return:
            xi (array): array (of length len(t)) that contains xi at given times t
        """
        if (self.omega_xi, self.xi_zero) is not None:
            ind, t_temp = checkInput(t, np.ndarray, (list, tuple, int, float))
            if ind:
                return self.omega_xi * t + self.xi_zero
            elif self.sim.t is not None:
                return self.omega_xi * self.sim.t + self.xi_zero

    def get_ploc(self, t=None):
        """
        Computes spacecraft positions in local frame (wrt LISA guiding center).

        :param t: (int/float/list/array)   time at which p_loc is to be computed
        :return:
            p_loc: (dict)   dictionary with temporal evolution of position
        """
        ind_t, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind_t:
            t_l = self.sim.t

        p_loc = dict()
        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        sig = self.sigma()
        Lsec3 = self.parameters['Lsec'] / np.sqrt(3)
        for i in range(0, 3):
            p_loc[str(i + 1)] = np.zeros((dim_t, 3, 1))
            p_loc[str(i + 1)][:, 0, 0] = -Lsec3[i] * np.cos(2 * sig[i])
            p_loc[str(i + 1)][:, 1, 0] = Lsec3[i] * np.sin(2 * sig[i])
            p_loc[str(i + 1)][:, 2, 0] = 0

            p_loc[str(i + 1)] = np.matmul(get_Euler(self.eta(t_l), self.xi(t_l), self.zeta),
                                          p_loc[str(i + 1)])
        return p_loc

    def get_constellation(self, swap=None):
        """
        Returns tuple with triples of sending/receiving crafts and transmitting link.
        :return:
        """
        ind_sw, temp = checkInput(swap, bool, (str, int))
        if ind_sw:
            sw = temp
        else:
            sw = self.swap
        if sw:
            return [[2, 1, 3], [1, 3, 2], [3, 2, 1], [3, -1, 2], [2, -3, 1], [1, -2, 3]]
        else:
            return [[3, 1, 2], [1, 2, 3], [2, 3, 1], [2, -1, 3], [3, -2, 1], [1, -3, 2]]

    def get_p(self, t=None):
        """
        Computes orbits of spacecrafts in SSB frame.

        :param t: (int/float/list/array)   time at which p is to be computed
        :returns:
            p: (dict)   dictionary with orbits of each craft
        """
        ind_t, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind_t:
            t_l = self.sim.t

        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        p = dict()
        p_loc = self.get_ploc(t=t_l)

        for i in range(0, 3):
            p[str(i + 1)] = np.zeros((3, dim_t))
            p[str(i + 1)][0, :] = self.parameters['Rsec'] * np.cos(self.eta(t_l)) + p_loc[str(i + 1)][:, 0, 0]
            p[str(i + 1)][1, :] = self.parameters['Rsec'] * np.sin(self.eta(t_l)) + p_loc[str(i + 1)][:, 1, 0]
            p[str(i + 1)][2, :] = p_loc[str(i + 1)][:, 2, 0]
        return p

    def get_n(self, t=None):
        """
        Computes the light propagation direction (unit normal vector of LISA arms).

        :param t: (int/float/list/array)   time at which n is to be computed
        :return:
            n: (dict)   dictionary with unit normal vector
        """
        ind_t, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind_t:
            t_l = self.sim.t

        n = dict()
        # n_temp = dict()
        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        sig = self.sigma()

        for i in range(0, 3):
            n[str(i + 1)] = np.zeros((3, dim_t))
            n[str(i + 1)][0, :] = np.sin(self.eta(t_l)) * np.cos(self.xi(t_l) + sig[i]) \
                                  - np.cos(self.eta(t_l)) * np.sin(self.zeta) \
                                  * np.sin(self.xi(t_l) + sig[i])
            n[str(i + 1)][1, :] = -np.cos(self.eta(t_l)) * np.cos(self.xi(t_l) + sig[i]) \
                                  - np.sin(self.eta(t_l)) * np.sin(self.zeta) \
                                  * np.sin(self.xi(t_l) + sig[i])
            n[str(i + 1)][2, :] = np.cos(self.zeta) * np.sin(sig[i] + self.xi(t_l))
        n['-1'] = -n['1']
        n['-2'] = -n['2']
        n['-3'] = -n['3']
        return n

    def L_light(self, t=None, link=None, accurate=True):
        """
        Computes the light propagation times between two crafts.

        :param t: (int/float/list/array)   time at which L_light is to be computed
        :param link: (int)  link along which propagation time is to be computed
        :param accurate: (bool) indicator whether accurate comutation (accrate = True) is used or not (accurate = false)
        :return:
            L_light: (dict) dictionary with light propagation times
        """
        ind, t_l = checkInput(t, np.ndarray, (list, tuple, int, float))
        if not ind:
            t_l = self.sim.t
        try:
            dim_t = len(t_l)
        except TypeError:
            dim_t = 1

        if not link == 'all' and link is not None:
            assert checkLink(link), "Wrong link given."
            slr = [[0, link, 0]]
        else:
            slr = self.get_constellation()

        L_lightprog = dict()
        if not accurate:
            for i in range(0, len(slr)):
                l = slr[i][1]
                L_lightprog[str(l)] = np.zeros((dim_t,))
                L_lightprog[str(l)] = self.L[np.abs(l) - 1] * np.ones((dim_t,))
            return L_lightprog
        else:
            delta = np.zeros((3,))
            if not self.swap:
                delta[0] = self.xi_zero
                delta[1] = self.xi_zero + 4 * pi / 3
                delta[2] = self.xi_zero + 2 * pi / 3
            else:
                delta[0] = self.xi_zero
                delta[1] = self.xi_zero + 2 * pi / 3
                delta[2] = self.xi_zero + 4 * pi / 3

            for i in range(0, len(slr)):
                l = slr[i][1]
                L_lightprog[str(l)] = np.zeros((dim_t,))
                L_lightprog[str(l)] = self.L[np.abs(l) - 1] + np.sign(l) \
                                      * self.omega_eta * self.parameters['Rsec'] \
                                      * self.L[np.abs(l) - 1] \
                                      * np.sin(self.omega_eta * t_l - delta[np.abs(l) - 1])
            return L_lightprog


if __name__ == '__main__':
    sim = Simulation()

    LISA_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                                 'LISA_parameters.txt')
    LISA_parameters = read_parameter_fun(LISA_filepath)
    LISA = CircularLISA(filepath=LISA_filepath, sim=sim)
    # p_trail = LISA.get_p()
    # L_light = LISA.L_light(link = 1)/sim.constant['c']
    # t_sim = LISA.sim.t

    # LISAEcc = EccentricLISA(filepath=LISA_filepath, sim=sim)
    # b = LISAEcc.beta()
    # p2 = LISAEcc.get_p()['2'] / sim.constant['c']
    # L = LISAEcc.L
    # L_light = LISAEcc.L_light(link=1)['1']
    #
    # ref_path = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'test',
    #                         'Orbit_SytheticLISA')
    # L_light_ref = np.loadtxt(os.path.join(ref_path, 'LISAEcc_L.out'))
    #
    # fig1 = plt.figure(1)
    # plt.plot(LISAEcc.sim.t, L_light, 'blue', label='LISA Franzi')
    # plt.plot(LISAEcc.sim.t, L_light, 'black', Linestyle=':', label='LISA Ref')
    # plt.grid()
    # plt.xlabel('time in sec')
    # plt.ylabel('light travel time (l_32)')
    # plt.title('Light travel time from spacecraft 3 to 2 (link 1)')
    # plt.legend()
    # plt.show()

    print('Finish')
    # fig1 = plt.figure(1)
    # plt.plot(t_sim, L_light)
    # plt.grid()
    # plt.xlabel('time in sec')
    # plt.ylabel('light travel time (l_32)')
    # plt.title('Light travel time from spacecraft 3 to 2 (link 1)')
    # plt.show()

    # t = LISA.t
    # p_local_val = LISA.p_local(paper = "vallisneri")
    # p_local = LISA.p_local()
    # p_val = LISA.p(paper = "vallisneri")
    # p = LISA.p()
    #
    # n = LISA.n()
    #
    # plt.close()
    # fig1 = plt.figure(1)
    # ax = plt.axes(projection='3d')
    # ax.plot3D(p_local_val['p_1'][0, :], p_local['p_1'][1, :], p_local['p_1'][2, :],
    #           'blue', label='spacecraft 0 (Vallis.)')
    # ax.plot3D(p_local_val['p_2'][0, :], p_local['p_2'][1, :], p_local['p_2'][2, :],
    #           'red', label='spacecraft 1 (Vallis.)')
    # ax.plot3D(p_local_val['p_3'][0, :], p_local['p_3'][1, :], p_local['p_3'][2, :],
    #           'green', label='spacecraft 2 (Vallis.)')
    #
    # ax.plot3D(p_local['p_1'][0, :], p_local['p_1'][1, :], p_local['p_1'][2, :],
    #           'black', Linestyle='--', label='spacecraft 0 (Krolak)')
    # ax.plot3D(p_local['p_2'][0, :], p_local['p_2'][1, :], p_local['p_2'][2, :],
    #           'black', Linestyle=':', label='spacecraft 1 (Krolak)')
    # ax.plot3D(p_local['p_3'][0, :], p_local['p_3'][1, :], p_local['p_3'][2, :],
    #           'black', Linestyle='-.', label='spacecraft 2 (Krolak)')
    # plt.xlabel('x direction')
    # plt.ylabel('y direction')
    # plt.legend(loc = 'upper right')
    # plt.title('Cartwheeling motion')
    # plt.tight_layout()
    # plt.show()
    #
    # fig6 = plt.figure(6)
    # ax6 = plt.axes(projection='3d')
    # ax6.plot3D(p_val['p_1'][0, :], p['p_1'][1, :], p['p_1'][2, :],
    #            'blue', label='orbit spacecraft 0 (Vallis.)')
    # ax6.plot3D(p_val['p_2'][0, :], p['p_2'][1, :], p['p_2'][2, :],
    #            'red', label='orbit spacecraft 1 (Vallis.)')
    # ax6.plot3D(p_val['p_3'][0, :], p['p_3'][1, :], p['p_3'][2, :],
    #            'green', label='orbit spacecraft 2 (Vallis.)')
    #
    # ax6.plot3D(p['p_1'][0, :], p['p_1'][1, :], p['p_1'][2, :],
    #            'black', Linestyle='--', label='orbit spacecraft 0 (Krolak)')
    # ax6.plot3D(p['p_2'][0, :], p['p_2'][1, :], p['p_2'][2, :],
    #            'black', Linestyle=':', label='orbit spacecraft 1 (Krolak)')
    # ax6.plot3D(p['p_3'][0, :], p['p_3'][1, :], p['p_3'][2, :],
    #            'black', Linestyle='-.', label='orbit spacecraft 2 (Krolak)')
    # plt.xlabel('x direction')
    # plt.ylabel('y direction')
    # plt.legend(loc = 'upper right')
    # plt.title('LISA motion')
    # plt.tight_layout()
    # plt.show()
    #
    #
    #
    # print('Finish')
