import sys
import os
from typing import Any, Union
from lisa_forwardmodel.utils.readParameters import read_parameter_fun
from lisa_forwardmodel.utils.checkInput import checkInput

import numpy as np

def NoneAssertion(value, err):
    assert value is not None, err

class Simulation():
    def __init__(self,
                 constant = None,
                 simulation_parameters=None,
                 t_min=None,
                 t_max=None,
                 t_res=None,
                 t=None):

        self.constant = constant
        self.simulation_parameters = simulation_parameters
        self.t_min = t_min
        self.t_max = t_max
        self.t_res = t_res
        self.t = t

        NoneAssertion(self.constant,
                      "Ensure that simulation object has attribute 'constant'.")
        NoneAssertion(self.simulation_parameters,
                      "Ensure that simulation object has attribute 'simulation_parameters'.")
        NoneAssertion(self.t_min,
                      "Ensure that simulation object has attribute 't_min'.")
        NoneAssertion(self.t_max,
                      "Ensure that simulation object has attribute 't_max'.")
        NoneAssertion(self.t,
                      "Ensure that simulation object has attribute 't'.")

    @property
    def constant(self):
        return self.__constant

    @constant.setter
    def constant(self, value):
        if value is not None and isinstance(value, dict):
            self.__constant = value
        else:
            constant_parameter_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                                       'input_data', 'Constants.txt')
            print("Constant parameters are taken from ", constant_parameter_filepath)
            self.__constant = read_parameter_fun(constant_parameter_filepath)

    @property
    def simulation_parameters(self):
        return self.__simulation_parameters

    @simulation_parameters.setter
    def simulation_parameters(self, value):
        if value is not None:
            self.__simulation_parameters = value
        else:
            simulation_parameter_filepath = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                                                         'input_data', 'Simulation_parameters.txt')
            print("Simulation parameters are taken from ", simulation_parameter_filepath)
            self.__simulation_parameters = read_parameter_fun(simulation_parameter_filepath)

    @property
    def t_min(self):
        return self.__t_min

    @t_min.setter
    def t_min(self, value):
        if value is not None:
            self.__t_min = value
        else:
            time_units = self.simulation_parameters['unit']
            if time_units.find('sec') > -1 or time_units == 's':
                self.__t_min = self.simulation_parameters['t_min']
            else:
                self.__t_min = self.simulation_parameters['t_min'] * self.constant['secs/year']

    @property
    def t_max(self):
        return self.__t_max

    @t_max.setter
    def t_max(self, value):
        if value is not None:
            self.__t_max = value
        else:
            time_units = self.simulation_parameters['unit']
            if time_units.find('sec') > -1 or time_units == 's':
                self.__t_max = self.simulation_parameters['t_max']
            else:
                self.__t_max = self.simulation_parameters['t_max'] * self.constant['secs/year']

    @property
    def t_res(self):
        return self.__t_res

    @t_res.setter
    def t_res(self, value):
        ind, temp = checkInput(value, int, float)
        if ind:
            self.__t_res = value
        else:
            self.__t_res = round(self.simulation_parameters['t_res'])

    @property
    def t(self):
        return self.__t

    @t.setter
    def t(self, value):
        if value is not None:
            if isinstance(value, np.ndarray):
                self.__t = value
            else:
                self.__t = np.array(value)
        else:
            if (self.t_res, self.t_max, self.t_min) is not None:
                __t = []
                t_temp = self.__t_min
                __t.append(t_temp)
                while t_temp < self.__t_max:
                    __t.append(t_temp + self.__t_res)
                    t_temp = __t[-1]
                if __t[-1] > self.__t_max:
                    __t[-1] = self.__t_max
                elif __t[-1] < self.__t_max:
                    __t.append(self.__t_max)
                self.__t = np.array(__t)

    def __eq__(self, other):
        if not isinstance(other, Simulation):
            return NotImplemented

        return (self.t_min == other.t_min
                and self.t_max == other.t_max
                and self.t_res == other.t_res
                and np.array_equal(self.t, other.t))

if __name__ == '__main__':
    sim = Simulation(t_res = 150)
    constants = sim.constant
    t = sim.t
    t_res = sim.t_res
    assert t[0] == sim.t_min, "Contradiction in initial time."
    assert t[-1] == sim.t_max, "Contradiction in final time."