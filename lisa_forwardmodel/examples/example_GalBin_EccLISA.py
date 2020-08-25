"""This examples shows,

    1. how to simulate a LISA object
    2. how to simulate a GW-source object
    3. how to simulate the basic interspacecraft Doppler measurements
    4. how to derive TDI observables from the basic Doppler measurements.

In all steps, no noise is considered."""

# import all necessary packages
import os
import sys

import h5py
from datetime import datetime
import argparse
import matplotlib.pyplot as plt
import numpy as np

import lisa_forwardmodel
from lisa_forwardmodel.objects.Simulation import Simulation
from lisa_forwardmodel.objects.LISA import EccentricLISA, CircularLISA
from lisa_forwardmodel.objects.TDI import InterSCDopplerObservable, TDI
from lisa_forwardmodel.objects.Waveforms import WaveformSimpleBinaries
from lisa_forwardmodel.utils.readParameters import read_parameter_fun


### reading in user input
parser = argparse.ArgumentParser()

parser.add_argument("-o", "--orbit",
                    default="eccentric",
                    type=str,
                    choices=["circular", "eccentric"],
                    help="Argument sets orbits of LISA SC around Sun. "
                         "Choices: circular or eccentric. "
                         "Default: eccentric")

parser.add_argument("-s", "--source",
                    default="GalBin",
                    type=str,
                    choices=["GalBin"],
                    help="Defines the source of the gravitational Waves. "
                         "Choices: GalBin (to be extended). Default: GalBin")

parser.add_argument("-t0", "--t0",
                    type=float,
                    default=None,
                    help="Defines the starting time of the simulation in seconds. "
                         "Default: taken from default Simulation.txt")

parser.add_argument("-tend", "--tend",
                    type=float,
                    default=None,
                    help="Defines the ending time of the simulation in seconds. "
                         "Default: taken from default Simulation.txt")

parser.add_argument("-tres", "--tres",
                    type=float,
                    default=None,
                    help="Defines the temporal resolution of the simulation in seconds. "
                         "Default: taken from default Simulation.txt")

parser.add_argument("-D", "--Doppler",
                    default=None,
                    type=str,
                    nargs="+",
                    choices=["y312", "y123", "y231", "y2-13", "y3-21", "y1-32"],
                    help="Defines the basic interspacecraft Doppler measurements that should be simulated. "
                         "Choices: ['y312', 'y123', 'y231', 'y2-13', 'y3-21', 'y1-32']"
                         "Default: None")

parser.add_argument("-TDI", "--TDI",
                    default="X1",
                    type=str,
                    nargs="+",
                    choices=["X", "X1", "X2", "X3", "AET", "A", "E", "T"],
                    help="Defines the TDI observables that should be simulated. "
                         "Choices: ['X', 'X1', 'X2', 'X3', 'AET', 'A', 'E', 'T']"
                         "Default: X1")

parser.add_argument("-v", "--verbose",
                    default=False,
                    action='store_true',
                    help="Defines whether additional information is shown or not. Default: False")

parser.add_argument("-path", "--save_path",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results'),
                    type=str,
                    help="Defines the directory path where to store the simulation results. "
                         "Default: /results")

args = vars(parser.parse_args())

### preparing counter for potential plotting
plt.close('all')
if args['verbose']:
    fig_count = 1

### preparing file names for later saving
date = datetime.today().strftime('%Y%m%d')
if args['Doppler'] is not None and args['TDI'] is not None:
    save_name = date + '_' + args['orbit'][:3] + '_' + args['source'] + '_' + 'TDI' + '_' + 'Doppler' + '.hdf5'
elif args['Doppler'] is not None and args['TDI'] is None:
    save_name = date + '_' + args['orbit'][:3] + '_' + args['source'] + '_' + 'Doppler' + '.hdf5'
elif args['Doppler'] is None and args['TDI'] is not None:
    save_name = date + '_' + args['orbit'][:3] + '_' + args['source'] + '_' + 'Doppler' + '.hdf5'
else:
    save_name = date + '_' + args['orbit'][:3] + '_' + args['source'] + '.hdf5'

if not os.path.isdir(args['save_path']):
    os.mkdir(args['save_path'])

### defining simulation object
sim = Simulation(t_min=args['t0'],
                 t_max=args['tend'],
                 t_res=args['tres'])
dim_t = len(sim.t)

### defining spaceinterferometer object
LISA_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                         'input_data', 'LISA_parameters.txt')
LISA_parameters = read_parameter_fun(LISA_file)

if args['orbit'] == 'eccentric':
    LISA = EccentricLISA(parameters = LISA_parameters, sim = sim)
elif args['orbit'] == 'circular':
    LISA = CircularLISA(parameters=LISA_parameters, sim = sim)

p = LISA.get_p()
### defining wave object
if args['source'] == "GalBin":
    wave_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'input_data',
                             'GalBin_parameters.txt')
    wave_parameters = read_parameter_fun(wave_file)
    wave = WaveformSimpleBinaries(parameters=wave_parameters, sim = sim)
    wave_sim = wave.h_plus_cross()
    if args['verbose']:
        t_delta = 50
        t_median = int(np.round(len(sim.t) / 2))

        plt.figure(fig_count)

        plt.subplot(211)
        ttl = plt.title('Plus polarization components (and zoomed in version) of ' + args['source'])
        ttl.set_position([.5, 1.1])
        plt.plot(sim.t, wave_sim['h_plus'], 'red', Linewidth = 1)
        plt.ylabel('h_plus (-)')
        plt.grid()

        plt.subplot(212)
        plt.plot(sim.t[(t_median-t_delta):(t_median+t_delta)],
                 wave_sim['h_plus'][t_median-t_delta:t_median+t_delta],
                 'red', Linewidth = 1)
        plt.ylabel('h_plus (-)')
        plt.xlabel('time (sec)')
        plt.grid()

        plt.subplots_adjust(wspace=0.75)
        plt.show()
        fig_count += 1

        plt.figure(fig_count)

        plt.subplot(211)
        ttl = plt.title('Cross polarization components (and zoomed in version) of ' + args['source'])
        ttl.set_position([.5, 1.1])
        plt.plot(sim.t, wave_sim['h_cross'], 'blue', Linewidth=1)
        plt.ylabel('h_cross (-)')
        plt.grid()

        plt.subplot(212)
        plt.plot(sim.t[t_median - t_delta:t_median + t_delta],
                 wave_sim['h_cross'][t_median - t_delta:t_median + t_delta],
                 'blue', Linewidth=1)
        plt.ylabel('h_cross (-)')
        plt.xlabel('time (sec)')
        plt.grid()

        plt.subplots_adjust(wspace=0.75)
        plt.show()
        fig_count += 1

y_sim = dict()
if args['Doppler'] is not None:
    Doppler_obj = InterSCDopplerObservable(detector=LISA, wave=wave)
    for y in args['Doppler']:
        # y_sim[y] = np.zeros((dim_t,))
        s = y[1]
        r = y[-1]
        y_sim[y] = Doppler_obj.y(s=eval(s), r=eval(r))
        if args['verbose']:
            plt.figure(fig_count)
            plt.title('Basic Doppler Measurement ' + y)
            plt.plot(sim.t, y_sim[y], 'red', Linewidth=1)
            plt.ylabel(y)
            plt.xlabel('time (sec)')
            plt.grid()
            plt.show()
            fig_count += 1

TDI_sim = dict()
if args['TDI'] is not None:
    TDI_obj = TDI(detector=LISA, wave=wave)
    if isinstance(args['TDI'], str):
        TDI_vars = [args['TDI']]
    else:
        TDI_vars = args['TDI']

    for obs in TDI_vars:
        if obs == 'X' or obs == 'X1' or obs == 'X2' or obs == 'X3':
            TDI_sim['X'] = dict()

        if obs == 'AET' or obs == 'A' or obs == 'E' or obs == 'T':
            TDI_sim['AET'] = dict()
            AET = TDI_obj.AET()

        if obs == 'X':
            TDI_sim[obs]['X1'] = TDI_obj.X1()
            TDI_sim[obs]['X2'] = TDI_obj.X2()
            TDI_sim[obs]['X3'] = TDI_obj.X3()
        else:
            if obs=='X1':
                TDI_sim['X'][obs] = TDI_obj.X1()
            if obs=='X2':
                TDI_sim['X'][obs] = TDI_obj.X2()
            if obs=='X3':
                TDI_sim['X'][obs] = TDI_obj.X3()

        if obs == 'AET':
            TDI_sim[obs]['A'] = AET['A']
            TDI_sim[obs]['E'] = AET['E']
            TDI_sim[obs]['T'] = AET['T']
        else:
            if obs=='A':
                TDI_sim['AET'][obs] = AET['A']
            if obs == 'E':
                TDI_sim['AET'][obs] = AET['E']
            if obs == 'T':
                TDI_sim['AET'][obs] = AET['T']

if args['verbose']:
    color = ['red', 'blue', 'green']
    try:
        no_AET = len(TDI_sim['AET'].keys())
        plt.figure(fig_count)
        plt.title('TDI observables AET')
        for count, cur in enumerate(TDI_sim['AET']):
            plt.subplot(no_AET, 1, count+1)
            plt.plot(sim.t, TDI_sim['AET'][cur], color[count], Linewidth = 1)
            plt.ylabel(cur)
            plt.grid()

        plt.xlabel('time (sec)')
        plt.show()
        fig_count += 1
    except KeyError:
        pass

    try:
        no_X = len(TDI_sim['X'].keys())
        plt.figure(fig_count)
        plt.title('TDI observables X')
        for count, cur in enumerate(TDI_sim['X']):
            plt.subplot(no_X, 1, count+1)
            plt.plot(sim.t, TDI_sim['X'][cur], color[count], Linewidth=1)
            plt.ylabel(cur)
            plt.grid()

        plt.xlabel('time (sec)')
        plt.show()
        fig_count += 1
    except KeyError:
        pass

### saving into hdf5 format
with h5py.File(os.path.join(args['save_path'],save_name,), 'w') as f:
    sim_grp = f.create_group("temporal_sim")
    data_t = sim_grp.create_dataset("t", np.shape(sim.t), data = sim.t)
    data_t_res = sim_grp.create_dataset("t_res", np.shape(sim.t_res), data = sim.t_res, dtype = float)
    data_t_min = sim_grp.create_dataset("t_min", np.shape(sim.t_min), data = sim.t_min, dtype = float)
    data_t_max = sim_grp.create_dataset("t_max", np.shape(sim.t_max), data = sim.t_max, dtype = float)

    detector_grp = f.create_group("detector")
    detector_parameters_grp = detector_grp.create_group("parameters")
    for no, k in enumerate(list(LISA.parameters.keys())):
        value = LISA.parameters[k]
        detector_data_temp = detector_parameters_grp.create_dataset(k, shape = np.shape(value), data = value)
    detector_orbits_grq = detector_grp.create_group("orbits")
    for no, k in enumerate(list(p.keys())):
        value = p[k]
        detector_data_temp = detector_orbits_grq.create_dataset(k, shape = np.shape(value), data = value)

    wave_grp = f.create_group("wave")
    wave_parameters_grp = wave_grp.create_group("parameters")
    for no, k in enumerate(list(wave.parameters.keys())):
        value = wave.parameters[k]
        wave_data_temp = wave_parameters_grp.create_dataset(k, shape = np.shape(value), data = value)
    wave_h_grp = wave_grp.create_group("h_components")
    for no, k in enumerate(list(wave_sim.keys())):
        value = wave_sim[k]
        wave_data_temp = wave_h_grp.create_dataset(k, shape = np.shape(value), data = value)

    Doppler_grp = f.create_group("Doppler")
    for no, k in enumerate(list(y_sim.keys())):
        value = y_sim[k]
        Doppler_data_temp = Doppler_grp.create_dataset(k, shape=np.shape(value), data=value)

    TDI_grp = f.create_group("TDI")
    for l in list(TDI_sim.keys()):
        TDI_subgrp = TDI_grp.create_group(l)
        for k in list(TDI_sim[l].keys()):
            value = TDI_sim[l][k]
            TDI_data_temp = TDI_subgrp.create_dataset(k, shape=np.shape(value), data=value)

print('Finish')
# args = parser.parse_args()
# print(args)
