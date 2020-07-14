"""This examples shows,

    1. how to simulate a LISA object
    2. how to simulate a GW-source object
    3. how to simulate the basic interspacecraft Doppler measurements
    4. how to derive TDI observables from the basic Doppler measurements.

In all steps, no noise is considered."""

# import all necessary packages
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
import argparse

from lisa_forwardmodel.objects.Simulation import Simulation
from lisa_forwardmodel.objects.LISA import EccentricLISA, CircularLISA
from lisa_forwardmodel.objects.TDI import InterSCDopplerObservable, TDI
from lisa_forwardmodel.objects.Waveforms import WaveformSimpleBinaries

parser = argparse.ArgumentParser()
parser.parse_args()

parser.add_argument('-o','--orbit',
                    default = 'eccentric',
                    type = str,
                    choices = ['circular', 'eccentric'],
                    help = 'Argument sets orbits of LISA SC around Sun. '
                           'Choices: circular or eccentric. '
                           'Default: eccentric')

parser.add_argument('-s','--source',
                    default = 'GalBin',
                    type = str,
                    choices = ['GalBin'],
                    help = 'Defines the source of the gravitational Waves. '
                           'Choices: GalBin (to be extended). '
                           'Default: GalBin')