lisa_forwardmodel
=======

### What is it?

lisa_forwardmodel is a Python package to simulate the LISA. 
These simulations include:

1.  the spacecrafts' orbits
2.  the response of LISA to a passing gravitational wave
3.  the interspacecraft Doppler measurements
4.  multiple TDI observables .

Note, that no noise is considered.

### Requirements and Installation

*The Pip Way*

0. Install a running version of Python (3.6)
1. Install ````pip```` (see [here](https://pip.pypa.io/en/stable/installing/))
2. Install ````setuptools```` (see [here](https://pypi.org/project/setuptools/))
3. Run setup script: ````python setup.py install````
4. Get started!

It is recommended to use a virtual environment (see [here](https://docs.python.org/3.6/tutorial/venv.html)).

*The Conda Way*
(recommended option)

0. Install conda (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/install/))
1. Run setup script: ````python setup.py install````
2. Get started!

Again, here it is recommended to create a virtual environment (see [here](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)).

### Usage

Generally, the code implements objects for the detector, the wave and the measurements. 
A simple application of these is given with the example code ``examples/example_GalBin_EccLISA.py``.
It intends to simulate the LISA orbits, a gravitational wave source, LISA's response to these waves as well as interspacecraft Doppler measurments and time-delayed linear combinations of these, the so-called TDI observables.
All input parameters and their default settings can be inspected by

````python lisa_forwardmodel/example/example_GalBin_EccLISA.py --help````.

An ```.hdf5``` file with the generated simulation data is automatically stored in ``\results``.

A more detailed description of the available objects will be given in ```docs```. 

#### Testing

To test any modifications of the code, run

````py.test````

in the main directory of your project.






