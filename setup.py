from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", newline = '\n') as gh:
    list_packages = gh.read().splitlines()

setup(
    name='lisa_forwardmodel',
    version='0.1',
    license='',
    author='Franziska Riegger',
    author_email='Franziska.Riegger@erdw.ethz.ch',
    description='lisa_forwardmodel simulates the LISA response without noise.',
    long_description=long_description,
    packages = ['lisa_forwardmodel',
                'lisa_forwardmodel.objects',
                'lisa_forwardmodel.utils',
                'lisa_forwardmodel.input_data',
                'lisa_forwardmodel.examples',
                'lisa_forwardmodel.test'],
    package_data={'lisa_forwardmodel': ['input_data/*.txt']},
    install_requires=list_packages,
    python_requires='>=3.6',
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)