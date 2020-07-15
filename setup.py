from setuptools import setup, find_packages

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
    install_requires=list_packages,
    python_requires='>=3.6',
    setup_requires=['pytest-runner'],
    tests_require=['pytest']
)
