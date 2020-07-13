from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='lisa_forwardmodel',
    version='0.1',
    url='',
    license='',
    author='Franziska Riegger',
    author_email='Franziska.Riegger@erdw.ethz.ch',
    description='lisa_forwardmodel simulates the LISA response without noise.',
    long_description=long_description,
    classifiers=[
            "Programming Language :: Python :: 3",
            "Operating System :: OS Independent",
        ],
    python_requires='>=3.8'
)
