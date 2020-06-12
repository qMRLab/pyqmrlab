from setuptools import setup, find_packages
from os import path

import pyqmrlab

# Get the directory where this current file is saved
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

req_path = path.join(here, 'requirements.txt')
with open(req_path, "r") as f:
    install_reqs = f.read().strip()
    install_reqs = install_reqs.split("\n")

setup(
    name='pyqmrlab',
    python_requires='>=3.7',
    version=pyqmrlab.__version__,
    description='Python package for Quantitative MRI',
    long_description=long_description,
    url='https://github.com/qMRLab/pyqmrlab',
    author='NeuroPoly Lab, Polytechnique Montreal',
    author_email='neuropoly@googlegroups.com',
    license='MIT',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
    keywords='',
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),
    install_requires=install_reqs,
    package_dir={'pyqmrlab': 'pyqmrlab'},
)
