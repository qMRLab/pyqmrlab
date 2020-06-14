# coding: utf-8

# Scientific modules imports
from abc import ABCMeta, abstractmethod
from pathlib import Path
import nibabel as nib
import scipy.io as sio

import pyqmrlab.utils as utils

class Abstract(object, metaclass=ABCMeta):

    @property
    @classmethod
    @abstractmethod
    def data_url(cls):
        """OSF.io url to default data."""
        return NotImplementedError

    @abstractmethod
    def load(self, args):
        """Method that loads data."""
        for key, value in args.items():
            if key != "self" and key != "__class__" and value != None:
                filepath = Path(value)
                if ".mat" in filepath.suffixes:
                    matDict = sio.loadmat(filepath)
                    setattr(self, key, matDict[key])
                elif ".nii" in filepath.suffixes:
                    img = nib.load(filepath)
                    setattr(self, key, img.get_fdata())

    @abstractmethod
    def save(self, img, filename):
        """Method that saves data."""
        niiimg = nib.Nifti1Image(img, affine=None, header=None)
        nib.save(niiimg, filename)

    @abstractmethod
    def fit(self):
        """Method that fits data."""

    def download(self, folder=None):
        if folder == None:
            utils.download_data(self.data_url)
        else:
            utils.download_data(self.data_url, folder)
