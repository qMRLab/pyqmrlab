# coding: utf-8
# pylint: disable=E1101

# Scientific modules imports
from abc import ABCMeta, abstractmethod
from pathlib import Path
import nibabel as nib
import scipy.io as sio
import numpy as np

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
                self._load_data(filepath, key)

    def _load_data(self, filepath, key=None):
        if ".mat" in filepath.suffixes:
            matDict = sio.loadmat(filepath)
            if key is not None:
                setattr(self, key, matDict[key])
            else:
                return matDict
        elif ".nii" in filepath.suffixes:
            img = nib.load(filepath)
            if key is not None:
                setattr(self, key, img.get_fdata())
            else:
                return img.get_fdata()

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

    def _apply_mask(self, **kwargs):
        if hasattr(self, "Mask"):
            Mask = self.Mask

            Mask[np.isnan(Mask)] = 0

            Mask = Mask.astype(bool)
            for key, value in kwargs.items():

                if len(Mask.shape) == 2 and len(value.shape) == 3:
                    Mask = np.expand_dims(Mask, 2)

                value = value * Mask
                value[np.isnan(value)] = 0
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                value[np.isnan(value)] = 0
                setattr(self, key, value)
