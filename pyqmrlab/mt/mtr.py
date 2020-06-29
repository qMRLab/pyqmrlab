# coding: utf-8

# Scientific modules imports
import numpy as np
import scipy.io as sio
import pyqmrlab.utils as utils
from pathlib import Path
import nibabel as nib
from pyqmrlab.abstract import Abstract

np.seterr(divide="ignore", invalid="ignore")


class mtr(Abstract):
    data_url = "https://osf.io/erm2s/download?version=1"

    def __init__(self):
        pass

    def load(self, MTon, MToff, Mask=None):
        args = locals()
        super().load(args)

    def save(self, filename=None):

        if filename == None:
            filename = "MTR.nii.gz"

        super().save(self.MTR, filename)

    def fit(self):
        self.MTR = (self.MToff - self.MTon) / self.MToff * 100

        self.apply_mask(MTR=self.MTR)
