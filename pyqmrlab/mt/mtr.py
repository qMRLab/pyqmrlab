# coding: utf-8

# Scientific modules imports
import numpy as np
import scipy.io as sio
import pyqmrlab.utils as utils
from pathlib import Path
import nibabel as nib

np.seterr(divide="ignore", invalid="ignore")


class mtr:
    def __init__(self):
        self.data_url = "https://osf.io/erm2s/download?version=1"
        self.inputs = ["MTon", "MToff", "Mask"]

        self.outputs = ["MTR"]

    def download(self, folder=None):
        if folder == None:
            utils.download_data(self.data_url)
        else:
            utils.download_data(self.data_url, folder)

    def load(self, MTon, MToff, Mask=None):
        args = locals()
        for key, value in args.items():
            if key != "self" and value != None:
                filepath = Path(value)
                if ".mat" in filepath.suffixes:
                    matDict = sio.loadmat(filepath)
                    setattr(self, key, matDict[key])

    def save(self, filename=None):

        if filename == None:
            filename = "MTR.nii.gz"

        img = nib.Nifti1Image(self.MTR, affine=None, header=None)
        nib.save(img, filename)

    def fit(self):
        self.MTR = (self.MToff - self.MTon) / self.MToff * 100

        if hasattr(self, "Mask"):
            self.Mask[np.isnan(self.Mask)] = 0

            self.Mask = self.Mask.astype(bool)
            self.MTR = self.MTR * self.Mask

        self.MTR[np.isnan(self.MTR)] = 0
