# coding: utf-8

# Scientific modules imports
import numpy as np
import pyqmrlab.utils as utils
from pyqmrlab.abstract import Abstract

np.seterr(divide="ignore", invalid="ignore")


class vfa(Abstract):
    data_url = "https://osf.io/7wcvh/download?version=1"

    def __init__(self, params=None):
        if params == None:
            self.params = {"FA": [3, 20], "TR": 0.015}

    def load(self, VFAData, B1map, Mask=None):
        args = locals()
        super().load(args)

    def save(self, filename=None):

        if filename == None:
            filename = "T1.nii.gz"

        super().save(self.T1, filename)

    def fit(self):
        VFAData = self.VFAData
        B1map = self.B1map
        Mask = self.Mask

        # Get and format parameters
        FA = np.deg2rad(self.params["FA"])
        TR = self.params["TR"]

        # Linearize data
        dims = VFAData.shape
        linData = np.reshape(VFAData, (dims[0] * dims[1] * dims[2], dims[3]))
        linB1 = np.reshape(B1map, (dims[0] * dims[1]))
        linFA = np.tile(FA, (dims[0] * dims[1], 1)) * np.transpose(
            np.tile(linB1, (dims[3], 1))
        )

        # Voxelwise linear fit
        y = linData / np.sin(linFA)
        x = linData / np.tan(linFA)

        xDims = x.shape
        lengthX = xDims[1]

        numerator = np.sum(x * y, 1) - np.sum(x, 1) * np.sum(y, 1) / lengthX
        denominator = np.sum(x ** 2, 1) - np.sum(x, 1) ** 2 / lengthX

        m = numerator / denominator
        b = np.mean(y) - m * np.mean(x)

        # Get T1 and M0 parameters
        linT1 = -TR / np.log(m)
        linM0 = b / (1 - m)

        # Unlinearize data
        self.T1 = np.reshape(linT1, (dims[0], dims[1], dims[2]))
        self.M0 = np.reshape(linM0, (dims[0], dims[1], dims[2]))

        # Apply masks and remove NaNs
        self.apply_mask(T1=self.T1)
        self.apply_mask(M0=self.M0)

    @staticmethod
    def simulate(params, type="analytical"):
        try:
            TR = params["TR"]  # seconds
            FA = np.array(np.deg2rad(params["FA"]))

            if "constant" in params:
                constant = params["constant"]
            else:
                constant = 1  # unitless

            if "T1" in params:
                T1 = params["T1"]
            else:
                T1 = 0.850  # ms
            Mz = (
                constant
                * ((1 - np.exp(-TR / T1)) / (1 - np.cos(FA) * np.exp(-TR / T1)))
                * np.sin(FA)
            )
            return Mz
        except:
            print("Incorrect parameters.")
