# coding: utf-8

# Scientific modules imports
import numpy as np
import scipy.io as sio
import pyqmrlab.utils as utils
from pathlib import Path
import nibabel as nib
from pyqmrlab.abstract import Abstract

np.seterr(divide="ignore", invalid="ignore")


class mtsat(Abstract):
    data_url = "https://osf.io/c5wdb/download?version=3"

    def __init__(self, params=None):
        pass

        if params == None:
            self.params = params = {
                "MTw": {"FA": 6, "TR": 0.028},
                "T1w": {"FA": 20, "TR": 0.018},
                "PDw": {"FA": 6, "TR": 0.028},
            }

    def load(self, MTw, PDw, T1w, Mask=None):
        args = locals()
        super().load(args)

    def save(self, filenames=None):

        if filenames == None:
            filename = ["MTsat.nii.gz", "T1.nii.gz"]
        
        super().save(self.MTsat, filenames[0])
        super().save(self.T1, filenames[1])

    def fit(self):

        FA_MTw = self.params["MTw"]["FA"]
        TR_MTw = self.params["MTw"]["TR"]
        FA_PDw = self.params["PDw"]["FA"]
        TR_PDw = self.params["PDw"]["TR"]
        FA_T1w = self.params["T1w"]["FA"]
        TR_T1w = self.params["T1w"]["TR"]

        # Convert FA's from deg to rad
        FA_MTw = np.deg2rad(FA_MTw)
        FA_PDw = np.deg2rad(FA_PDw)
        FA_T1w = np.deg2rad(FA_T1w)

        R1 = self.__R1(self.T1w, self.PDw, FA_T1w, TR_T1w, FA_PDw, TR_PDw)

        A = self.__A(self.T1w, self.PDw, FA_T1w, TR_T1w, FA_PDw, TR_PDw)

        self.MTsat = 100 * (
            TR_MTw * (FA_MTw * (A / self.MTw) - 1) * R1 - (FA_MTw ** 2) / 2
        )
        self.T1 = 1 / R1

        if hasattr(self, "Mask"):
            self.Mask[np.isnan(self.Mask)] = 0

            self.Mask = self.Mask.astype(bool)
            self.MTsat = self.MTsat * self.Mask
            self.T1 = self.T1 * self.Mask

        self.MTsat[np.isnan(self.MTsat)] = 0
        self.T1[np.isnan(self.T1)] = 0

    def __R1(self, T1w, PDw, FA_T1w, TR_T1w, FA_PDw, TR_PDw):

        num_1 = (FA_T1w / TR_T1w) * T1w
        num_2 = (FA_PDw / TR_PDw) * PDw
        denum_1 = PDw / FA_PDw
        denum_2 = T1w / FA_T1w
        return 0.5 * (num_1 - num_2) / (denum_1 - denum_2)

    def __A(self, T1w, PDw, FA_T1w, TR_T1w, FA_PDw, TR_PDw):

        num_1 = TR_PDw * FA_T1w / FA_PDw
        num_2 = TR_T1w * FA_PDw / FA_T1w
        num_3 = PDw * T1w
        denum_1 = TR_PDw * FA_T1w * T1w
        denum_2 = TR_T1w * FA_PDw * PDw
        return (num_1 - num_2) * (num_3 / (denum_1 - denum_2))
