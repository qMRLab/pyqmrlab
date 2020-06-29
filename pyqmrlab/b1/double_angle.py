# coding: utf-8

# Scientific modules imports
import numpy as np
import pyqmrlab.utils as utils
from pyqmrlab.abstract import Abstract

np.seterr(divide="ignore", invalid="ignore")


class double_angle(Abstract):
    data_url = "https://osf.io/kytxw/download?version=1"

    def __init__(self, params=None):
        if params == None:
            self.params = {"FA": 60}

    def load(self, img1, img2, Mask=None):
        args = locals()
        super().load(args)

    def save(self, filename=None):

        if filename == None:
            filename = "B1.nii.gz"

        super().save(self.B1, filename)

    def fit(self):
        img1 = self.img1
        img2 = self.img2
        FA = np.deg2rad(self.params["FA"])

        self.B1 = np.abs(np.arccos(img2 / (2 * img1)) / (FA))

        self.apply_mask(B1=self.B1)
