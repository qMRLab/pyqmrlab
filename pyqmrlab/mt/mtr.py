# coding: utf-8
"""Calculate magnetization transfer ratio (MTR)

This module is for calculating MTR from magnetization transfer (MT)-weighted
MRI data.

  Typical usage example:

  from pathlib import Path
  from pyqmrlab.mt import MTR

  mtr_obj = MTR()
  
  # Download sample dataset
  mtr_obj.download(folder='data/')

  # Load data
  data_folder = Path('data/mt_ratio/')
  mtr_obj.load(
      MTon = data_folder / 'MTon.mat',
      MToff = data_folder / 'MToff.mat',
      Mask = data_folder / 'Mask.mat'
      )

  # Fit data
  mtr_obj.fit()

  # Save to NIFTI
  mtr_obj.save(filename = data_folder / 'MTR.nii')
"""

from pyqmrlab.abstract import *

np.seterr(divide="ignore", invalid="ignore")


class MTR(Abstract):
    """Magnetization Transfer Ratio (MTR) data processing class

    Fits MTR data and saves to NIfTI. Demo dataset available for download.

    Attributes:
        data_url: Link to demo dataset.
    """

    data_url = "https://osf.io/erm2s/download?version=1"

    def __init__(self):
        pass

    def load(self, MTon, MToff, Mask=None):
        args = locals()
        super().load(args)

    def save(self, filename=None):
        if filename == None:
            filename = "MTR.nii.gz"

        super().save(self.mtr, filename)

    def fit(self):
        self.mtr = (self.MToff - self.MTon) / self.MToff * 100

        self._apply_mask(mtr=self.mtr)
