# coding: utf-8
"""Calculate magnetization transfer ratio (MTR)

This module is for calculating MTR from magnetization transfer (MT)-weighted
MRI data.

  Typical usage example:

  from pathlib import Path
  from pyqmrlab.b1 import DoubleAngle

  b1_obj = DoubleAngle()
  
  # Download sample dataset
  b1_obj.download(folder='data/')

  # Load data
  data_folder = Path('data/b1_dam_multi-slice/')
  b1_obj.load(
      img1 = data_folder / 'epseg_60deg.nii.gz',
      img2 = data_folder / 'epseg_120deg.nii.gz',
      )

  # Fit data
  b1_obj.fit()

  # Save to NIFTI
  b1_obj.save(filename = data_folder / 'B1.nii')
"""

from pyqmrlab.abstract import *

np.seterr(divide="ignore", invalid="ignore")


class DoubleAngle(Abstract):
    """Double angle B1 mapping data processing class

    Fits double angle B1 mapping data and saves to NIfTI. Demo dataset
    available for download.

    Attributes:
        data_url: Link to demo dataset.
        params: Dictionnary of pulse sequence parameters for each 
                measurements. Measurement keys: flip_angle (degrees).
    """

    data_url = "https://osf.io/kytxw/download?version=1"
    params = None
    img1 = None
    img2 = None

    def __init__(self, params=None):
        """Initializes a DoubleAngle object.

        Assigns double angle B1 mapping pulse sequence parameters to the
        `params` attribute, or if none are given assigns the parameters to be
        used with the demo data.

        Args:
        params: Dictionnary of pulse sequence parameters for each 
                measurements. Measurement keys: flip_angle (degrees).

        Returns:
            DoubleAngle class object with parameters initialized.
        """
        if params == None:
            self.params = {"flip_angle": 60}

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
        flip_angle = np.deg2rad(self.params["flip_angle"])

        self.B1 = np.abs(np.arccos(img2 / (2 * img1)) / (flip_angle))

        self._apply_mask(B1=self.B1)
