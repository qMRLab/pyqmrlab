# coding: utf-8
"""Calculate magnetization transfer saturation (MTsat)

This module is for calculating MTsat from magnetization transfer (MT)-weighted
MRI data.

  Typical usage example:

  from pathlib import Path
  from pyqmrlab.mt import MTsat

  mt_sat_obj = MTsat()
  
  # Download sample dataset
  mt_sat_obj.download(folder='data/')

  # Load data
  data_folder = Path('data/mt_sat/')
  mt_sat_obj.load(
      MTw = data_folder / 'MTw.nii.gz',
      PDw = data_folder / 'PDw.nii.gz',
      T1w = data_folder / 'T1w.nii.gz'
      )

  # Fit data
  mt_sat_obj.fit()

  # Save to NIFTI
  mt_sat_obj.save(filenames = [
      data_folder / 'MTsat.nii', 
      data_folder / 'T1.nii'
      ])
"""

from pyqmrlab.abstract import *

np.seterr(divide="ignore", invalid="ignore")


class MTsat(Abstract):
    """Magnetization Transfer saturation (MTsat) data processing class

    Fits MTsat data and saves to NIfTI. Demo dataset available for download.

    Attributes:
        data_url: Link to demo dataset.
        params: Dictionnary of pulse sequence parameters for each 
                measurements. Measurement keys: MTw, T1w, PDw. Keys for each 
                measurement: flip_angle (degrees) and repetition_time 
                (seconds).
    """

    data_url = "https://osf.io/c5wdb/download?version=3"
    params = None
    T1w = None
    PDw = None
    MTw = None

    def __init__(self, params=None):
        """Initializes a MTsat object.

        Assigns MTsat pulse sequence parameters to that `params` attribute, or
        if none are given assigns the parameters to be used with the demo data.

        Args:
            params: Dictinonnary of pulse sequence parameters for each
                measurements. Measurement keys: MTw, T1w, PDw. Keys for each 
                measurement: flip_angle (degrees) and repetition_time 
                (seconds).

        Returns:
            MTsat class object with parameters initialized.
        """
        if params == None:
            self.params = {
                "MTw": {"flip_angle": 6, "repetition_time": 0.028},
                "T1w": {"flip_angle": 20, "repetition_time": 0.018},
                "PDw": {"flip_angle": 6, "repetition_time": 0.028},
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
        """Calculates MTsat and T1 maps from an MTsat dataset.

        Referenes:
            Helms, G., Dathe, H., Kallenberg, K., Dechent, P., 2008. 
            High-resolution maps of magnetization transfer with inherent 
            correction for RF inhomogeneity and T1 relaxation obtained from 3D
            FLASH MRI. Magn. Reson. Med. 60, 1396-1407.
        """

        FA_MTw = self.params["MTw"]["flip_angle"]
        TR_MTw = self.params["MTw"]["repetition_time"]
        FA_PDw = self.params["PDw"]["flip_angle"]
        TR_PDw = self.params["PDw"]["repetition_time"]
        FA_T1w = self.params["T1w"]["flip_angle"]
        TR_T1w = self.params["T1w"]["repetition_time"]

        # Convert flip_angles from deg to rad
        FA_MTw = np.deg2rad(FA_MTw)
        FA_PDw = np.deg2rad(FA_PDw)
        FA_T1w = np.deg2rad(FA_T1w)

        R1 = self.__R1(self.T1w, self.PDw, FA_T1w, TR_T1w, FA_PDw, TR_PDw)

        A = self.__A(self.T1w, self.PDw, FA_T1w, TR_T1w, FA_PDw, TR_PDw)

        self.MTsat = 100 * (
            TR_MTw * (FA_MTw * (A / self.MTw) - 1) * R1 - (FA_MTw ** 2) / 2
        )
        self.T1 = 1 / R1

        self._apply_mask(MTsat=self.MTsat, T1=self.T1)

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
