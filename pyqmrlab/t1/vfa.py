# coding: utf-8
# coding: utf-8
"""Calculate variable flip angle (VFA) T1 mapping

This module is for calculating T1 from gradient echo data acquired with
different flip angles.

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


class VFA(Abstract):
    """Variable flip angle T1 mapping data processing class

    Fits variable flip angle T1 mapping data and saves to NIfTI. Demo dataset
    available for download.

    Attributes:
        data_url: Link to demo dataset.
        params: Dictionnary of pulse sequence parameters for each 
                measurements. Measurement keys: flip_angle (degrees), 
                repetition_time (seconds).
    """

    data_url = "https://osf.io/7wcvh/download?version=1"

    def __init__(self, params=None):
        """Initializes a VFA object.

        Assigns variable flip angle T1 mapping pulse sequence parameters to the
        `params` attribute, or if none are given assigns the parameters to be
        used with the demo data.

        Args:
        params: Dictionnary of pulse sequence parameters for each 
                measurements. Measurement keys: flip_angle (degrees), 
                repetition_time (seconds).

        Returns:
            VFA class object with parameters initialized.
        """
        if params == None:
            self.params = {"flip_angle": [3, 20], "repetition_time": 0.015}

    def load(self, VFAData, B1map, Mask=None):
        args = locals()
        super().load(args)

    def save(self, filename=None):
        if filename == None:
            filename = "T1.nii.gz"

        super().save(self.T1, filename)

    def fit(self):
        vfa_data = self.VFAData
        B1_map = self.B1map

        # Get and format parameters
        flip_angle = np.deg2rad(self.params["flip_angle"])
        repetition_time = self.params["repetition_time"]

        # Linearize data
        dims = vfa_data.shape
        lin_data = np.reshape(vfa_data, (dims[0] * dims[1] * dims[2], dims[3]))
        lin_B1 = np.reshape(B1_map, (dims[0] * dims[1]))
        lin_flip_angle = np.tile(flip_angle, (dims[0] * dims[1], 1)) * np.transpose(
            np.tile(lin_B1, (dims[3], 1))
        )

        # Voxelwise linear fit
        y = lin_data / np.sin(lin_flip_angle)
        x = lin_data / np.tan(lin_flip_angle)

        xDims = x.shape
        lengthX = xDims[1]

        numerator = np.sum(x * y, 1) - np.sum(x, 1) * np.sum(y, 1) / lengthX
        denominator = np.sum(x ** 2, 1) - np.sum(x, 1) ** 2 / lengthX

        m = numerator / denominator
        b = np.mean(y) - m * np.mean(x)

        # Get T1 and M0 parameters
        lin_T1 = -repetition_time / np.log(m)
        lin_M0 = b / (1 - m)

        # Unlinearize data
        self.T1 = np.reshape(lin_T1, (dims[0], dims[1], dims[2]))
        self.M0 = np.reshape(lin_M0, (dims[0], dims[1], dims[2]))

        # Apply masks and remove NaNs
        self._apply_mask(T1=self.T1)
        self._apply_mask(M0=self.M0)

    @staticmethod
    def simulate(params, type="analytical"):
        """Simulates signal for a variable flip angle T1 mapping experiment.

        Generates longitudinal magnetization for the spoiled gradient echo 
        pulse sequence.

        Args:
        params: Dictionnary of pulse sequence parameters for each 
                measurements, as well as T1 of ths pins. Parameter keys:
                flip_angle (numpy array of angles in degrees), repetition_time
                (seconds), T1 (seconds).
        type: Simulation type. 'analytical': assumes the analytical
              steady-state solution to the pulse sequence experiment.

        Returns:
            Mz: numpy array of the longitudinal magnetization for each flip
                angle measurement.
        """
        if type is "analytical":
            try:
                repetition_time = params["repetition_time"]  # seconds
                flip_angle = np.array(np.deg2rad(params["flip_angle"]))

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
                    * (
                        (1 - np.exp(-repetition_time / T1))
                        / (1 - np.cos(flip_angle) * np.exp(-repetition_time / T1))
                    )
                    * np.sin(flip_angle)
                )
                return Mz
            except:
                print("Incorrect parameters.")
