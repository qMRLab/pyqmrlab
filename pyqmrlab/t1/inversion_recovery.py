# coding: utf-8

# Scientific modules imports

from collections import namedtuple
import numpy as np
import scipy.io as sio
import nibabel as nib
from pathlib import Path
import pyqmrlab.utils as utils
from pyqmrlab.abstract import Abstract

np.seterr(divide="ignore", invalid="ignore")


class InversionRecovery(Abstract):
    IRData = None
    Mask = None
    data_url = "https://osf.io/cmg9z/download?version=3"

    T1 = None
    a = None
    b = None
    residual = None
    idx = None

    def __init__(self, params=None):
        if params == None:
            self.params = {
                "inversion_times": [
                    0.350,
                    0.500,
                    0.650,
                    0.800,
                    0.950,
                    1.100,
                    1.250,
                    1.400,
                    1.700,
                ],
                "repetition_time": None,
            }
        else:
            if "repetition_time" not in params:
                params["repetition_time"] = None
            self.params = {
                "inversion_times": params["inversion_times"],
                "repetition_time": params["repetition_time"],
            }

    def load(
        self,
        magnitude=None,
        phase=None,
        real=None,
        imaginary=None,
        complex=None,
        Mask=None,
    ):

        if magnitude is not None and phase is not None:
            magnitude = Path(magnitude)
            mag_data = self._load_data(magnitude)

            phase = Path(phase)
            raw_phase_data = self._load_data(phase)

            phase_data = (
                raw_phase_data.astype(float)
                / np.max(raw_phase_data[:].astype(float))
                * np.pi
            )

            complex_data = mag_data * np.exp(1j * phase_data)

            setattr(self, "IRData", complex_data)
        elif real is not None and imaginary is not None:
            real = Path(real)
            real_data = self._load_data(real)

            imaginary = Path(imaginary)
            imaginary_data = self._load_data(imaginary)

            complex_data = real_data + 1j * imaginary_data
            setattr(self, "IRData", complex_data)

        elif complex is not None:
            complex = Path(complex)
            self._load_data(complex, "IRData")

        else:
            magnitude = Path(magnitude)
            self._load_data(magnitude, "IRData")

        if Mask is not None:
            Mask = Path(Mask)
            mag_data = self._load_data(Mask, "Mask")

    def save(self, filename=None):
        if filename == None:
            filename = "T1.nii.gz"

        super().save(self.T1, filename)

    def fit(self, model="Barral"):
        # nonlin- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        IRData = self.IRData

        NoneType = type(None)

        if type(self.Mask) == NoneType:
            dshape = IRData.shape
            self.Mask = np.ones(dshape[0:3])
        Mask = self.Mask.astype(bool)

        # Get and format parameters
        inversion_times = np.array(self.params["inversion_times"])

        if model is "Barral":

            dshape = IRData.shape

            lin_data = IRData.reshape(-1, dshape[-1])
            lin_mask = Mask.reshape(-1)

            T1 = np.zeros(lin_data.shape[0])
            if np.iscomplex(IRData).all:
                a = np.zeros(lin_data.shape[0], dtype="complex_")
                b = np.zeros(lin_data.shape[0], dtype="complex_")
            else:
                a = np.zeros(lin_data.shape[0])
                b = np.zeros(lin_data.shape[0])
            residuals = np.zeros(lin_data.shape[0])
            idx = np.zeros(lin_data.shape[0])

            for vox in range(lin_data.shape[0]):

                if lin_mask[vox]:
                    (
                        T1[vox],
                        a[vox],
                        b[vox],
                        residuals[vox],
                        idx[vox],
                    ) = self._fit_barral(np.squeeze(lin_data[vox, :]), inversion_times)

            self.T1 = T1.reshape(dshape[0:3])
            self.a = a.reshape(dshape[0:3])
            self.b = b.reshape(dshape[0:3])
            self.residuals = residuals.reshape(dshape[0:3])
            self.idx = idx.reshape(dshape[0:3])

            # Apply masks and remove NaNs
            self._apply_mask(T1=self.T1)
            self._apply_mask(a=self.a)
            self._apply_mask(b=self.b)
            self._apply_mask(residuals=self.residuals)
            self._apply_mask(idx=self.idx)

    def _fit_barral(self, data, inversion_times):
        extra = {
            "TI_vec": inversion_times,
            "T1_vec": np.arange(start=0.001, stop=5.001, step=0.001),
        }

        nls_dict = self._get_nls_dict(extra)
        T1_est, a_est, b_est, residual, idx = self._rd_nls(data, nls_dict)

        # This case happens for complex data, maybe cleanup in _rd_nls in the
        # future
        if isinstance(a_est, np.ndarray):
            a_est = a_est[0]
        if isinstance(b_est, np.ndarray):
            b_est = b_est[0]

        return T1_est, a_est, b_est, residual, idx

    def _get_nls_dict(self, extra):
        nls_dict = {}
        nls_dict["TI_vec"] = extra["TI_vec"].reshape(-1, 1)
        nls_dict["N_TIs"] = len(nls_dict["TI_vec"])
        nls_dict["T1_vec"] = extra["T1_vec"]
        nls_dict["T1_start"] = nls_dict["T1_vec"][0]
        nls_dict["T1_stop"] = nls_dict["T1_vec"][-1]
        nls_dict["N_T1s"] = len(nls_dict["T1_vec"])

        # Hardcoded algorithm settings from Barral's oringal implementation
        nls_dict["nls_algorithm"] = "grid"
        nls_dict["number_zoom"] = 2
        nls_dict["zoom_seach_length"] = 21

        if nls_dict["nls_algorithm"] is "grid":
            alpha_vec = np.expand_dims(1 / nls_dict["T1_vec"], 0)
            nls_dict["the_exp"] = np.exp(-np.matmul(nls_dict["TI_vec"], alpha_vec))
            nls_dict["rho_norm_vec"] = np.expand_dims(
                np.sum(nls_dict["the_exp"] ** 2, 0)
                - 1 / nls_dict["N_TIs"] * (np.sum(nls_dict["the_exp"], 0) ** 2),
                1,
            )

        return nls_dict

    def _rd_nls(self, data, nls_dict):

        if nls_dict["nls_algorithm"] is "grid":

            if np.all(np.iscomplex(data)):
                (T1_est, b_est, a_est, residual,) = self._calc_nls_estimates(
                    data, nls_dict
                )
                idx = None
            else:
                # Ensure data is magnitude
                data = np.abs(data)

                # Initialize variables
                a_est = np.zeros(2)
                b_est = np.zeros(2)
                T1_est = np.zeros(2)
                residual_tmp = np.zeros(2)

                # Find the min of the data
                index_min = np.argmin(data)

                # loop for each
                for pol_index in [0, 1]:

                    if pol_index is 0:
                        # First, we set all elements up to and including
                        # the smallest element to minus

                        data_temp = data * np.concatenate(
                            (
                                -np.ones(index_min + 1),
                                np.ones(nls_dict["N_TIs"] - index_min - 1),
                            )
                        )
                    else:
                        # Second, we set all elements up to (not including)
                        # the smallest element to minus

                        data_temp = data * np.concatenate(
                            (
                                -np.ones(index_min),
                                np.ones(nls_dict["N_TIs"] - (index_min)),
                            )
                        )

                    (
                        T1_est[pol_index],
                        b_est[pol_index],
                        a_est[pol_index],
                        residual_tmp[pol_index],
                    ) = self._calc_nls_estimates(data_temp, nls_dict)

                ind = np.argmin(residual_tmp)
                a_est = a_est[ind]
                b_est = b_est[ind]
                T1_est = T1_est[ind]
                residual = residual_tmp[ind]

                if ind == 0:
                    # best fit when inverting the signal at the minimum
                    idx = index_min
                else:
                    # best fit when NOT inverting the signal at the minimum
                    idx = index_min - 1

        return T1_est, a_est, b_est, residual, idx

    def _calc_nls_estimates(self, data_temp, nls_dict):

        the_exp = nls_dict["the_exp"]

        # The sum of the data
        y_sum = np.sum(data_temp)

        # Compute the vector of rho'*t for different rho,
        # where rho = exp(-TI/T1) and y = dataTmp
        rho_t_y_vec = (
            np.matmul(data_temp, the_exp)
            - 1 / float(nls_dict["N_TIs"]) * np.sum(the_exp, 0) * y_sum
        )

        # rho_norm_vec is a vector containing the norm-squared of rho over TI,
        # where rho = exp(-TI/T1), for different T1's.
        rho_norm_vec = nls_dict["rho_norm_vec"]

        # Find the max of the maximizing criterion
        ind = np.argmax(np.abs(rho_t_y_vec.reshape(-1, 1)) ** 2 / rho_norm_vec)

        T1_vec = nls_dict["T1_vec"]

        zoom_seach_length = nls_dict["zoom_seach_length"]
        for _ in range(2, nls_dict["number_zoom"] + 1):

            if ind > 0 and ind < len(T1_vec) - 1:
                T1_vec = np.linspace(
                    T1_vec[ind - 1], T1_vec[ind + 1], zoom_seach_length
                )
            elif ind == 0:
                T1_vec = np.linspace(T1_vec[ind], T1_vec[ind + 2], zoom_seach_length)
            else:
                T1_vec = np.linspace(T1_vec[ind - 2], T1_vec[ind], zoom_seach_length)

            alpha_vec = np.expand_dims(1 / T1_vec, 0)
            the_exp = np.exp(-np.matmul(nls_dict["TI_vec"], alpha_vec))
            rho_norm_vec = np.expand_dims(
                np.sum(the_exp ** 2, 0)
                - 1 / nls_dict["N_TIs"] * (np.sum(the_exp, 0) ** 2),
                1,
            )

            y_exp_sum = np.matmul(data_temp.reshape(-1, 1).T, the_exp)
            rho_t_y_vec = y_exp_sum - 1 / nls_dict["N_TIs"] * np.sum(the_exp, 0) * y_sum

            ind = np.argmax(np.abs(rho_t_y_vec.reshape(-1, 1)) ** 2 / rho_norm_vec)

        # The estimated parameters
        T1_est = T1_vec[ind]
        b_est = rho_t_y_vec[:, ind] / rho_norm_vec[ind]
        a_est = 1 / nls_dict["N_TIs"] * (y_sum - b_est * np.sum(the_exp[:, ind]))

        # Compute the residual
        model_value = a_est + b_est * np.exp(-nls_dict["TI_vec"] / T1_est)
        residual = (
            1
            / np.sqrt(nls_dict["N_TIs"])
            * np.linalg.norm(1 - model_value.T / data_temp)
        )

        return T1_est, b_est, a_est, residual

    @staticmethod
    def simulate(params, type="analytical"):
        if type is "analytical":
            try:

                inversion_times = np.array(params["inversion_times"])  # seconds

                if "T1" in params:
                    T1 = params["T1"]
                else:
                    T1 = 0.850  # ms

                if "repetition_time" in params:
                    repetition_time = params["repetition_time"]  # seconds
                else:
                    repetition_time = np.inf

                if "excitation_flip_angle" in params:
                    excitation_flip_angle = np.deg2rad(params["excitation_flip_angle"])
                else:
                    excitation_flip_angle = np.deg2rad(90)

                if "inversion_flip_angle" in params:
                    inversion_flip_angle = np.deg2rad(params["inversion_flip_angle"])
                else:
                    inversion_flip_angle = np.deg2rad(190)

                if "constant" in params:
                    constant = params["constant"]
                else:
                    constant = 1  # unitless

                Mz = constant * (
                    (
                        1
                        - np.cos(inversion_flip_angle) * np.exp(-repetition_time / T1)
                        - (1 - np.cos(inversion_flip_angle))
                        * np.exp(-inversion_times / T1)
                    )
                    / (
                        1
                        - np.cos(inversion_flip_angle)
                        * np.cos(excitation_flip_angle)
                        * np.exp(-repetition_time / T1)
                    )
                )
                return Mz
            except:
                print("Incorrect parameters.")
