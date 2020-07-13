# coding: utf-8

# Scientific modules imports

from collections import namedtuple
import numpy as np
import pyqmrlab.utils as utils
from pyqmrlab.abstract import Abstract

np.seterr(divide="ignore", invalid="ignore")


class InversionRecovery(Abstract):
    data_url = "https://osf.io/cmg9z/download?version=3"

    def __init__(self, params=None):
        if params == None:
            self.params = {
                "inversion_times": [350, 500, 650, 800, 950, 1100, 1250, 1400, 1700],
                "repetition_time": None,
            }
        else:
            self.params = {
                "inversion_times": params["inversion_times"],
                "repetition_time": params["repetition_time"],
            }

    def load(self, IRData, Mask=None):
        args = locals()
        super().load(args)

    def save(self, filename=None):
        if filename == None:
            filename = "T1.nii.gz"

        super().save(self.T1, filename)

    def fit(self, model="Barral"):
        # nonlin- https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html
        IRData = self.IRData
        Mask = self.Mask

        # Get and format parameters
        inversion_times = np.array(self.params["inversion_times"])
        repetition_time = np.array(self.params["repetition_time"])

        if model is "Barral":
            results = self._fit_barral(np.squeeze(IRData), inversion_times)
            self.T1 = results.T1
            self.a = results.a
            self.b = results.b
            self.residuals = results.residuals
            self.idx = results.idx

    def _fit_barral(self, data, inversion_times):
        extra = {
            "TI_vec": inversion_times,
            "T1_vec": np.arange(start=0.001, stop=5.001, step=0.001),
        }

        nls_dict = self._get_nls_dict(extra)
        results = self._rd_nls(data, nls_dict)

        return results

    def _get_nls_dict(self, extra):
        nls_dict = {}
        nls_dict["TI_vec"] = extra["TI_vec"][:]
        nls_dict["number_TIs"] = len(nls_dict["TI_vec"])
        nls_dict["T1_vec"] = extra["T1_vec"][:]
        nls_dict["T1_start"] = nls_dict["T1_vec"][0]
        nls_dict["T1_stop"] = nls_dict["T1_vec"][-1]
        nls_dict["number_T1s"] = len(nls_dict["T1_vec"])

        # Hardcoded algorithm settings from Barral's oringal implementation
        nls_dict["nls_algorithm"] = "grid"
        nls_dict["number_zoom"] = 2
        nls_dict["zoom_seach_length"] = 21

        if nls_dict["nls_algorithm"] is "grid":
            alpha_vec = np.expand_dims(1 / nls_dict["T1_vec"], 0)
            nls_dict["the_exp"] = np.exp(
                -np.matmul(nls_dict["TI_vec"].reshape(-1, 1), alpha_vec)
            )
            nls_dict["rho_norm_vec"] = np.expand_dims(
                np.sum(nls_dict["the_exp"] ** 2, 0)
                - 1 / nls_dict["number_TIs"] * (np.sum(nls_dict["the_exp"], 0) ** 2),
                1,
            )

        return nls_dict

    def _rd_nls(self, data, nls_dict):

        # Ensure data is magnitude
        data = np.abs(data)

        # Initialize variables
        a_est = np.zeros((1, 2))
        b_est = np.zeros((1, 2))
        T1_est = np.zeros((1, 2))
        res_est = np.zeros((1, 2))

        # Find the min of the data
        index_min = np.argmin(data)

        if nls_dict["nls_algorithm"] is "grid":
            resTmp = np.zeros((1, 2))
            # loop for each
            for pol_index in [0, 1]:

                theExp = nls_dict["the_exp"]

                if pol_index is 0:
                    # First, we set all elements up to and including
                    # the smallest element to minus
                    data_temp = data *  np.concatenate(
                        (
                            -np.ones(index_min+1),
                            np.ones(nls_dict["number_TIs"] - index_min-1)
                        )
                    )
                else:
                    # Second, we set all elements up to (not including)
                    # the smallest element to minus

                    data_temp = data * np.concatenate(
                        (
                            -np.ones(index_min),
                            np.ones(nls_dict["number_TIs"] - (index_min))
                        )
                    )

                # The sum of the data
                y_sum = np.sum(data_temp)
                

                # Compute the vector of rho'*t for different rho,
                # where rho = exp(-TI/T1) and y = dataTmp
                rhoTyVec = np.matmul(data_temp, theExp) - 1/float(nls_dict["number_TIs"])*np.sum(theExp,0)*y_sum

                # rhoNormVec is a vector containing the norm-squared of rho over TI,
                # where rho = exp(-TI/T1), for different T1's.
                rhoNormVec = nls_dict["rho_norm_vec"]
                
                # Find the max of the maximizing criterion
                ind = np.argmax( np.abs(rhoTyVec.reshape(-1, 1))**2/rhoNormVec )

                T1vec = nls_dict["T1_vec"]

                T1LenZ = nls_dict["zoom_seach_length"]
                for kk in range(2, nls_dict["number_zoom"]+1):

                    if( ind > 0 and ind < len(T1vec) ):
                        T1vec = np.linspace(T1vec[ind-1],T1vec[ind+1],T1LenZ)
                    elif(ind == 0):
                        T1vec = np.linspace(T1vec[ind],T1vec[ind+2],T1LenZ)
                    else:
                        T1vec = np.linspace(T1vec[ind-2],T1vec[ind],T1LenZ)

                    alpha_vec = np.expand_dims(1 / T1vec, 0)
                    the_exp = np.exp(
                        -np.matmul(nls_dict["TI_vec"].reshape(-1, 1), alpha_vec)
                    )
                    rho_norm_vec = np.expand_dims(
                        np.sum(the_exp ** 2, 0)
                        - 1 / nls_dict["number_TIs"] * (np.sum(the_exp, 0) ** 2),
                        1,
                    )

                    yExpSum = np.matmul(data_temp.reshape(-1, 1).T, the_exp)
                    rhoTyVec = yExpSum - 1/nls_dict["number_TIs"]*np.sum(the_exp,0)*y_sum

                    ind = np.argmax( np.abs(rhoTyVec.reshape(-1, 1))**2/rho_norm_vec )

                # The estimated parameters
                T1_est[:,pol_index] = T1vec[ind]
                b_est[:,pol_index] = rhoTyVec[:,ind] / rho_norm_vec[ind]
                a_est[:,pol_index] = 1/nls_dict["number_TIs"]*(y_sum - b_est[:,pol_index]*np.sum(the_exp[:,ind]))
                
                # Compute the residual
                modelValue = a_est[:,pol_index] + b_est[:,pol_index]*np.exp(-nls_dict["TI_vec"].reshape(-1, 1)/T1_est[:,pol_index])
                resTmp[:,pol_index] = 1/np.sqrt(nls_dict["number_TIs"]) * np.linalg.norm(1 - modelValue.T/data_temp)

        ind = np.argmin(resTmp)
        aEst = a_est[:,ind]
        bEst = b_est[:,ind]
        T1Est = T1_est[:,ind]
        if ind==0:
            idx = index_min; # best fit when inverting the signal at the minimum
        else:
            idx = index_min-1; # best fit when NOT inverting the signal at the minimum

        results = namedtuple("results", ["T1", "a", "b", "residuals", "idx"])
        return results(T1Est, aEst, bEst, resTmp[:,ind], idx)

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
