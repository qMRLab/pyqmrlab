# coding: utf-8
"""Calculate Inversion Recovery (IR) T1 mapping

This module is for fitting T1 from inversion recovery data acquired with
different inversion times.

  Typical usage example:

  from pathlib import Path
  from pyqmrlab.t1 import InversionRecovery

  ir_obj = InversionRecovery()
  
  # Download sample dataset
  ir_obj.download(folder='data/')

  # Load data
  data_folder = Path('data/')
  
  magnitude_data = data_folder / "inversion_recovery/IRData.mat"
  mask_data = data_folder / "inversion_recovery/Mask.mat"
  
  ir_obj.load(magnitude=magnitude_data, Mask = mask_data)

  # Fit data
  ir_obj.fit()

  # Save to NIFTI
  ir_obj.save(filename = data_folder / 'T1.nii.gz')
"""

from pyqmrlab.abstract import *

np.seterr(divide="ignore", invalid="ignore")


class InversionRecovery(Abstract):
    """Inversion recovery (IR) T1 mapping data processing class.

    Fits inversionrecovery T1 mapping data and saves to NIfTI. Demo dataset
    available for download.

    Attributes:
        data_url: Link to demo dataset.
        params: Dictionnary of pulse sequence parameters for each 
                measurements. Measurement keys: inversion_times (seconds), 
                repetition_time (seconds).
    """

    IRData = None
    Mask = None
    data_url = "https://osf.io/cmg9z/download?version=3"

    T1 = None
    a = None
    b = None
    residuals = None
    idx = None

    def __init__(self, params=None):
        """Initializes an InversionRecovery object.

        Assigns inversion recovery T1 mapping pulse sequence parameters to the
        `params` attribute, or if none are given assigns the parameters to be
        used with the demo data.

        Args:
        params: Dictionnary of pulse sequence parameters for each 
                measurements. Measurement keys: inversion_times (degrees), 
                repetition_time (seconds).

        Returns:
            InversionRecovery class object with parameters initialized.
        """
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
        """Load dataset

        Loads an inversion recovery dataset into the class object. Compatible
        with NIfTI and *.mat filetypes. Data provided can be: magnitude-only,
        magnitude and phase, real and imaginary, or complex (*.mat only).

        Args:
        magnitude: Path to magnitude dataset.
        phase: Path to phase dataset. Phase data is normalized in this method
                to set the range to -pi and pi.
        real: Path to real dataset.
        imaginary: Path to imaginary dataset.
        complex: Path to complex dataset (*.mat only)
        mask: Path to binary mask. Optional.

        """
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
            self._load_data(Mask, "Mask")

    def save(self, filename=None):
        if filename == None:
            filename = "T1.nii.gz"

        super().save(self.T1, filename)

    def fit(self, model="Barral"):
        """Fit data from an inversion recovery T1 mapping experiment.

        Generates longitudinal magnetization for the spoiled gradient echo 
        pulse sequence. All fitting models are compatible with "short" TR
        pulse sequence protocols (i.e. full recovery is not assumed).

        Args:
        model: Fitting model. Options: "Barral".

        Implementation details
        "Barral": Fits the equation a+b*exp(-TI/T1), where a and b will be
        complex constants if the data is also complex. Repetition time is not
        used.

        Barral, J.K., Gudmundson, E., Stikov, N., Etezadi‐Amoli, M., Stoica, P.
        and Nishimura, D.G. (2010), A robust methodology for in vivo T1
        mapping. Magn. Reson. Med., 64: 1057-1067. doi:10.1002/mrm.22497
        """

        # Note for future implementations:
        # Nonlinear - https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html

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

            results = {}
            results['T1'] = np.zeros(lin_data.shape[0])

            if np.iscomplex(IRData).any():
                results['a'] = np.zeros(lin_data.shape[0], dtype="complex_")
                results['b'] = np.zeros(lin_data.shape[0], dtype="complex_")
            else:
                results['a'] = np.zeros(lin_data.shape[0])
                results['b'] = np.zeros(lin_data.shape[0])
            results['residuals'] = np.zeros(lin_data.shape[0])
            results['idx'] = np.zeros(lin_data.shape[0])

            for vox in range(lin_data.shape[0]):

                if lin_mask[vox]:
                    (
                        results['T1'][vox],
                        results['a'][vox],
                        results['b'][vox],
                        results['residuals'][vox],
                        results['idx'][vox],
                    ) = self._fit_barral(np.squeeze(lin_data[vox, :]), inversion_times)

            for key, value in results.items():
                # Reshape vector back to volume dimensions
                setattr(self, key, value.reshape(dshape[0:3]))

                # Apply masks and remove NaNs
                self._apply_mask(**{key: getattr(self, key)})

    def _fit_barral(self, data, inversion_times):
        """T1 fitting algorithm using the Barral method

        The algorithm is an adaptation of Joelle Barral's open-source MATLAB
        implementation: http://www-mrsrl.stanford.edu/~jbarral/t1map.html

        Reference:
        Barral, J.K., Gudmundson, E., Stikov, N., Etezadi‐Amoli, M., Stoica, P.
        and Nishimura, D.G. (2010), A robust methodology for in vivo T1
        mapping. Magn. Reson. Med., 64: 1057-1067. doi:10.1002/mrm.22497
        """
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
        """Define the dictionary for the non-linear least square implementation
        used in the Barral method.
        """
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
        """Reduced-dimension non-linear least squares implementation for the
        Barral method.
        """
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
        """Calculate the estimates for the non-linear least squares of the
        Barral implementation.
        """
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
        """Simulates signal for a inversion recovery T1 mapping experiment.

        Generates longitudinal magnetization for the inversion recovery
        pulse sequence.

        Args:
        params: Dictionnary of pulse sequence parameters for each 
                measurements, as well as T1 of ths pins. Parameter keys:
                excitation_flip_angle (numpy array of angles in degrees),
                inversion_flip_angle (numpy array of angles in degrees), 
                repetition_time (seconds), constant (real or complex),
                T1 (seconds).
        type: Simulation type. 'analytical': assumes the analytical
              steady-state solution to the pulse sequence experiment.

        Returns:
            Mz: numpy array of the longitudinal magnetization for each
                inversion recovery measurement.
        """
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
