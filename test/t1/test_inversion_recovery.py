# coding: utf-8

import pytest
from pathlib import Path
from pyqmrlab.t1 import InversionRecovery
from pyqmrlab.utils import download_data
import httplib2
import shutil
import numpy as np
import scipy as sp


class TestCore(object):
    def setup(self):
        self.tmpPath = Path("tmp/")

    @classmethod
    def teardown_class(cls):
        tmpPath = Path("tmp/")
        shutil.rmtree(tmpPath)

    # --------------attribute tests-------------- #
    def test_data_url_link_exists(self):

        ir_obj = InversionRecovery()

        h = httplib2.Http()

        try:
            resp = h.request(ir_obj.data_url, "HEAD")
            assert int(resp[0]["status"]) < 400
        except Exception:
            pytest.fail("Website not found.")

    # --------------download tests-------------- #
    def test_download(self):
        ir_obj = InversionRecovery()
        ir_obj.download(self.tmpPath)

        expected_folder = self.tmpPath / "inversion_recovery"
        expected_files = [
            self.tmpPath / "inversion_recovery/IRData.mat",
            self.tmpPath / "inversion_recovery/Mask.mat",
        ]
        assert expected_folder.is_dir()
        for file in expected_files:
            assert file.is_file()

    # --------------load tests-------------- #
    def test_load(self):
        ir_obj = InversionRecovery()

        IRData = self.tmpPath / "inversion_recovery/IRData.mat"
        Mask = self.tmpPath / "inversion_recovery/Mask.mat"

        ir_obj.load(magnitude=IRData, Mask=Mask)

        assert isinstance(ir_obj.IRData, np.ndarray)
        assert isinstance(ir_obj.Mask, np.ndarray)

        expected_shape_data = (128, 128, 1, 9)
        expected_shape_mask = (128, 128)

        assert ir_obj.IRData.shape == expected_shape_data
        assert ir_obj.Mask.shape == expected_shape_mask

    # --------------load tests-------------- #
    def test_load_complex_mag_phase(self):
        url_data_mag_phase = "https://osf.io/67b8p/download/"
        download_data(url_data_mag_phase, folder=self.tmpPath)

        magnitude = (
            self.tmpPath
            / "20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST_Magnitude.nii.gz"
        )
        phase = (
            self.tmpPath
            / "20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST_Phase.nii.gz"
        )

        ir_obj = InversionRecovery()
        ir_obj.load(magnitude=magnitude, phase=phase)

        assert isinstance(ir_obj.IRData, np.ndarray)
        assert np.iscomplex(ir_obj.IRData).all

    # --------------simulate tests-------------- #
    def test_simulate(self):

        params = {
            "excitation_flip_angle": 90,
            "inversion_flip_angle": 180,
            "inversion_times": [0.050, 0.400, 1.100, 2.500],
            "repetition_time": 2.550,
            "T1": 0.850,
        }

        Mz = InversionRecovery.simulate(params, "analytical")

        expected_value = np.array([-0.83595922, -0.19948239, 0.50150778, 0.94417993])
        actual_value = Mz

        assert np.allclose(actual_value, expected_value)

    # --------------fit tests-------------- #
    def test_fit_simulate_1vox(self):
        params = {
            "excitation_flip_angle": 90,
            "inversion_flip_angle": 180,
            "inversion_times": [0.050, 0.400, 1.100, 2.500],
            "repetition_time": 2.550,
            "T1": 0.850,
        }

        ir_obj = InversionRecovery(params)

        Mz = InversionRecovery.simulate(params, "analytical")

        ir_obj.IRData = np.zeros((1, 1, 1, len(params["inversion_times"])))

        ir_obj.IRData[0, 0, 0, :] = np.abs(Mz)

        ir_obj.Mask = np.ones((1, 1))

        ir_obj.fit()

        expected_value = params["T1"]
        actual_value = ir_obj.T1

        assert actual_value == pytest.approx(expected_value, abs=0.01)

    def test_fit_simulate_1vox_complex(self):
        magnitude = 112
        phase = -np.pi / 4

        params = {
            "excitation_flip_angle": 90,
            "inversion_flip_angle": 180,
            "inversion_times": [0.050, 0.400, 1.100, 2.500],
            "repetition_time": 2.550,
            "T1": 1.150,
            "constant": magnitude * np.exp(1j * phase),
        }

        ir_obj = InversionRecovery(params)

        Mz = InversionRecovery.simulate(params, "analytical")

        ir_obj.IRData = np.zeros(
            (1, 1, 1, len(params["inversion_times"])), dtype="complex_"
        )

        ir_obj.IRData[0, 0, 0, :] = Mz

        assert np.all(np.iscomplex(ir_obj.IRData))

        ir_obj.Mask = np.ones((1, 1))

        ir_obj.fit()

        expected_value = params["T1"]
        actual_value = ir_obj.T1

        assert actual_value == pytest.approx(expected_value, abs=0.01)

        assert np.iscomplex(ir_obj.a).all
        assert np.iscomplex(ir_obj.b).all

    def test_fit_simulate_3vox(self):

        T1_arr = [0.850, 1.1, 0.53]
        params = {
            "excitation_flip_angle": 90,
            "inversion_flip_angle": 180,
            "inversion_times": [0.050, 0.400, 1.100, 2.500],
            "repetition_time": 2.550,
            "T1": T1_arr[0],
        }
        ir_obj = InversionRecovery(params)

        ir_obj.IRData = np.zeros((3, 1, 1, 4))
        ir_obj.Mask = np.ones((3, 1))

        Mz = InversionRecovery.simulate(params, "analytical")
        ir_obj.IRData[0, 0, 0, :] = np.abs(Mz)

        params["T1"] = T1_arr[1]
        Mz = InversionRecovery.simulate(params, "analytical")
        ir_obj.IRData[1, 0, 0, :] = np.abs(Mz)

        params["T1"] = T1_arr[2]
        Mz = InversionRecovery.simulate(params, "analytical")
        ir_obj.IRData[2, 0, 0, :] = np.abs(Mz)

        ir_obj.fit()

        expected_value = np.array([T1_arr[0], T1_arr[1], T1_arr[2]])
        actual_value = np.squeeze(ir_obj.T1)

        assert np.allclose(actual_value, expected_value)

    def test_fit(self):
        ir_obj = InversionRecovery()

        IRData = self.tmpPath / "inversion_recovery/IRData.mat"
        Mask = self.tmpPath / "inversion_recovery/Mask.mat"

        ir_obj.load(magnitude=IRData, Mask=Mask)

        ir_obj.fit()

        expected_median_value = 0.7621
        actual_median_value = np.median(ir_obj.T1[ir_obj.T1 != 0])

        assert actual_median_value == pytest.approx(expected_median_value, abs=0.001)

    def test_fit_complex_mag_phase(self):
        url_data_mag_phase = "https://osf.io/67b8p/download/"
        download_data(url_data_mag_phase, folder=self.tmpPath)

        magnitude = (
            self.tmpPath
            / "20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST_Magnitude.nii.gz"
        )
        phase = (
            self.tmpPath
            / "20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST/20200121_matthewgrechsollars_ICL_NIST_Phase.nii.gz"
        )

        ir_obj = InversionRecovery(
            params={"inversion_times": [0.050, 0.400, 1.100, 2.400]}
        )
        ir_obj.load(magnitude=magnitude, phase=phase)

        ir_obj.fit()

        expected_median_value = 2.2035
        actual_median_value = np.median(ir_obj.T1[ir_obj.T1 != 0])

        outputFile = self.tmpPath / "T1.nii.gz"
        ir_obj.save(outputFile)

        assert actual_median_value == pytest.approx(expected_median_value, abs=0.001)

        assert np.iscomplex(ir_obj.a).all
        assert np.iscomplex(ir_obj.b).all



    def test_fit_complex_real_imaginary(self):
        url_data_mag_phase = "https://osf.io/qnhjt/download/"
        download_data(url_data_mag_phase, folder=self.tmpPath)

        real = (
            self.tmpPath
            / "20200210_guillaumegilbert_muhc_NIST/20200210_guillaumegilbert_muhc_NIST/20200210_guillaumegilbert_muhc_NIST_Real.nii.gz"
        )
        imaginary = (
            self.tmpPath
            / "20200210_guillaumegilbert_muhc_NIST/20200210_guillaumegilbert_muhc_NIST/20200210_guillaumegilbert_muhc_NIST_Imaginary.nii.gz"
        )

        ir_obj = InversionRecovery(
            params={"inversion_times": [0.050, 0.400, 1.100, 2.500]}
        )
        ir_obj.load(real=real, imaginary=imaginary)

        ir_obj.fit()

        expected_median_value = 3.4587
        actual_median_value = np.median(ir_obj.T1[ir_obj.T1 != 0])

        outputFile = self.tmpPath / "T1.nii.gz"
        ir_obj.save(outputFile)

        assert actual_median_value == pytest.approx(expected_median_value, abs=0.001)

        assert np.iscomplex(ir_obj.a).all
        assert np.iscomplex(ir_obj.b).all
    # --------------save tests-------------- #

    def test_save(self):
        ir_obj = InversionRecovery()

        IRData = self.tmpPath / "inversion_recovery/IRData.mat"
        Mask = self.tmpPath / "inversion_recovery/Mask.mat"

        ir_obj.load(magnitude=IRData, Mask=Mask)

        ir_obj.fit()

        outputFile = self.tmpPath / "T1.nii.gz"
        ir_obj.save(outputFile)

        assert outputFile.is_file()
