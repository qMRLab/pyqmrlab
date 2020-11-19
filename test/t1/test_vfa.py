# coding: utf-8

import pytest
from pathlib import Path
from pyqmrlab.t1 import VFA
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
        if tmpPath.exists():
            shutil.rmtree(tmpPath)

    # --------------attribute tests-------------- #
    def test_data_url_link_exists(self):

        vfa_obj = VFA()

        h = httplib2.Http()

        try:
            resp = h.request(vfa_obj.data_url, "HEAD")
            assert int(resp[0]["status"]) < 400
        except Exception:
            pytest.fail("Website not found.")

    # --------------initialization tests-------------- #
    def test_passed_parameters_are_stored(self):
        params = {"flip_angle": [1, 2, 3, 4], "repetition_time": 0.030, "T1": 1.1}
        vfa_obj = VFA(params)

        expected_flip_angle = params['flip_angle']
        actual_flip_angle = vfa_obj.params['flip_angle']

        assert expected_flip_angle == actual_flip_angle

        expected_repetition_time = params['repetition_time']
        actual_repetition_time = vfa_obj.params['repetition_time']

        assert expected_repetition_time == actual_repetition_time

    # --------------download tests-------------- #
    def test_download(self):
        vfa_obj = VFA()
        vfa_obj.download(self.tmpPath)

        expected_folder = self.tmpPath / "vfa_t1"
        expected_files = [
            self.tmpPath / "vfa_t1/VFAData.nii.gz",
            self.tmpPath / "vfa_t1/B1map.nii.gz",
            self.tmpPath / "vfa_t1/Mask.nii.gz",
        ]
        assert expected_folder.is_dir()
        for file in expected_files:
            assert file.is_file()

    # --------------load tests-------------- #
    def test_load(self):
        vfa_obj = VFA()

        VFAData = self.tmpPath / "vfa_t1/VFAData.nii.gz"
        B1map = self.tmpPath / "vfa_t1/B1map.nii.gz"
        Mask = self.tmpPath / "vfa_t1/Mask.nii.gz"

        vfa_obj.load(VFAData, B1map, Mask)

        assert isinstance(vfa_obj.VFAData, np.ndarray)
        assert isinstance(vfa_obj.B1map, np.ndarray)
        assert isinstance(vfa_obj.Mask, np.ndarray)

        expected_shape_data = (128, 128, 1, 2)
        expected_shape_b1 = (128, 128)
        expected_shape_mask = (128, 128)

        assert vfa_obj.VFAData.shape == expected_shape_data
        assert vfa_obj.B1map.shape == expected_shape_b1
        assert vfa_obj.Mask.shape == expected_shape_mask

    # --------------simulate tests-------------- #
    def test_simulate(self):
        params = {"flip_angle": [3, 20], "repetition_time": 0.015, "T1": 0.850}
        vfa_obj = VFA(params)

        Mz = VFA.simulate(params, "analytical")

        expected_value = np.array([0.04859526, 0.07795592])
        actual_value = Mz

        assert np.allclose(actual_value, expected_value)

    # --------------fit tests-------------- #
    def test_fit_simulate_1vox(self):
        params = {"flip_angle": [3, 20], "repetition_time": 0.015, "T1": 0.850}
        vfa_obj = VFA(params)

        Mz = VFA.simulate(params, "analytical")

        vfa_obj.VFAData = np.ones((1, 1, 1, 2))

        vfa_obj.VFAData[0, 0, 0, :] = Mz

        vfa_obj.B1map = np.ones((1, 1))
        vfa_obj.Mask = np.ones((1, 1))

        vfa_obj.fit()

        expected_value = params["T1"]
        actual_value = vfa_obj.T1

        assert actual_value == pytest.approx(expected_value, abs=0.01)

    def test_fit_simulate_3vox(self):
        params = {"flip_angle": [3, 20], "repetition_time": 0.015, "T1": 0.850}
        vfa_obj = VFA(params)

        vfa_obj.VFAData = np.ones((3, 1, 1, 2))
        vfa_obj.Mask = np.ones((3, 1))
        vfa_obj.B1map = np.ones((3, 1))
        vfa_obj.B1map[1, 0] = 0.95
        vfa_obj.B1map[2, 0] = 1.05

        Mz = VFA.simulate(params, "analytical")
        vfa_obj.VFAData[0, 0, 0, :] = Mz
        
        params["flip_angle"] = np.array([3, 20]) * vfa_obj.B1map[1, 0]
        Mz = VFA.simulate(params, "analytical")
        vfa_obj.VFAData[1, 0, 0, :] = Mz

        params["flip_angle"] = np.array([3, 20]) * vfa_obj.B1map[2, 0]
        Mz = VFA.simulate(params, "analytical")
        vfa_obj.VFAData[2, 0, 0, :] = Mz

        vfa_obj.fit()

        expected_value = np.array([params["T1"], params["T1"], params["T1"]])
        actual_value = vfa_obj.T1

        assert np.allclose(actual_value, expected_value)

    def test_fit_simulate_1vox_3_angles(self):
        params = {"flip_angle": [3, 10, 20], "repetition_time": 0.015, "T1": 0.850}
        vfa_obj = VFA(params)

        Mz = VFA.simulate(params, "analytical")

        vfa_obj.VFAData = np.ones((1, 1, 1, 3))

        vfa_obj.VFAData[0, 0, 0, :] = Mz

        vfa_obj.B1map = np.ones((1, 1))
        vfa_obj.Mask = np.ones((1, 1))

        vfa_obj.fit()

        expected_value = params["T1"]
        actual_value = vfa_obj.T1

        assert actual_value == pytest.approx(expected_value, abs=0.01)

    def test_fit(self):
        vfa_obj = VFA()

        VFAData = self.tmpPath / "vfa_t1/VFAData.nii.gz"
        B1map = self.tmpPath / "vfa_t1/B1map.nii.gz"
        Mask = self.tmpPath / "vfa_t1/Mask.nii.gz"

        vfa_obj.load(VFAData, B1map, Mask)

        vfa_obj.fit()

        expected_median_value = 1.33
        actual_median_value = np.median(vfa_obj.T1[vfa_obj.T1 != 0])

        assert actual_median_value == pytest.approx(expected_median_value, abs=0.1)

    # --------------save tests-------------- #
    def test_save(self):
        vfa_obj = VFA()

        VFAData = self.tmpPath / "vfa_t1/VFAData.nii.gz"
        B1map = self.tmpPath / "vfa_t1/B1map.nii.gz"
        Mask = self.tmpPath / "vfa_t1/Mask.nii.gz"

        vfa_obj.load(VFAData, B1map, Mask)

        vfa_obj.fit()

        outputFile = self.tmpPath / "T1.nii.gz"
        vfa_obj.save(outputFile)

        assert outputFile.is_file()
