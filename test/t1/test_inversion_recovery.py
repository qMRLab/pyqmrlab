# coding: utf-8

import pytest
from pathlib import Path
from pyqmrlab.t1 import inversion_recovery
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
        pass

    # --------------attribute tests-------------- #
    def test_data_url_link_exists(self):

        ir_obj = inversion_recovery()

        h = httplib2.Http()

        try:
            resp = h.request(ir_obj.data_url, "HEAD")
            assert int(resp[0]["status"]) < 400
        except Exception:
            pytest.fail("Website not found.")

    # --------------download tests-------------- #
    def test_download(self):
        ir_obj = inversion_recovery()
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
        ir_obj = inversion_recovery()

        IRData = self.tmpPath / "inversion_recovery/IRData.mat"
        Mask = self.tmpPath / "inversion_recovery/Mask.mat"

        ir_obj.load(IRData, Mask)

        assert isinstance(ir_obj.IRData, np.ndarray)
        assert isinstance(ir_obj.Mask, np.ndarray)

        expected_shape_data = (128, 128, 1, 9)
        expected_shape_mask = (128, 128)

        assert ir_obj.IRData.shape == expected_shape_data
        assert ir_obj.Mask.shape == expected_shape_mask

    # --------------simulate tests-------------- #
    def test_simulate(self):
        ir_obj = inversion_recovery()
        params = {"FA": [3, 20], "TR": 0.015, "T1": 0.850}

        Mz = vfa.simulate(params, "analytical")

        expected_value = np.array([0.04859526, 0.07795592])
        actual_value = Mz

        assert np.allclose(actual_value, expected_value)

    # --------------fit tests-------------- #
    def test_fit_simulate_1vox(self):
        ir_obj = inversion_recovery()
        params = {"FA": [3, 20], "TR": 0.015, "T1": 0.850}

        Mz = vfa.simulate(params, "analytical")

        ir_obj.VFAData = np.ones((1, 1, 1, 2))

        ir_obj.VFAData[0, 0, 0, :] = Mz

        ir_obj.B1map = np.ones((1, 1))
        ir_obj.Mask = np.ones((1, 1))

        ir_obj.fit()

        expected_value = params["T1"]
        actual_value = ir_obj.T1

        assert actual_value == pytest.approx(expected_value, abs=0.01)

    def test_fit_simulate_3vox(self):
        ir_obj = inversion_recovery()
        params = {"FA": [3, 20], "TR": 0.015, "T1": 0.850}
        ir_obj.VFAData = np.ones((3, 1, 1, 2))
        ir_obj.Mask = np.ones((3, 1))
        ir_obj.B1map = np.ones((3, 1))
        ir_obj.B1map[1, 0] = 0.95
        ir_obj.B1map[2, 0] = 1.05

        Mz = vfa.simulate(params, "analytical")
        ir_obj.VFAData[0, 0, 0, :] = Mz

        params["FA"] = np.array([3, 20]) * ir_obj.B1map[1, 0]
        Mz = vfa.simulate(params, "analytical")
        ir_obj.VFAData[1, 0, 0, :] = Mz

        params["FA"] = np.array([3, 20]) * ir_obj.B1map[2, 0]
        Mz = vfa.simulate(params, "analytical")
        ir_obj.VFAData[2, 0, 0, :] = Mz

        ir_obj.fit()

        expected_value = np.array([params["T1"], params["T1"], params["T1"]])
        actual_value = ir_obj.T1

        assert np.allclose(actual_value, expected_value)

    def test_fit(self):
        ir_obj = inversion_recovery()

        VFAData = self.tmpPath / "vfa_t1/VFAData.nii.gz"
        B1map = self.tmpPath / "vfa_t1/B1map.nii.gz"
        Mask = self.tmpPath / "vfa_t1/Mask.nii.gz"

        ir_obj.load(VFAData, B1map, Mask)

        ir_obj.fit()

        expected_median_value = 1.33
        actual_median_value = np.median(ir_obj.T1[ir_obj.T1 != 0])

        assert actual_median_value == pytest.approx(expected_median_value, abs=0.1)

    # --------------save tests-------------- #
    def test_save(self):
        ir_obj = inversion_recovery()

        VFAData = self.tmpPath / "vfa_t1/VFAData.nii.gz"
        B1map = self.tmpPath / "vfa_t1/B1map.nii.gz"
        Mask = self.tmpPath / "vfa_t1/Mask.nii.gz"

        ir_obj.load(VFAData, B1map, Mask)

        ir_obj.fit()

        outputFile = self.tmpPath / "T1.nii.gz"
        ir_obj.save(outputFile)

        assert outputFile.is_file()
