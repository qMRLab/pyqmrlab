# coding: utf-8

import pytest
from pathlib import Path
from pyqmrlab.mt import mtr
import httplib2
import shutil
import numpy as np


class TestCore(object):
    def setup(self):
        self.tmpPath = Path("tmp/")

    @classmethod
    def teardown_class(cls):
        tmpPath = Path("tmp/")
        shutil.rmtree(tmpPath)

    # --------------attribute tests-------------- #
    def test_data_url_link_exists(self):

        mtr_obj = mtr()

        h = httplib2.Http()

        try:
            resp = h.request(mtr_obj.data_url, "HEAD")
            assert int(resp[0]["status"]) < 400
        except Exception:
            pytest.fail("Website not found.")

    # --------------download tests-------------- #
    def test_download(self):
        mtr_obj = mtr()
        mtr_obj.download(self.tmpPath)

        expected_folder = self.tmpPath / "mt_ratio"
        expected_files = [
            self.tmpPath / "mt_ratio/MToff.mat",
            self.tmpPath / "mt_ratio/Mask.mat",
            self.tmpPath / "mt_ratio/MTon.mat",
        ]
        assert expected_folder.is_dir()
        for file in expected_files:
            assert file.is_file()

    # --------------load tests-------------- #
    def test_load(self):
        mtr_obj = mtr()

        MTon = self.tmpPath / "mt_ratio/MTon.mat"
        MToff = self.tmpPath / "mt_ratio/MToff.mat"
        Mask = self.tmpPath / "mt_ratio/Mask.mat"

        mtr_obj.load(MTon, MToff, Mask)

        assert isinstance(mtr_obj.MTon, np.ndarray)
        assert isinstance(mtr_obj.MToff, np.ndarray)
        assert isinstance(mtr_obj.MToff, np.ndarray)

        expected_shape = (128, 135, 75)
        assert mtr_obj.MTon.shape == expected_shape
        assert mtr_obj.MToff.shape == expected_shape
        assert mtr_obj.MToff.shape == expected_shape

    # --------------fit tests-------------- #
    def test_fit(self):
        mtr_obj = mtr()

        MTon = self.tmpPath / "mt_ratio/MTon.mat"
        MToff = self.tmpPath / "mt_ratio/MToff.mat"
        Mask = self.tmpPath / "mt_ratio/Mask.mat"

        mtr_obj.load(MTon, MToff, Mask)

        mtr_obj.fit()

        expected_mean_value = 43.183624
        actual_mean_value = np.mean(mtr_obj.MTR[mtr_obj.MTR != 0])

        assert actual_mean_value == pytest.approx(expected_mean_value)

    # --------------save tests-------------- #
    def test_save(self):
        mtr_obj = mtr()

        MTon = self.tmpPath / "mt_ratio/MTon.mat"
        MToff = self.tmpPath / "mt_ratio/MToff.mat"
        Mask = self.tmpPath / "mt_ratio/Mask.mat"

        mtr_obj.load(MTon, MToff, Mask)

        mtr_obj.fit()

        outputFile = self.tmpPath / "MTR.nii.gz"
        mtr_obj.save(outputFile)

        assert outputFile.is_file()
