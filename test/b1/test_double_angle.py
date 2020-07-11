# coding: utf-8

import pytest
from pathlib import Path
from pyqmrlab.b1 import DoubleAngle
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

        double_angle_obj = DoubleAngle()

        h = httplib2.Http()

        try:
            resp = h.request(double_angle_obj.data_url, "HEAD")
            assert int(resp[0]["status"]) < 400
        except Exception:
            pytest.fail("Website not found.")

    # --------------download tests-------------- #
    def test_download(self):
        double_angle_obj = DoubleAngle()
        double_angle_obj.download(self.tmpPath)

        expected_folder = self.tmpPath / "b1_dam_multi-slice"
        expected_files = [
            self.tmpPath / "b1_dam_multi-slice/epseg_60deg.nii.gz",
            self.tmpPath / "b1_dam_multi-slice/epseg_120deg.nii.gz",
        ]
        assert expected_folder.is_dir()
        for file in expected_files:
            assert file.is_file()

    # --------------load tests-------------- #
    def test_load(self):
        double_angle_obj = DoubleAngle()

        img1 = self.tmpPath / "b1_dam_multi-slice/epseg_60deg.nii.gz"
        img2 = self.tmpPath / "b1_dam_multi-slice/epseg_120deg.nii.gz"

        double_angle_obj.load(img1, img2)

        assert isinstance(double_angle_obj.img1, np.ndarray)
        assert isinstance(double_angle_obj.img2, np.ndarray)

        expected_shape = (128, 128, 50)
        assert double_angle_obj.img1.shape == expected_shape
        assert double_angle_obj.img2.shape == expected_shape

    # --------------fit tests-------------- #
    def test_fit(self):
        double_angle_obj = DoubleAngle()

        img1 = self.tmpPath / "b1_dam_multi-slice/epseg_60deg.nii.gz"
        img2 = self.tmpPath / "b1_dam_multi-slice/epseg_120deg.nii.gz"

        double_angle_obj.load(img1, img2)

        double_angle_obj.fit()

        expected_median_value = 0.92
        actual_median_value = np.median(double_angle_obj.B1[double_angle_obj.B1 != 0])

        assert actual_median_value == pytest.approx(expected_median_value, abs=0.1)

    # --------------save tests-------------- #
    def test_save(self):
        double_angle_obj = DoubleAngle()

        img1 = self.tmpPath / "b1_dam_multi-slice/epseg_60deg.nii.gz"
        img2 = self.tmpPath / "b1_dam_multi-slice/epseg_120deg.nii.gz"

        double_angle_obj.load(img1, img2)

        double_angle_obj.fit()

        outputFile = self.tmpPath / "B1.nii.gz"
        double_angle_obj.save(outputFile)

        assert outputFile.is_file()
