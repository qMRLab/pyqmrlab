# coding: utf-8

import pytest
from pathlib import Path
from pyqmrlab.mt import mtsat
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

        mtsat_obj = mtsat()

        h = httplib2.Http()

        try:
            resp = h.request(mtsat_obj.data_url, "HEAD")
            assert int(resp[0]["status"]) < 400
        except Exception:
            pytest.fail("Website not found.")

    # --------------download tests-------------- #
    def test_download(self):
        mtsat_obj = mtsat()
        mtsat_obj.download(self.tmpPath)

        expected_folder = self.tmpPath / "mt_sat"
        expected_files = [
            self.tmpPath / "mt_sat/MTw.nii.gz",
            self.tmpPath / "mt_sat/PDw.nii.gz",
            self.tmpPath / "mt_sat/T1w.nii.gz",
        ]
        assert expected_folder.is_dir()
        for file in expected_files:
            assert file.is_file()

    # --------------load tests-------------- #
    def test_load(self):
        mtsat_obj = mtsat()

        MTw = self.tmpPath / "mt_sat/MTw.nii.gz"
        PDw = self.tmpPath / "mt_sat/PDw.nii.gz"
        T1w = self.tmpPath / "mt_sat/T1w.nii.gz"

        mtsat_obj.load(MTw, PDw, T1w)

        assert isinstance(mtsat_obj.MTw, np.ndarray)
        assert isinstance(mtsat_obj.PDw, np.ndarray)
        assert isinstance(mtsat_obj.T1w, np.ndarray)

        expected_shape = (128, 128, 96)
        assert mtsat_obj.MTw.shape == expected_shape
        assert mtsat_obj.PDw.shape == expected_shape
        assert mtsat_obj.T1w.shape == expected_shape

    # --------------fit tests-------------- #
    def test_fit(self):
        mtsat_obj = mtsat()

        MTw = self.tmpPath / "mt_sat/MTw.nii.gz"
        PDw = self.tmpPath / "mt_sat/PDw.nii.gz"
        T1w = self.tmpPath / "mt_sat/T1w.nii.gz"

        mtsat_obj.load(MTw, PDw, T1w)

        mtsat_obj.fit()

        expected_median_value = 2.6
        actual_median_value = np.median(mtsat_obj.MTsat[mtsat_obj.MTsat != 0])

        assert actual_median_value == pytest.approx(expected_median_value, abs=0.1)

    # --------------save tests-------------- #
    def test_save(self):
        mtsat_obj = mtsat()

        MTw = self.tmpPath / "mt_sat/MTw.nii.gz"
        PDw = self.tmpPath / "mt_sat/PDw.nii.gz"
        T1w = self.tmpPath / "mt_sat/T1w.nii.gz"

        mtsat_obj.load(MTw, PDw, T1w)

        mtsat_obj.fit()

        outputFile = [
            self.tmpPath / "MTsat.nii.gz",
            self.tmpPath / "T1.nii.gz",
        ]

        mtsat_obj.save(outputFile)

        for file in outputFile:
            assert file.is_file()
