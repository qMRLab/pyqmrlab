# coding: utf-8

import pytest

import pyqmrlab

class TestCore(object):

    # --------------module tests-------------- #
    @pytest.mark.unit
    def test_qmrlab_module(self):

        assert pyqmrlab.__version__
