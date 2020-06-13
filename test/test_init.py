# coding: utf-8

import pytest

import pyqmrlab


class TestCore(object):

    # --------------module tests-------------- #
    def test_qmrlab_module(self):
        assert pyqmrlab.__version__
