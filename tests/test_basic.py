# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jpegq.core import Image, clamp8, dct_8x8, quantize  # noqa: E402


def test_clamp():
    assert clamp8(-1) == 0
    assert clamp8(999) == 255


def test_dct_dc_only():
    block = [[10.0] * 8 for _ in range(8)]
    coeff = dct_8x8(block)
    assert coeff[0][0] > 0


def test_quantize_zeroes_high_freq():
    coeff = [[0.0] * 8 for _ in range(8)]
    coeff[7][7] = 1.0
    q = quantize(coeff)
    assert q[7][7] == 0


def test_image_default():
    img = Image(4, 4)
    assert img.at(0, 0) == (0, 0, 0)
