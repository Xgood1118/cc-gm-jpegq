# -*- coding: utf-8 -*-
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jpegq.core import (  # noqa: E402
    Image, ZIGZAG, LUMA_Q, clamp8, dct_8x8, idct_8x8,
    quantize, dequantize, rgb_to_ycbcr, roundtrip, psnr,
)


def test_clamp():
    assert clamp8(-1) == 0
    assert clamp8(999) == 255


def test_image_default():
    img = Image(4, 4)
    assert img.at(0, 0) == (0, 0, 0)


def test_constant_block_roundtrip_lossless():
    """常数块 DCT -> IDCT 往返必须完全一致（1e-9）。"""
    for value in (0.0, 37.5, -128.0, 127.0):
        block = [[value] * 8 for _ in range(8)]
        back = idct_8x8(dct_8x8(block))
        for x in range(8):
            for y in range(8):
                assert abs(back[x][y] - value) < 1e-9


def test_dct_idct_roundtrip_no_shift():
    """任意块往返后不能整体偏移。"""
    block = [[float((x * 8 + y) * 3 - 90) for y in range(8)] for x in range(8)]
    back = idct_8x8(dct_8x8(block))
    for x in range(8):
        for y in range(8):
            assert abs(back[x][y] - block[x][y]) < 1e-9


def test_dct_dc_only():
    block = [[10.0] * 8 for _ in range(8)]
    coeff = dct_8x8(block)
    assert coeff[0][0] > 0
    # 常数块只有 DC 分量
    for u in range(8):
        for v in range(8):
            if (u, v) != (0, 0):
                assert abs(coeff[u][v]) < 1e-9


def test_quantize_zigzag_mapping():
    """量化表按 zigzag 顺序对应频率位置。

    LUMA_Q[1]=11 应作用于 ZIGZAG[1]=1 即 (0,1)，
    LUMA_Q[2]=10 应作用于 ZIGZAG[2]=8 即 (1,0)。
    若按自然顺序取表，(1,0) 会用到 LUMA_Q[8]=12，结果不同。
    """
    assert ZIGZAG[1] == 1 and ZIGZAG[2] == 8
    coeff = [[0.0] * 8 for _ in range(8)]
    coeff[0][1] = 6.0   # 6/11 -> 1；自然顺序下同样是 11，无法区分
    coeff[1][0] = 6.0   # 6/10 -> 1；自然顺序是 6/12 -> 0
    q = quantize(coeff, LUMA_Q, 1.0)
    assert q[0][1] == 1
    assert q[1][0] == 1


def test_quantize_zeroes_high_freq():
    coeff = [[0.0] * 8 for _ in range(8)]
    coeff[7][7] = 1.0
    q = quantize(coeff)
    assert q[7][7] == 0


def test_quantize_scale_monotonic():
    coeff = [[0.0] * 8 for _ in range(8)]
    coeff[2][2] = 30.0
    q1 = quantize(coeff, LUMA_Q, 1.0)[2][2]
    q3 = quantize(coeff, LUMA_Q, 3.0)[2][2]
    assert abs(q1) >= abs(q3)


def test_dequantize_inverse():
    coeff = [[0.0] * 8 for _ in range(8)]
    coeff[0][0] = 100.0
    q = quantize(coeff)
    back = dequantize(q)
    assert abs(back[0][0] - round(100.0 / 16) * 16) < 1e-9


def _gradient_image(w, h):
    img = Image(w, h)
    for j in range(h):
        for i in range(w):
            img.set(i, j, (int(i * 255 / max(1, w - 1)),
                           int(j * 255 / max(1, h - 1)), 120))
    return img


def test_chroma_half_resolution():
    img = _gradient_image(16, 16)
    y, cb, cr = rgb_to_ycbcr(img)
    assert len(y) == 16 and len(y[0]) == 16
    assert len(cb) == 8 and len(cb[0]) == 8
    assert len(cr) == 8 and len(cr[0]) == 8


def test_chroma_half_resolution_odd():
    img = _gradient_image(13, 9)
    y, cb, cr = rgb_to_ycbcr(img)
    assert len(cb) == 5 and len(cb[0]) == 7
    assert len(cr) == 5 and len(cr[0]) == 7


def test_roundtrip_rebuilds_all_channels():
    """写回时三个通道都要重建，纯色图往返后颜色不能跑偏。"""
    img = Image(16, 16, [(200, 30, 90)] * 256)
    out = roundtrip(img, 1.0)
    for j in range(16):
        for i in range(16):
            r, g, b = out.at(i, j)
            assert abs(r - 200) <= 8
            assert abs(g - 30) <= 8
            assert abs(b - 90) <= 8


def test_boundary_block_size_preserved():
    """宽高不是 8 的倍数时，往返后尺寸不变。"""
    for w, h in ((20, 12), (13, 9), (8, 8), (1, 1)):
        img = _gradient_image(w, h)
        out = roundtrip(img, 1.0)
        assert out.width == w and out.height == h


def test_boundary_block_no_streak():
    """边界块用镜像填充：边缘行/列不能与相邻像素出现拉伸式重复。"""
    img = _gradient_image(20, 12)
    out = roundtrip(img, 1.0)
    # 渐变图最右一列若被钳制复制，会与前一列完全相等；镜像填充不会
    diffs = [abs(out.at(19, j)[0] - out.at(18, j)[0]) for j in range(12)]
    assert max(diffs) > 0


def test_psnr_threshold_and_monotonic():
    img = _gradient_image(48, 32)
    out1 = roundtrip(img, 1.0)
    out3 = roundtrip(img, 3.0)
    p1 = psnr(img, out1)
    p3 = psnr(img, out3)
    assert p1 > 30.0
    assert p3 < p1
