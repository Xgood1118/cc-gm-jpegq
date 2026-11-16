# -*- coding: utf-8 -*-
"""JPEG 量化往返实现。

- DCT / IDCT 成对归一（正交归一系数 1/4 * C(u) * C(v)），往返无整体偏移。
- 量化表按 zigzag 顺序存放，量化时映射回自然 (u, v) 频率位置。
- 色度按 4:2:0 下采样（2x2 均值），写回时最近邻上采样并重建 RGB。
- 边界块用镜像（对称反射）填充，不靠钳制重复边缘像素。
"""

import math
import struct
import zlib

ZIGZAG = [
    0, 1, 8, 16, 9, 2, 3, 10,
    17, 24, 32, 25, 18, 11, 4, 5,
    12, 19, 26, 33, 40, 48, 41, 34,
    27, 20, 13, 6, 7, 14, 21, 28,
    35, 42, 49, 56, 57, 50, 43, 36,
    29, 22, 15, 23, 30, 37, 44, 51,
    58, 59, 52, 45, 38, 31, 39, 46,
    53, 60, 61, 54, 47, 55, 62, 63,
]

# 量化表按 zigzag 扫描顺序存放（低频在前，高频在后）。
LUMA_Q = [
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77,
    24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101,
    72, 92, 95, 98, 112, 100, 103, 99,
]

CHROMA_Q = [
    17, 18, 24, 47, 99, 99, 99, 99,
    18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99,
    47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99,
]

_COS = [[math.cos((2 * x + 1) * u * math.pi / 16.0) for x in range(8)]
        for u in range(8)]
_C = [1.0 / math.sqrt(2.0)] + [1.0] * 7


class Image(object):
    def __init__(self, width, height, pixels=None):
        self.width = int(width)
        self.height = int(height)
        self.pixels = pixels or [(0, 0, 0)] * (self.width * self.height)

    def at(self, x, y):
        return self.pixels[y * self.width + x]

    def set(self, x, y, rgb):
        self.pixels[y * self.width + x] = rgb


def load_png(path):
    with open(path, "rb") as f:
        data = f.read()
    pos = 8
    idat = b""
    width = height = ctype = None
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if tag == b"IHDR":
            width, height, _bit, ctype = struct.unpack(">IIBB", body[:10])
        elif tag == b"IDAT":
            idat += body
        elif tag == b"IEND":
            break
    raw = zlib.decompress(idat)
    ch = 3 if ctype == 2 else 4
    stride = width * ch
    pixels = []
    for y in range(height):
        ft = raw[y * (stride + 1)]
        line = raw[y * (stride + 1) + 1: y * (stride + 1) + 1 + stride]
        if ft != 0:
            raise ValueError("只支持无过滤")
        for x in range(width):
            pixels.append(tuple(line[x * ch:x * ch + 3]))
    return Image(width, height, pixels)


def save_png(img, path):
    raw = bytearray()
    for y in range(img.height):
        raw.append(0)
        for x in range(img.width):
            raw.extend(img.at(x, y))
    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", img.width, img.height, 8, 2, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    with open(path, "wb") as f:
        f.write(data)


def rgb_to_ycbcr(img):
    """转 YCbCr，色度按 4:2:0 下采样。

    返回 (y, cb, cr)：y 为全分辨率平面（已做 -128 电平偏移），
    cb / cr 为 ceil(w/2) x ceil(h/2) 的半分辨率平面（2x2 盒式均值，
    边缘不足 2x2 时按实际像素数平均）。
    """
    w, h = img.width, img.height
    y = [[0.0] * w for _ in range(h)]
    cw, chh = (w + 1) // 2, (h + 1) // 2
    cb = [[0.0] * cw for _ in range(chh)]
    cr = [[0.0] * cw for _ in range(chh)]
    for j in range(h):
        for i in range(w):
            r, g, b = img.at(i, j)
            y[j][i] = 0.299 * r + 0.587 * g + 0.114 * b - 128
    for j in range(chh):
        for i in range(cw):
            sb = sr = 0.0
            n = 0
            for dy in (0, 1):
                for dx in (0, 1):
                    px, py = 2 * i + dx, 2 * j + dy
                    if px < w and py < h:
                        r, g, b = img.at(px, py)
                        sb += -0.1687 * r - 0.3313 * g + 0.5 * b
                        sr += 0.5 * r - 0.4187 * g - 0.0813 * b
                        n += 1
            cb[j][i] = sb / n
            cr[j][i] = sr / n
    return y, cb, cr


def dct_8x8(block):
    """二维 DCT-II，带正交归一系数 1/4 * C(u) * C(v)。"""
    out = [[0.0] * 8 for _ in range(8)]
    for u in range(8):
        cu = _C[u]
        cosu = _COS[u]
        for v in range(8):
            cosv = _COS[v]
            s = 0.0
            for x in range(8):
                row = block[x]
                cx = cosu[x]
                for yy in range(8):
                    s += row[yy] * cx * cosv[yy]
            out[u][v] = 0.25 * cu * _C[v] * s
    return out


def idct_8x8(coeff):
    """二维 DCT-III（dct_8x8 的逆），同样带归一系数。"""
    out = [[0.0] * 8 for _ in range(8)]
    for x in range(8):
        for yy in range(8):
            s = 0.0
            for u in range(8):
                cu = _C[u] * _COS[u][x]
                row = coeff[u]
                for v in range(8):
                    s += _C[v] * row[v] * cu * _COS[v][yy]
            out[x][yy] = 0.25 * s
    return out


def _natural_table(table):
    """量化表按 zigzag 顺序给出，映射回自然 (u, v) 位置的 64 项表。"""
    nat = [0] * 64
    for k in range(64):
        nat[ZIGZAG[k]] = table[k]
    return nat


def quantize(coeff, table=LUMA_Q, scale=1.0):
    """量化。table 按 zigzag 顺序索引到对应的频率位置。"""
    nat = _natural_table(table)
    out = [[0] * 8 for _ in range(8)]
    for u in range(8):
        for v in range(8):
            q = max(1.0, nat[u * 8 + v] * scale)
            out[u][v] = int(round(coeff[u][v] / q))
    return out


def dequantize(coeff, table=LUMA_Q, scale=1.0):
    nat = _natural_table(table)
    out = [[0.0] * 8 for _ in range(8)]
    for u in range(8):
        for v in range(8):
            q = max(1.0, nat[u * 8 + v] * scale)
            out[u][v] = coeff[u][v] * q
    return out


def _mirror(i, n):
    """对称反射索引：-1 -> 0? 否，-1 -> 1 的整波反射；n -> n-2 方向。

    用于边界块填充：块外像素取镜像位置，而不是钳制重复边缘像素。
    """
    if n <= 1:
        return 0
    while i < 0 or i >= n:
        if i < 0:
            i = -i
        else:
            i = 2 * n - 2 - i
    return i


def _process_plane(plane, table, scale):
    """对一个平面做 分块 -> DCT -> 量化 -> 反量化 -> IDCT，返回同尺寸平面。"""
    h = len(plane)
    w = len(plane[0])
    out = [[0.0] * w for _ in range(h)]
    for by in range(0, h, 8):
        for bx in range(0, w, 8):
            block = [[plane[_mirror(by + x, h)][_mirror(bx + yy, w)]
                      for yy in range(8)] for x in range(8)]
            coeff = dct_8x8(block)
            q = quantize(coeff, table, scale)
            back = idct_8x8(dequantize(q, table, scale))
            for x in range(8):
                py = by + x
                if py >= h:
                    continue
                row = out[py]
                bkx = back[x]
                for yy in range(8):
                    px = bx + yy
                    if px < w:
                        row[px] = bkx[yy]
    return out


def roundtrip(img, scale=1.0):
    """完整量化往返：RGB -> YCbCr 4:2:0 -> 分块 DCT 量化 -> 重建 RGB。"""
    w, h = img.width, img.height
    y, cb, cr = rgb_to_ycbcr(img)
    yq = _process_plane(y, LUMA_Q, scale)
    cbq = _process_plane(cb, CHROMA_Q, scale)
    crq = _process_plane(cr, CHROMA_Q, scale)
    out = Image(w, h)
    for j in range(h):
        cj = j // 2
        cbrow = cbq[cj]
        crrow = crq[cj]
        yrow = yq[j]
        for i in range(w):
            yy = yrow[i] + 128
            cbv = cbrow[i // 2]
            crv = crrow[i // 2]
            r = clamp8(round(yy + 1.402 * crv))
            g = clamp8(round(yy - 0.344136 * cbv - 0.714136 * crv))
            b = clamp8(round(yy + 1.772 * cbv))
            out.set(i, j, (r, g, b))
    return out


def clamp8(v):
    return 0 if v < 0 else (255 if v > 255 else int(v))


def psnr(a, b):
    se = 0.0
    for j in range(a.height):
        for i in range(a.width):
            pa, pb = a.at(i, j), b.at(i, j)
            for k in range(3):
                d = pa[k] - pb[k]
                se += d * d
    mse = se / (a.width * a.height * 3)
    if mse == 0:
        return 99.0
    return 10 * math.log10(255.0 * 255.0 / mse)


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="jpegq")
    ap.add_argument("png")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="roundtrip.png")
    args = ap.parse_args(argv)
    img = load_png(args.png)
    out = roundtrip(img, args.scale)
    save_png(out, args.out)
    print("量化往返完成，PSNR %.2f dB" % psnr(img, out))
    return 0
