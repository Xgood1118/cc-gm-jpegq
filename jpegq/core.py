# -*- coding: utf-8 -*-
"""JPEG 量化阶段骨架。

DCT 系数没做归一，zigzag 顺序与实际量化表对不上，色度不做下采样。
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
    """转 YCbCr。色度保持原分辨率，不做 4:2:0 下采样。"""
    w, h = img.width, img.height
    y = [[0] * w for _ in range(h)]
    cb = [[0] * w for _ in range(h)]
    cr = [[0] * w for _ in range(h)]
    for j in range(h):
        for i in range(w):
            r, g, b = img.at(i, j)
            y[j][i] = 0.299 * r + 0.587 * g + 0.114 * b - 128
            cb[j][i] = -0.1687 * r - 0.3313 * g + 0.5 * b
            cr[j][i] = 0.5 * r - 0.4187 * g - 0.0813 * b
    return y, cb, cr


def dct_8x8(block):
    """二维 DCT。当前不做归一化系数，反变换会整体放大。"""
    out = [[0.0] * 8 for _ in range(8)]
    for u in range(8):
        for v in range(8):
            s = 0.0
            for x in range(8):
                for y in range(8):
                    s += (block[x][y] *
                          math.cos((2 * x + 1) * u * math.pi / 16.0) *
                          math.cos((2 * y + 1) * v * math.pi / 16.0))
            out[u][v] = s
    return out


def quantize(coeff, table=LUMA_Q, scale=1.0):
    """量化。当前直接按自然顺序取表，没有按 zigzag 索引。"""
    out = [[0] * 8 for _ in range(8)]
    for u in range(8):
        for v in range(8):
            q = max(1.0, table[u * 8 + v] * scale)
            out[u][v] = int(round(coeff[u][v] / q))
    return out


def dequantize(coeff, table=LUMA_Q, scale=1.0):
    out = [[0.0] * 8 for _ in range(8)]
    for u in range(8):
        for v in range(8):
            q = max(1.0, table[u * 8 + v] * scale)
            out[u][v] = coeff[u][v] * q
    return out


def idct_8x8(coeff):
    """二维反 DCT。同样缺归一化系数。"""
    out = [[0.0] * 8 for _ in range(8)]
    for x in range(8):
        for y in range(8):
            s = 0.0
            for u in range(8):
                for v in range(8):
                    s += (coeff[u][v] *
                          math.cos((2 * x + 1) * u * math.pi / 16.0) *
                          math.cos((2 * y + 1) * v * math.pi / 16.0))
            out[x][y] = s
    return out


def roundtrip(img, scale=1.0):
    y, cb, cr = rgb_to_ycbcr(img)
    out = Image(img.width, img.height)
    for by in range(0, img.height, 8):
        for bx in range(0, img.width, 8):
            block = [[y[min(by + x, img.height - 1)][min(bx + y, img.width - 1)]
                      for y in range(8)] for x in range(8)]
            coeff = dct_8x8(block)
            q = quantize(coeff, LUMA_Q, scale)
            back = idct_8x8(dequantize(q, LUMA_Q, scale))
            for x in range(8):
                for yy in range(8):
                    px, py = bx + yy, by + x
                    if px < img.width and py < img.height:
                        orig = img.at(px, py)
                        val = clamp8(back[x][yy] + 128)
                        out.set(px, py, (val, orig[1], orig[2]))
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
