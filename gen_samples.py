# -*- coding: utf-8 -*-
"""生成 96x64 测试图：左半渐变，右半高频条纹。"""
import os
import struct
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 96, 64


def main():
    raw = bytearray()
    for y in range(H):
        raw.append(0)
        for x in range(W):
            if x < W // 2:
                raw.extend((int(x * 2.4), int(y * 3.6), 120))
            else:
                v = 255 if (x // 2) % 2 else 20
                raw.extend((v, v, 60))
    def chunk(tag, body):
        return (struct.pack(">I", len(body)) + tag + body +
                struct.pack(">I", zlib.crc32(tag + body) & 0xFFFFFFFF))
    ihdr = struct.pack(">IIBBBBB", W, H, 8, 2, 0, 0, 0)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
            chunk(b"IDAT", zlib.compress(bytes(raw), 9)) + chunk(b"IEND", b""))
    os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
    path = os.path.join(HERE, "data", "photo.png")
    with open(path, "wb") as f:
        f.write(data)
    print("已生成", path, W, "x", H)


if __name__ == "__main__":
    main()
