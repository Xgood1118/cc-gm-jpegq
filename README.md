# DCT 量化往返

骨架能走通转色彩空间、分块 DCT、量化、反量化、反 DCT、写回 PNG 这条链路，但结果不能看。

## 已知缺口

- `dct_8x8` 与 `idct_8x8` 都没乘归一化系数，正变换放大 64 倍、反变换再放大 64 倍，
  往返后整体亮度溢出，画面全白。
- `quantize` 按自然顺序取量化表，实际要按 zigzag 顺序索引，高低频的量化强度反了。
- 色度不做 4:2:0 下采样，CB 与 CR 保持原分辨率，压缩率与真实 JPEG 差一倍。
- 写回时只替换了 Y，CB 与 CR 直接沿用原值，等于只做了亮度通道。
- 没有 DC 预测，每块都独立编码。
- 边界块用钳制取像素，右下角的块会把边缘像素重复，出现拉伸痕迹。

## 运行

```
python gen_samples.py
python -m jpegq.cli data/photo.png --scale 1.0 --out roundtrip.png
python -m pytest tests -q
```

`photo.png` 96x64，左半是双通道渐变，右半是两像素宽的高频条纹，用来检验量化对高频的影响。
