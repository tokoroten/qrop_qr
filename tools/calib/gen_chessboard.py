#!/usr/bin/env python3
"""印刷用チェスボード(キャリブレーション用)を生成する。

既定: 10x7 マス（= 内側コーナー 9x6）。A4 に等倍印刷し、平らな板に貼って使う。
calibrate_fisheye.py の --cols/--rows（内側コーナー数）と必ず一致させること。

依存: pip install pillow
"""
import argparse
from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--squares-x", type=int, default=10, help="横のマス数（内側コーナーは-1）")
    ap.add_argument("--squares-y", type=int, default=7, help="縦のマス数（内側コーナーは-1）")
    ap.add_argument("--square-px", type=int, default=200, help="1マスのピクセル")
    ap.add_argument("--margin", type=int, default=120, help="周囲の白余白px")
    ap.add_argument("--out", default="chessboard_10x7.png")
    a = ap.parse_args()

    w = a.squares_x * a.square_px + a.margin * 2
    h = a.squares_y * a.square_px + a.margin * 2
    img = Image.new("RGB", (w, h), "white")
    d = ImageDraw.Draw(img)
    for j in range(a.squares_y):
        for i in range(a.squares_x):
            if (i + j) % 2 == 0:
                x0 = a.margin + i * a.square_px
                y0 = a.margin + j * a.square_px
                d.rectangle([x0, y0, x0 + a.square_px, y0 + a.square_px], fill="black")
    img.save(a.out)
    print(f"wrote {a.out}  {a.squares_x}x{a.squares_y} squares "
          f"(inner corners {a.squares_x-1}x{a.squares_y-1})  {w}x{h}px")


if __name__ == "__main__":
    main()
