#!/usr/bin/env python3
"""Qrop QR テスト用フォーム生成。

QRコード（フィールド仕様 CQR1 形式）と、その右側にOCR対象テキストを並べた
PNGを生成する。詳細フォーマットは docs/SPEC.md を参照。

依存: pip install qrcode pillow
"""
import qrcode
from PIL import Image, ImageDraw, ImageFont


def find_font(candidates, size):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def make_form(out_path, payload, text, lang):
    # 低バージョン化のため EC=L（黒モジュールを大きく＝ブラーに強い）
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=2)
    qr.add_data(payload)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    qs = qr_img.size[0]  # QRは正方形。これが「1単位」

    # キャンバス: 高さ=QR、幅=QRの右に十分なテキスト領域（SPECの x=1.2,w=8 を表現）
    canvas = Image.new("RGB", (qs * 11, qs), "white")
    canvas.paste(qr_img, (0, 0))

    if lang.startswith("ja"):
        font = find_font([
            "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        ], int(qs * 0.5))
    else:
        font = find_font([
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ], int(qs * 0.5))

    draw = ImageDraw.Draw(canvas)
    # フィールドは x=1.2*qs から（QR1.2個ぶん右）。縦は中央寄せ
    draw.text((int(qs * 1.2), int(qs * 0.25)), text, fill="black", font=font)
    canvas.save(out_path)
    print(f"wrote {out_path}  (payload={payload!r})")


if __name__ == "__main__":
    make_form("form_en.png", "CQR1,name,en,1.2,0,8,1", "Yamada Taro", "en")
    make_form("form_ja.png", "CQR1,user_name,ja_jp,1.2,0,8,1", "なかやま しんた", "ja")
