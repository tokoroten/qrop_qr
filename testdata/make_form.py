#!/usr/bin/env python3
"""Qrop QR テスト用フォーム生成（単一フィールド）。

QRコード（フィールド仕様 CQR1 形式）と、その指定オフセットにOCR対象テキストを置いた
PNGを生成する。詳細フォーマットは docs/SPEC.md を参照。

重要: フィールドのオフセット/サイズは「QRシンボル（黒モジュール部分）の左上・一辺」を
1単位として計算する（クワイエットゾーン=白枠は含めない）。アプリは ML Kit cornerPoints=
シンボル四隅を単位正方形にするため、白枠込みで計算すると遠いフィールドほど大きくズレる。

依存: pip install qrcode pillow
"""
import qrcode
from PIL import Image, ImageDraw, ImageFont

JA_FONTS = ["C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc",
            "C:/Windows/Fonts/msgothic.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
EN_FONTS = ["C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]
BORDER = 4  # クワイエットゾーン(白枠)のモジュール数


def find_font(candidates, size):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def gen_qr(payload, size_px):
    """QRを size_px 角に生成。返り値 (img, sym_off, sym_px)。
    sym_off=画像TL→シンボルTLのpx, sym_px=シンボル一辺px(=アプリの1単位)。"""
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=BORDER)
    qr.add_data(payload)
    qr.make(fit=True)
    sym_modules = len(qr.modules)
    img_modules = sym_modules + 2 * BORDER
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB").resize((size_px, size_px), Image.NEAREST)
    module_px = size_px / img_modules
    return img, BORDER * module_px, sym_modules * module_px


def fit_font(text, max_w, max_h, candidates, pad=0.14):
    aw, ah = max_w * (1 - 2 * pad), max_h * (1 - 2 * pad)
    size = max(8, int(ah))
    while size > 7:
        font = find_font(candidates, size)
        l, t, r, b = font.getbbox(text)
        if (r - l) <= aw and (b - t) <= ah:
            return font, (l, t, r, b)
        size -= 2
    font = find_font(candidates, 8)
    return font, font.getbbox(text)


def make_form(out_path, name, lang, x, y, w, h, text, qr_px=300, margin=40):
    payload = f"CQR1,{name},{lang},{x},{y},{w},{h}"
    img, sym_off, sym_px = gen_qr(payload, qr_px)
    qx = qy = margin
    sx0, sy0 = qx + sym_off, qy + sym_off          # シンボルTL = unit原点
    fx, fy = sx0 + x * sym_px, sy0 + y * sym_px
    fw, fh = w * sym_px, h * sym_px

    W = int(max(qx + qr_px, fx + fw) + margin)
    H = int(max(qy + qr_px, fy + fh) + margin)
    canvas = Image.new("RGB", (W, H), "white")
    canvas.paste(img, (qx, qy))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(150, 180, 220), width=2)
    cands = JA_FONTS if lang.startswith("ja") else EN_FONTS
    font, (bl, bt, br, bb) = fit_font(text, fw, fh, cands)
    draw.text((fx + fw * 0.14 - bl, fy + (fh - (bb - bt)) / 2 - bt), text, fill="black", font=font)
    canvas.save(out_path)
    print(f"wrote {out_path}  (payload={payload!r})")


if __name__ == "__main__":
    make_form("form_en.png", "name", "en", 1.2, 0, 8, 1, "Yamada Taro")
    make_form("form_ja.png", "user_name", "ja_jp", 1.2, 0, 8, 1, "なかやま しんた")
