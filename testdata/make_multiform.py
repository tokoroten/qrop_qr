#!/usr/bin/env python3
"""Qrop QR マルチ読み取り（複数QR＝1QR1フィールド）テスト帳票生成。

1枚の画像に、様々な大きさ・様々なオフセット(右/下/左/上, 負値含む)の
QR＋読み取り領域を並べ、各領域に適当なダミー文字列を配置する。

各QRは CQR1 形式 `CQR1,<name>,<lang>,<x>,<y>,<w>,<h>` を埋め込む。
座標はQRサイズ=1単位の相対値で、フィールドのピクセル矩形は
  left = qr_x + x*S, top = qr_y + y*S, width = w*S, height = h*S   (S=QR一辺px)
としてアプリの透視変換と一致させる。読み取り対象テキストはこの矩形内に描く。

依存: pip install qrcode pillow
"""
import qrcode
from PIL import Image, ImageDraw, ImageFont

JA_FONTS = [
    "C:/Windows/Fonts/meiryo.ttc", "C:/Windows/Fonts/YuGothM.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
]
EN_FONTS = [
    "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def find_font(candidates, size):
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


BORDER = 4  # クワイエットゾーン(白枠)のモジュール数


def gen_qr(payload, size_px):
    """QRを生成し size_px 角に最近傍リサイズ。

    返り値: (img, sym_off, sym_px)
      sym_off … 画像TLから「シンボルTL(=黒モジュール左上)」までのpxオフセット
      sym_px  … シンボル一辺のpx ＝ アプリの「1単位」(ML Kit cornerPoints基準・白枠を含まない)
    アプリは cornerPoints=シンボル四隅を単位正方形にするので、フィールドは白枠を除いた
    シンボル基準で配置しないと、白枠ぶんだけ原点・スケールがズレる。
    """
    qr = qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L, box_size=10, border=BORDER)
    qr.add_data(payload)
    qr.make(fit=True)
    sym_modules = len(qr.modules)             # シンボルのモジュール数（白枠を除く）
    img_modules = sym_modules + 2 * BORDER    # 画像のモジュール数（白枠を含む）
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB").resize((size_px, size_px), Image.NEAREST)
    module_px = size_px / img_modules
    return img, BORDER * module_px, sym_modules * module_px


def fit_font(text, max_w, max_h, candidates, pad=0.14):
    """矩形(max_w×max_h)にpad込みで収まる最大フォントを総当りで探す。"""
    avail_w = max_w * (1 - 2 * pad)
    avail_h = max_h * (1 - 2 * pad)
    size = max(8, int(avail_h))
    while size > 7:
        font = find_font(candidates, size)
        l, t, r, b = font.getbbox(text)
        if (r - l) <= avail_w and (b - t) <= avail_h:
            return font, (l, t, r, b)
        size -= 2
    font = find_font(candidates, 8)
    return font, font.getbbox(text)


# (qx, qy, S=QR一辺px, name, lang, x, y, w, h, text)  ※x,y,w,h はQRサイズ単位の相対値
FIELDS = [
    dict(qx=90,  qy=175,  S=120, name="invoice_no", lang="en",    x=1.3,  y=0.05,  w=6.0, h=0.8, text="INV-2026-0042"),
    dict(qx=90,  qy=370,  S=150, name="user_name",  lang="ja_jp", x=0.0,  y=1.25,  w=6.5, h=0.9, text="なかやま しんた"),
    dict(qx=95,  qy=735,  S=205, name="address_en", lang="en",    x=1.2,  y=0.20,  w=4.3, h=1.2, text="1600 Amphitheatre Pkwy"),
    dict(qx=905, qy=1190, S=130, name="issue_date", lang="en",    x=-5.6, y=-0.95, w=6.0, h=0.7, text="2026-06-04"),
    dict(qx=985, qy=1335, S=165, name="address_ja", lang="ja_jp", x=-5.4, y=0.10,  w=5.0, h=0.9, text="東京都千代田区一番町"),
    dict(qx=90,  qy=1565, S=115, name="serial_no",  lang="en",    x=1.4,  y=0.00,  w=5.2, h=0.7, text="A-7F3K-99Z"),
]

W, H = 1240, 1754  # A4 縦 @150dpi 相当


def main(out_path="form_multi.png"):
    canvas = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(canvas)

    title = find_font(EN_FONTS, 34)
    draw.text((90, 70), "Qrop QR  multi-field sample", fill="black", font=title)
    sub = find_font(JA_FONTS, 22)
    draw.text((90, 116), "複数QR（1QR=1フィールド）／様々な大きさ・オフセットの読み取りサンプル", fill=(90, 90, 90), font=sub)

    cap_font = find_font(EN_FONTS, 15)
    print(f"{'name':<12} {'lang':<6} {'payload':<44} field_px(L,T,W,H)")
    for f in FIELDS:
        S = f["S"]
        payload = f"CQR1,{f['name']},{f['lang']},{f['x']},{f['y']},{f['w']},{f['h']}"
        img, sym_off, sym_px = gen_qr(payload, S)
        canvas.paste(img, (f["qx"], f["qy"]))

        # フィールド矩形は「シンボルTL＋シンボル単位」で計算（アプリの cornerPoints 基準と一致）
        sx0 = f["qx"] + sym_off   # シンボルTL = アプリのunit原点(0,0)
        sy0 = f["qy"] + sym_off
        fx = sx0 + f["x"] * sym_px
        fy = sy0 + f["y"] * sym_px
        fw = f["w"] * sym_px
        fh = f["h"] * sym_px

        # 読み取り領域の枠（薄い水色）＋ダミー文字列
        draw.rectangle([fx, fy, fx + fw, fy + fh], outline=(150, 180, 220), width=2)
        cands = JA_FONTS if f["lang"].startswith("ja") else EN_FONTS
        font, (bl, bt, br, bb) = fit_font(f["text"], fw, fh, cands)
        tw, th = br - bl, bb - bt
        tx = fx + fw * 0.14 - bl
        ty = fy + (fh - th) / 2 - bt
        draw.text((tx, ty), f["text"], fill="black", font=font)

        # QR直上に小さな注記（QRは読み取り対象ではないので干渉しない）
        draw.text((f["qx"], f["qy"] - 20), f"{f['name']} [{f['lang']}]", fill=(120, 120, 120), font=cap_font)

        print(f"{f['name']:<12} {f['lang']:<6} {payload:<44} ({fx:.0f},{fy:.0f},{fw:.0f},{fh:.0f})")

    canvas.save(out_path)
    print(f"\nwrote {out_path}  ({W}x{H}, {len(FIELDS)} fields)")


if __name__ == "__main__":
    main()
