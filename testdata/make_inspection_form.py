#!/usr/bin/env python3
"""Qrop QR 手書きデモ用「設備点検記録票」生成（A4縦 @300dpi, 1QR=1記入欄）。

印刷して現場で手書き記入してもらい、THINKLET で見て認識させるデモ用の帳票。

  form_inspection.pdf         … 印刷用（A4。拡大縮小印刷でもOK＝座標はQR相対のため）
  form_inspection.png         … 同じ内容のPNG
  form_inspection_sample.png  … 記入例（手書き風フォントで記入済み）。リハーサル・画面表示用

手書き認識のための配慮:
  - 項目名・単位は記入枠の外（QRの左）に印刷し、OCR対象に入れない
  - 記入枠は薄い水色（ドロップアウトカラー寄り）。OCR切り出しは枠線の 1/16単位 内側
  - 数字/英字の欄は en（Latin OCR）、氏名・備考は ja（日本語OCR）

座標はQRシンボル（白枠を除く）=1単位。CQR2 生成は make_multiform.py と共通。
依存: pip install qrcode pillow
"""
import random
import qrcode
from qrcode.util import QRData, MODE_8BIT_BYTE
from PIL import Image, ImageDraw

from make_multiform import cqr2, find_font, JA_FONTS, EN_FONTS, BORDER

DPI = 300
W, H = 2480, 3508           # A4 縦 @300dpi
MODULE_PX = 10              # 1モジュール=10px≈0.85mm（整数pxで歪みなく描画）
QR_VERSION = 2
ROW_TOP, ROW_PITCH = 470, 400
LABEL_X, QR_X = 150, 560

# 記入枠（QR単位, 1/16の倍数＝Q8.4で誤差なし）。OCR切り出しは枠の 1/16 内側。
BOX_X, BOX_Y, BOX_W, BOX_H = 20 / 16, 0.0, 88 / 16, 1.0
INSET = 1 / 16
FIELD = (BOX_X + INSET, BOX_Y + INSET, BOX_W - 2 * INSET, BOX_H - 2 * INSET)

BOX_COLOR = (150, 190, 235)
INK = (25, 30, 75)          # 記入例のインク色（青黒）
HAND_FONTS = ["C:/Windows/Fonts/UDDigiKyokashoN-R.ttc"] + JA_FONTS

# (id, name, lang, 項目名, 英字/単位の補足, 記入例)
FIELDS = [
    (1, "date",      "en",    "点検日",     "Date (YYYY/MM/DD)", "2026/10/02"),
    (2, "equip_id",  "en",    "設備番号",   "Equipment ID",      "P-1024"),
    (3, "inspector", "ja_jp", "点検者",     "Inspector",         "山田 太郎"),
    (4, "pressure",  "en",    "吐出圧力",   "Pressure [MPa]",    "0.85"),
    (5, "temp",      "en",    "軸受温度",   "Temp [°C]",         "42.5"),
    (6, "result",    "en",    "判定",       "OK / NG",           "OK"),
    (7, "note",      "ja_jp", "備考",       "Note",              "異音なし"),
]


def gen_qr(payload):
    """QRを整数モジュールpxでそのまま生成。返り値 (img, sym_off, sym_px)。"""
    # 全行で version 2（25×25）に固定＝QRと記入枠の大きさを揃える（payload≤26B: name≤16字）
    qr = qrcode.QRCode(version=QR_VERSION, error_correction=qrcode.constants.ERROR_CORRECT_M,
                       box_size=MODULE_PX, border=BORDER)
    qr.add_data(QRData(bytes(payload), mode=MODE_8BIT_BYTE, check_data=False))
    qr.make(fit=False)
    img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    return img, BORDER * MODULE_PX, len(qr.modules) * MODULE_PX


def unit_rect(sx0, sy0, sym_px, x, y, w, h):
    return (sx0 + x * sym_px, sy0 + y * sym_px, sx0 + (x + w) * sym_px, sy0 + (y + h) * sym_px)


def draw_form(sample=False):
    page = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(page)
    rnd = random.Random(7)

    d.text((LABEL_X, 150), "設備点検記録票", fill="black", font=find_font(JA_FONTS, 96))
    d.text((LABEL_X + 760, 190), "Equipment Inspection Record  —  Qrop QR demo",
           fill=(110, 110, 110), font=find_font(EN_FONTS, 40))
    d.text((LABEL_X, 320), "水色の枠の中に、1行で・大きく・はっきり記入してください（枠の外には書かないでください）",
           fill=(60, 60, 60), font=find_font(JA_FONTS, 40))

    label_font = find_font(JA_FONTS, 64)
    sub_font = find_font(EN_FONTS, 34)
    rows = []
    for i, (fid, name, lang, label, sub, example) in enumerate(FIELDS):
        top = ROW_TOP + i * ROW_PITCH
        payload = cqr2(name, lang, *FIELD, id=fid)
        qimg, sym_off, sym_px = gen_qr(payload)
        page.paste(qimg, (QR_X, top))
        sx0, sy0 = QR_X + sym_off, top + sym_off

        # 項目名（QRの左, シンボル縦中央）
        cy = sy0 + sym_px / 2
        d.text((LABEL_X, cy - 72), label, fill="black", font=label_font)
        d.text((LABEL_X, cy + 14), sub, fill=(100, 100, 100), font=sub_font)

        box = unit_rect(sx0, sy0, sym_px, BOX_X, BOX_Y, BOX_W, BOX_H)
        d.rectangle(box, outline=BOX_COLOR, width=4)

        if sample:
            font = find_font(HAND_FONTS, int(sym_px * 0.55))
            l, t, r, b = font.getbbox(example)
            tx = box[0] + 40 + rnd.uniform(0, 60) - l
            ty = box[1] + (box[3] - box[1] - (b - t)) / 2 - t + rnd.uniform(-10, 10)
            txt = Image.new("RGBA", (r - l + 40, b - t + 40), (0, 0, 0, 0))
            ImageDraw.Draw(txt).text((20 - l, 20 - t), example, fill=INK, font=font)
            txt = txt.rotate(rnd.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True)
            page.paste(txt, (int(tx + l - 20), int(ty + t - 20)), txt)

        rows.append((fid, name, lang, len(payload), sym_px, payload.hex()))

    foot = find_font(JA_FONTS, 34)
    d.text((LABEL_X, H - 260), "※ QRコードは折り曲げ・汚損しないでください。拡大縮小印刷でも読み取れます（座標はQR基準の相対値）。",
           fill=(110, 110, 110), font=foot)
    d.text((LABEL_X, H - 205), "※ 各QRには、右隣の記入欄の位置・項目名・言語（CQR2形式）が埋め込まれています。",
           fill=(110, 110, 110), font=foot)
    return page, rows


def main():
    page, rows = draw_form(sample=False)
    page.save("form_inspection.png", dpi=(DPI, DPI))
    Image.init()  # PDF内部のJPEGエンコーダ登録（未初期化だと KeyError: 'JPEG'）
    page.save("form_inspection.pdf", resolution=DPI, quality=95)
    draw_form(sample=True)[0].save("form_inspection_sample.png", dpi=(DPI, DPI))
    print(f"{'id':<3}{'name':<11}{'lang':<7}{'bytes':>5} {'sym_px':>6}  hex")
    for fid, name, lang, n, sym_px, hx in rows:
        print(f"{fid:<3}{name:<11}{lang:<7}{n:>4}B {sym_px:>6}  {hx}")
    print("wrote form_inspection.pdf / .png / _sample.png")


if __name__ == "__main__":
    main()
