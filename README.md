# Qrop QR

Qrop QRは、Connected Worker時代における、新しいOCRソリューションです。

ウェアラブルデバイスを装着した労働者が書類を眺めるだけで、自動的に適切な情報がOCRされ、データベースに格納されていくという未来を目指しています。

## 概要

[コンセプト資料(Google Slide)](https://docs.google.com/presentation/d/1UusCIhZIF972x3nY2aXQVvD0EoTUW-_OT408t3NrU2s/edit?slide%3Did.p#slide%3Did.p)

![概要資料](docs/image.png)

Qrop QRは、**QRコードに含まれた情報からOCRすべき相対座標を読み取り**、その範囲の領域を透視変換で切り出してOCRを行うアプリケーションです。

書類にあらかじめ「OCR対象フィールドの位置・名前・言語」を記述したQRコードを貼っておけば、装着者がその書類を見るだけで、フィールドが切り出され、OCRされ、（任意で）読み上げ・保存されます。

## 実装

本リポジトリは、Fairy Devices社のウェアラブル端末 **THINKLET (LC01)** 向けの **Kotlin / Android ネイティブ実装** です。

THINKLETは **Google Play Services を持たない** AOSP/Fairy OS 端末のため、QR検出・OCR・TTSはすべて **端末内（オフライン）** で完結させています。

### 機能

- **QR検出 → 相対座標で領域切り出し → OCR**
  - QRコードのサイズを1単位とした相対座標 (`x, y, w, h`) でフィールドを定義（[docs/SPEC.md](docs/SPEC.md)）
  - QRの4隅から `Matrix.setPolyToPoly` で透視変換し、フィールド領域だけを正対画像に切り出してOCR
  - **回転不変**：ファインダパターンでQRシンボルの上方向を判定し、端末/QRを90°・180°回しても
    フィールド位置が破綻しない（ML Kit cornerPoints は画像基準のため、軽量なピクセル判定で補正）
- **オフライン OCR**：ML Kit（バンドル版）Text Recognition v2（Latin + Japanese）
- **オフライン QR検出**：ML Kit（バンドル版）Barcode Scanning
- **読み上げ (TTS)**：Fairy Josee（`ai.fd.josee.app.tts`、日英オフライン）。未導入時はTTS無効で動作継続
- **モーションブラー対策**：Camera2 manual sensor によるアダプティブ高速シャッター
  - 露光時間の上限を **1/62s (`MAX_EXP_NS`)** に固定し、明るさを測りながらブラーを抑えつつ露出を自動調整（QRはブラーに弱いため）
- **3分割UI**：上＝カメラLive＋認識枠（QR=緑 / フィールド=シアン）、中＝切り出した透視変換画像、下＝OCR結果文字列

### 動作環境 / ビルド

- 端末: THINKLET LC01（Android 8.1 / API 27, arm64-v8a, GMSなし, 広角カメラ）
- ツールチェイン: AGP 8.10 / Kotlin 2.1 / compileSdk 36 / minSdk 27 / Gradle 8.13 / JDK 21 (JBR)
- ABI: `arm64-v8a` のみ

```bash
# ビルド & インストール（adb が通る状態で）
./gradlew :app:assembleDebug
adb install -r -g app/build/outputs/apk/debug/app-debug.apk
```

### TTS（Josee）について

THINKLET内蔵の Pico TTS (`com.svox.pico`) は `synthesizeText` でネイティブクラッシュするため使えません。
代わりに Fairy製 **Josee**（OpenJTalk[日] + Flite[英]）を利用します。Joseeはリリース配布が無く**ソースからビルド**が必要です（[FairyDevicesRD/droid.josee.tts](https://github.com/FairyDevicesRD/droid.josee.tts), NDK 26.3）。

```bash
adb install -r josee-tts-*.apk
adb shell settings put secure tts_default_synth ai.fd.josee.app.tts
```

### テスト用フォーム

[testdata/](testdata/) にサンプルフォーム（QR＋OCR対象テキスト）があります。

- `form_en.png` … `CQR1,name,en,1.2,0,8,1` ＋ "Yamada Taro"
- `form_ja.png` … `CQR1,user_name,ja_jp,1.2,0,8,1` ＋ "なかやま しんた"

`testdata/make_form.py` で再生成できます（`pip install qrcode pillow`）。

## 特許
- 本実装は、富士通が保有していた特許第４３９８４７４号に酷似しています
  - https://www.j-platpat.inpit.go.jp/c1801/PU/JP-4398474/15/ja
  - しかし、当該特許は失効しているため、特許権の侵害はないと考えています

## 履歴

過去の試作（Pythonプロトタイプ / JS・C# 各版）は本リポジトリから整理・削除しました。必要な場合は Git 履歴を参照してください（`Initial commit` 以前のコミット）。
