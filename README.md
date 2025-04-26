# Qrop QR

Qrop QRは、Connected Worker時代における、新しいOCRソリューションです。

ウェアラブルデバイスを装着した労働者が書類を眺めるだけで、自動的に適切な情報がOCRされ、データベースに格納されていくという未来を目指しています。 

## 概要

[コンセプト資料(Google Slide)](https://docs.google.com/presentation/d/1UusCIhZIF972x3nY2aXQVvD0EoTUW-_OT408t3NrU2s/edit?slide%3Did.p#slide%3Did.p)

![概要資料](docs/image.png)

Qrop QRは、QRコードに含まれた情報から、OCRするべき相対座標を読み取り、その範囲の領域を切り出してOCRを行うアプリケーションです。

## フォルダ構造

- docs: ドキュメント
- 01_python_proto: Pythonによるプロトタイプ版
- 02_js_demo: JavaScriptによるWeb版のプロトタイプ版
- 03_csharp_windows: C#によるWindowsアプリ版
- 04_csharp_Android: C#によるAndroidアプリ版
- 05_csharp_thinklet: Android版をThinkletに移植したもの

## 特許
- 本実装は、富士通が保有していた特許第４３９８４７４号に酷似しています
  - https://www.j-platpat.inpit.go.jp/c1801/PU/JP-4398474/15/ja
  - しかし、当該特許は失効しているため、特許権の侵害はないと考えています


