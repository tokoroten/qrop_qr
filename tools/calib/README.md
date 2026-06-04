# カメラキャリブレーション（魚眼歪み補正）

THINKLET の広角カメラは魚眼気味で、画面端ほど直線が湾曲します。これにより QR の隅から
フィールド領域を外挿（透視変換）すると、QRから離れた位置ほどズレます。これを補正するため、
カメラの内部パラメータ `K` と魚眼歪み係数 `D` を一度だけ推定し、アプリに焼き込みます。

## 準備

```bash
pip install opencv-python numpy pillow
```

## 手順

### 1. チェスボードを用意

```bash
python gen_chessboard.py            # chessboard_10x7.png（内側コーナー 9x6）
```

**印刷は必須ではありません。** 内部パラメータの推定にマスの物理サイズは無関係なので、
フラットな液晶（モニタ／タブレット）に表示してもOK。ただし:

- マスが**正確に正方形**に表示されること（ビューアの拡大で縦横比を崩さない／引き伸ばさない）
- **グレア・映り込みを避け、画面輝度は下げる**（白飛びすると角検出が落ちる）
- 紙に貼る場合は**平らな板**に貼る（波打ち厳禁）

### 2. アプリのキャリブ撮影モードで撮影

アプリ起動中に **物理ボタンの「音量↑＋音量↓ 同時押し」** でキャリブ撮影モードに入る（再ビルド不要）。
TTSで「キャリブレーションモードを開始します」と案内され、一定間隔（既定 ~1.2s）で解析フレームを自動保存。
**30枚で自動終了**（TTSで「撮影完了」）。途中で止めたい時は再度「音量↑＋↓同時押し」。

- チェスボードを**様々な角度・距離・位置**で見せる（正面／左右に傾ける／上下に傾ける）
- 特に**フレームの四隅**にもボードが来るように（魚眼歪みは端に強く出る＝端のデータが命）
- 推奨 **30枚**（自動終了枚数。`CALIB_MAX` で調整可）

撮影画像は端末内 `Android/data/com.example.qropqr/files/calib/` に溜まる。PCへ取得:

```bash
adb pull /sdcard/Android/data/com.example.qropqr/files/calib ./shots
```

### 3. キャリブレーション実行

```bash
python calibrate_fisheye.py --images ./shots --cols 9 --rows 6 --out calib.json
```

- `--cols/--rows` は**内側コーナー数**（= マス数 - 1）。`gen_chessboard.py` の `--squares-x/y` と合わせる
- `RMS reprojection error` が **1.0px 未満**なら良好。大きい場合は撮り直し
- `undistort_preview.jpg`（左:歪みあり / 右:補正後）で直線が真っ直ぐになっているか目視確認

### 4. アプリへ焼き込み

`calib.json` の値を `Fisheye.kt` の `Calib` に転記し、`Calib.enabled = true` にして再ビルド。
`image_size` は撮影時の解析解像度。アプリは実行時のフレーム解像度に合わせて `fx,fy,cx,cy` を
線形スケールするので、解像度が違っても可。

## モデル

OpenCV の魚眼(equidistant)モデル。アプリ側 `Fisheye` 実装と一致:

```
a,b: 歪み無し正規化座標   r = sqrt(a^2+b^2),  theta = atan(r)
theta_d = theta*(1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
歪み正規化 = (theta_d/r)*(a,b)      画素 = K * 歪み正規化
```
