# カメラキャリブレーション（魚眼歪み補正）

THINKLET の広角カメラは魚眼気味で、画面端ほど直線が湾曲します。これにより QR の隅から
フィールド領域を外挿（透視変換）すると、QRから離れた位置ほどズレます。これを補正するため、
カメラの内部パラメータ `K` と魚眼歪み係数 `D` を推定します。

**推定は端末内で完結します**（OpenCV同梱）。PC不要。以下は端末内キャリブの手順と、参考用の
オフライン手順です。

## 1. チェスボードを用意

```bash
python gen_chessboard.py            # chessboard_10x7.png（内側コーナー 9x6）
```

**印刷は必須ではありません。** 内部パラメータの推定にマスの物理サイズは無関係なので、
フラットな液晶（モニタ／タブレット）に表示してもOK。ただし:

- マスが**正確に正方形**に表示されること（ビューアの拡大で縦横比を崩さない／引き伸ばさない）
- **グレア・映り込みを避け、画面輝度は下げる**（白飛びすると角検出が落ちる）
- 紙に貼る場合は**平らな板**に貼る（波打ち厳禁）

## 2. 端末内キャリブ（推奨・PC不要）

アプリ起動中に **物理ボタンの「音量↑＋音量↓ 同時押し」** でキャリブモードに入る（再ビルド不要）。

1. TTS「キャリブレーションモードを開始します」。
2. チェスボードを **様々な角度・距離・位置**で見せる（正面／左右・上下に傾ける）。
   特に **フレームの四隅**にもボードが来るように（魚眼歪みは端に強く出る＝端のデータが命）。
   コーナー検出に成功するたび TTS「視点 N」。
3. **15視点**（`CAL_VIEWS`）集まると端末内で `fisheye.calibrate` を実行 →
   TTS「キャリブレーション完了。誤差 x.x ピクセル」。
4. 結果は `filesDir/calib.json` に永続化され、**再起動後も自動ロード**して以後の補正に反映。

途中で抜けたい時は再度「音量↑＋↓同時押し」。

> 内部実装: `MainActivity` が `findChessboardCornersSB(9x6)` でコーナーを蓄積し、
> `Calib3d.fisheye_calibrate` で K,D を推定→`Fisheye.kt` の `Calib` を実行時更新＋永続化。

## （参考）オフラインで推定し既定値を焼き込む

端末内キャリブの代わりに、PCで推定して `Calib` の**既定値**を更新することもできます（検証用途や、
出荷時の焼き込み値を作る場合）。

```bash
pip install opencv-python numpy pillow
# 端末から撮影フレームを取得（撮影モードでフレーム保存する旧ビルドが必要）
adb pull /sdcard/Android/data/com.example.qropqr/files/calib ./shots
python calibrate_fisheye.py --images ./shots --cols 9 --rows 6 --out calib.json
```

- `--cols/--rows` は**内側コーナー数**（= マス数 - 1）。`gen_chessboard.py` の `--squares-x/y` と合わせる
- `RMS reprojection error` が **1.0px 未満**なら良好。`undistort_preview.jpg` で直線化を目視確認
- `calib.json` の値を `Fisheye.kt` の `Calib` 既定値に転記して再ビルド

## モデル

OpenCV の魚眼(equidistant)モデル。アプリ側 `Fisheye` 実装と一致:

```
a,b: 歪み無し正規化座標   r = sqrt(a^2+b^2),  theta = atan(r)
theta_d = theta*(1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
歪み正規化 = (theta_d/r)*(a,b)      画素 = K * 歪み正規化
```
