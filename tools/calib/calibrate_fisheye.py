#!/usr/bin/env python3
"""THINKLET カメラの魚眼キャリブレーション。

アプリのキャリブ撮影モードで撮った画像群（チェスボードを様々な角度・位置で写したもの）
からカメラ内部パラメータ K と魚眼歪み係数 D を推定し、calib.json に出力する。

OpenCV の魚眼モデル(equidistant)を使用:
  r = sqrt(a^2+b^2)  (歪み無し正規化座標), theta = atan(r)
  theta_d = theta*(1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)
  歪み正規化座標 = (theta_d/r)*(a,b),  画素 = K * それ
このモデル・係数は Android 実装(Fisheye.kt 相当)と一致する。

使い方:
  pip install opencv-python numpy
  python calibrate_fisheye.py --images ./shots --cols 9 --rows 6 --out calib.json
出力 calib.json を MainActivity.Calib に転記し Calib.enabled=true にする。

依存: pip install opencv-python numpy
"""
import argparse
import glob
import json
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="撮影画像フォルダ")
    ap.add_argument("--cols", type=int, default=9, help="内側コーナー数(横) = squares_x - 1")
    ap.add_argument("--rows", type=int, default=6, help="内側コーナー数(縦) = squares_y - 1")
    ap.add_argument("--out", default="calib.json")
    ap.add_argument("--preview", default="undistort_preview.jpg")
    a = ap.parse_args()

    pattern = (a.cols, a.rows)
    # 物体座標（スケールは任意。内部パラメータには影響しない）
    objp = np.zeros((1, pattern[0] * pattern[1], 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:pattern[0], 0:pattern[1]].T.reshape(-1, 2)

    files = sorted(sum([glob.glob(os.path.join(a.images, e))
                        for e in ("*.jpg", "*.jpeg", "*.png")], []))
    if not files:
        raise SystemExit(f"no images in {a.images}")

    objpoints, imgpoints, used = [], [], []
    img_size = None
    sub = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    find_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if img_size is None:
            img_size = gray.shape[::-1]  # (w,h)
        elif gray.shape[::-1] != img_size:
            print(f"  skip {os.path.basename(f)}: size mismatch {gray.shape[::-1]} != {img_size}")
            continue
        ok, corners = cv2.findChessboardCorners(gray, pattern, find_flags)
        if not ok:
            print(f"  no board: {os.path.basename(f)}")
            continue
        corners = cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), sub)
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(1, -1, 2))
        used.append(f)
        print(f"  OK: {os.path.basename(f)}")

    n = len(objpoints)
    print(f"\nboards detected: {n}/{len(files)}  image_size={img_size}")
    if n < 6:
        raise SystemExit("検出枚数が少なすぎます（最低6, 推奨15枚以上）。撮り直してください。")

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    flags = (cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC +
             cv2.fisheye.CALIB_FIX_SKEW)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-6)

    # 悪条件画像は CALIB_CHECK_COND で弾かれるため、該当indexを除外して再試行
    while True:
        rvecs = [np.zeros((1, 1, 3)) for _ in objpoints]
        tvecs = [np.zeros((1, 1, 3)) for _ in objpoints]
        try:
            rms, _, _, _, _ = cv2.fisheye.calibrate(
                objpoints, imgpoints, img_size, K, D, rvecs, tvecs,
                flags + cv2.fisheye.CALIB_CHECK_COND, crit)
            break
        except cv2.error as e:
            msg = str(e)
            idx = None
            if "CALIB_CHECK_COND" in msg and "input array " in msg:
                try:
                    idx = int(msg.split("input array ")[1].split(" ")[0])
                except Exception:
                    idx = None
            if idx is None or idx >= len(objpoints):
                # CHECK_COND 無しで最後の試行
                rvecs = [np.zeros((1, 1, 3)) for _ in objpoints]
                tvecs = [np.zeros((1, 1, 3)) for _ in objpoints]
                rms, _, _, _, _ = cv2.fisheye.calibrate(
                    objpoints, imgpoints, img_size, K, D, rvecs, tvecs, flags, crit)
                break
            print(f"  drop ill-conditioned: {os.path.basename(used[idx])}")
            del objpoints[idx], imgpoints[idx], used[idx]
            if len(objpoints) < 6:
                raise SystemExit("有効画像が減りすぎました。撮り直してください。")

    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, k3, k4 = [float(D[i, 0]) for i in range(4)]
    print(f"\nRMS reprojection error: {rms:.4f} px  (1.0未満が目安)")
    print(f"K: fx={fx:.3f} fy={fy:.3f} cx={cx:.3f} cy={cy:.3f}")
    print(f"D: k1={k1:.6f} k2={k2:.6f} k3={k3:.6f} k4={k4:.6f}")

    out = {
        "model": "opencv_fisheye",
        "image_size": [int(img_size[0]), int(img_size[1])],
        "fx": float(fx), "fy": float(fy), "cx": float(cx), "cy": float(cy),
        "k1": k1, "k2": k2, "k3": k3, "k4": k4,
        "rms_px": float(rms), "n_images": len(objpoints),
    }
    with open(a.out, "w", encoding="utf-8") as fp:
        json.dump(out, fp, indent=2)
    print(f"wrote {a.out}")

    # 検証プレビュー（歪み補正の見た目確認）
    sample = cv2.imread(used[0])
    newK = K.copy()
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), newK, img_size, cv2.CV_16SC2)
    und = cv2.remap(sample, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(a.preview, np.hstack([sample, und]))
    print(f"wrote {a.preview} (左:歪みあり 右:補正後)")


if __name__ == "__main__":
    main()
