"""
# このコメントはAI向けの設計資料です。

- まず、jupyter_experimental_code/qr_detect_opencv.ipynb を読み込み、以後の設計の参考にせよ

## 画面設計
- ウィンドウは、左側と右側に分け、比率は1:1とする
- 左側にカメラから撮影された画像＋QRコードの四隅に従って枠線を表示
- 右側には、QRコードの座標を元に、入力画像を透視変換した画像を表示
  - QRコードが検出できない場合は、直近に検出したQRコードの座標を元に透視変換画像を表示する
  - 複数のQRコードが検出された場合は、最後に検出したQRコードの内容と等しいQRコードを利用する
- ウィンドウはリアルタイムに更新し続ける
"""

import cv2
import numpy as np
import time
import os
import json

class QRPerspectiveDemo:
    def __init__(self, camera_id=0, window_name="QR Perspective Demo"):
        """
        QRコード検出と透視変換のデモアプリケーションを初期化
        
        Args:
            camera_id: カメラデバイスのID（デフォルト: 0）
            window_name: 表示ウィンドウの名前
        """
        self.cap = cv2.VideoCapture(camera_id)
        self.window_name = window_name
        self.detector = cv2.QRCodeDetector()
        self.camera_id = camera_id
        
        # カメラの最大解像度を設定
        self.set_max_resolution()
        
        # 直近で検出したQRコードの情報を保持
        self.last_qr_points = None
        self.last_qr_info = None
        self.last_detection_time = 0
        
    def set_max_resolution(self):
        """
        カメラの利用可能な解像度を取得し、最大解像度を設定。
        キャッシュを利用して起動を高速化する。
        """
        # キャッシュファイルのパスを設定
        cache_dir = os.path.join(os.path.expanduser("~"), ".qr_perspective_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cache_file = os.path.join(cache_dir, "camera_resolutions.json")
        
        # カメラ情報を取得
        camera_name = self.get_camera_name()
        camera_key = f"{camera_name}_{self.camera_id}"
        
        # キャッシュを読み込み
        cached_resolutions = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    cached_resolutions = json.load(f)
                print("カメラ解像度のキャッシュを読み込みました")
            except (json.JSONDecodeError, IOError) as e:
                print(f"キャッシュの読み込みに失敗しました: {e}")
        
        # キャッシュに該当カメラの情報があれば使用
        if camera_key in cached_resolutions:
            max_width, max_height = cached_resolutions[camera_key]
            print(f"キャッシュから最大解像度を設定: {max_width}x{max_height}")
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_height)
            
            # 設定された解像度を確認
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            # キャッシュと実際の解像度が一致するか確認
            if abs(actual_width - max_width) < 1 and abs(actual_height - max_height) < 1:
                return
            print(f"キャッシュの解像度と実際の解像度が一致しません。再スキャンします。")
        
        # 一般的な解像度のリスト（高解像度から低解像度の順）
        resolutions = [
            (3840, 2160),  # 4K
            (2560, 1440),  # 2K
            (1920, 1080),  # Full HD
            (1280, 720),   # HD
            (640, 480),    # VGA
            (320, 240)     # QVGA
        ]
        
        max_width = 0
        max_height = 0
        max_resolution = None
        
        print(f"カメラ '{camera_key}' の最大解像度をスキャンしています...")
        
        # 各解像度でカメラが対応しているか確認
        for width, height in resolutions:
            # 幅と高さを設定
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # 実際に設定された解像度を取得
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            
            print(f"試行解像度: {width}x{height}, 実際の解像度: {actual_width}x{actual_height}")
            
            # 最大解像度を更新
            if actual_width * actual_height > max_width * max_height:
                max_width = actual_width
                max_height = actual_height
                max_resolution = (actual_width, actual_height)
        
        # 最大解像度を設定
        if max_resolution:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_resolution[1])
            print(f"カメラの最大解像度を設定: {max_resolution[0]}x{max_resolution[1]}")
            
            # キャッシュに保存
            cached_resolutions[camera_key] = max_resolution
            try:
                with open(cache_file, 'w') as f:
                    json.dump(cached_resolutions, f)
                print(f"最大解像度をキャッシュに保存しました: {max_resolution}")
            except IOError as e:
                print(f"キャッシュの保存に失敗しました: {e}")
    
    def get_camera_name(self):
        """
        接続されているカメラ名を取得
        """
        # カメラのプロパティを取得して識別子を作成
        try:
            # カメラの固有識別子を取得する試み
            # OpenCVはすべてのプロパティをサポートしているわけではないので複数試す
            backend_name = self.cap.getBackendName()
            guid = self.cap.get(cv2.CAP_PROP_GUID)
            
            # ハードウェア固有のプロパティを組み合わせて識別子を作成
            identifiers = []
            
            if backend_name:
                identifiers.append(f"backend:{backend_name}")
            
            if guid and guid != 0:
                identifiers.append(f"guid:{guid}")
                
            # 他にも利用可能なプロパティがあれば追加
            # 例: FourCC, カメラのシリアル番号など
            
            # 十分な情報が得られなかった場合は汎用的な識別子を使用
            if not identifiers:
                return f"generic_camera"
                
            return "_".join(identifiers)
        except Exception as e:
            print(f"カメラ名の取得中にエラーが発生しました: {e}")
            return f"unknown_camera"
        
    def detect_qr_codes(self, frame):
        """
        画像からQRコードを検出
        
        Args:
            frame: 入力画像（カメラフレーム）
            
        Returns:
            tuple: (検出成功フラグ, デコードされた情報リスト, QRコードの座標リスト)
        """
        retval, decoded_info, points, _ = self.detector.detectAndDecodeMulti(frame)
        return retval, decoded_info, points
    
    def draw_qr_boundaries(self, frame, qr_points, qr_info):
        """
        QRコードの境界線と情報を描画
        
        Args:
            frame: 描画対象の画像
            qr_points: QRコードの座標
            qr_info: QRコードの内容
            
        Returns:
            修正された画像
        """
        img_with_qr = frame.copy()
        
        if qr_points is not None:
            for i, (info, pts) in enumerate(zip(qr_info, qr_points)):
                # 座標を整数形式に変換
                pts = pts.astype(np.int32)
                
                # QRコードの境界線を描画
                cv2.polylines(img_with_qr, [pts], True, (0, 255, 0), 2)
                
                # デコードされた情報を追加
                x, y = pts[0][0], pts[0][1]
                cv2.putText(img_with_qr, f"QR: {info}", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return img_with_qr
    
    def apply_perspective_transform(self, frame, qr_points):
        """
        QRコードの座標に基づいて透視変換を適用
        
        Args:
            frame: 入力画像
            qr_points: QRコードの座標
            
        Returns:
            透視変換された画像
        """
        if qr_points is None:
            return frame.copy()
        
        # 入力点の形式を変換 (4, 1, 2) -> (4, 2)
        src_pts = qr_points.reshape(4, 2).astype(np.float32)
        
        # 元画像のサイズを取得
        frame_height, frame_width = frame.shape[:2]
        
        # QRコードの辺の長さを計算
        side_lengths = []
        for j in range(4):
            p1 = src_pts[j]
            p2 = src_pts[(j+1) % 4]
            side_lengths.append(np.sqrt(((p2[0] - p1[0]) ** 2) + ((p2[1] - p1[1]) ** 2)))
        
        avg_side_length = np.mean(side_lengths)
        
        # 出力画像の中央に配置するための調整
        center_output_x = frame_width // 2
        center_output_y = frame_height // 2
        
        # 理想的な正方形のQRコードの座標を計算
        half_side = avg_side_length
        dst_pts = np.array([
            [center_output_x - half_side, center_output_y - half_side],  # 左上
            [center_output_x + half_side, center_output_y - half_side],  # 右上
            [center_output_x + half_side, center_output_y + half_side],  # 右下
            [center_output_x - half_side, center_output_y + half_side]   # 左下
        ], dtype=np.float32)
        
        # 透視変換の行列を計算
        perspective_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        
        # 画像全体に透視変換を適用
        corrected_img = cv2.warpPerspective(frame, perspective_matrix, (frame_width, frame_height))
        
        # 変換後のQRコードの位置を描画（確認用）
        for j, point in enumerate(dst_pts):
            x, y = point
            cv2.circle(corrected_img, (int(x), int(y)), 5, (255, 0, 0), -1)
            next_point = dst_pts[(j+1) % 4]
            cv2.line(corrected_img, 
                     (int(x), int(y)), 
                     (int(next_point[0]), int(next_point[1])), 
                     (0, 255, 0), 2)
                
        return corrected_img
    
    def process_frame(self, frame):
        """
        フレームを処理し、QRコード検出と透視変換を適用
        
        Args:
            frame: 入力フレーム
            
        Returns:
            tuple: (QRコード検出結果画像, 透視変換画像)
        """
        # QRコード検出
        retval, decoded_info, points = self.detect_qr_codes(frame)
        
        selected_qr_points = None
        selected_qr_info = []
        
        if retval:
            self.last_detection_time = time.time()
            
            # 前回検出されたQRコード情報がある場合
            if self.last_qr_info is not None:
                # 前回と同じ内容のQRコードを探す
                for i, info in enumerate(decoded_info):
                    if info == self.last_qr_info:
                        selected_qr_points = points[i]
                        selected_qr_info = [info]
                        break
            
            # 前回と同じQRコードが見つからなかった場合は最後のQRコードを使用
            if selected_qr_points is None and len(points) > 0:
                selected_qr_points = points[-1]
                selected_qr_info = [decoded_info[-1]]
                self.last_qr_info = decoded_info[-1]
            
            self.last_qr_points = selected_qr_points
            
        else:
            # QRコードが検出されなかった場合、直近の座標を使用
            if self.last_qr_points is not None and time.time() - self.last_detection_time < 5:  # 5秒以内に検出されたもののみ使用
                selected_qr_points = self.last_qr_points
                selected_qr_info = [self.last_qr_info] if self.last_qr_info else [""]
        
        # QRコード境界線を描画
        qr_overlay = self.draw_qr_boundaries(frame, [selected_qr_points] if selected_qr_points is not None else None, selected_qr_info)
        
        # 透視変換を適用
        perspective_img = self.apply_perspective_transform(frame, selected_qr_points)
        
        return qr_overlay, perspective_img
    
    def run(self):
        """
        デモアプリケーションを実行
        """
        # ウィンドウのプロパティを設定（サイズ変更可能、アスペクト比固定）
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # 最初のフレームを取得してウィンドウサイズを設定
        ret, frame = self.cap.read()
        if not ret:
            print("Error: Failed to capture initial frame")
            return
            
        height, width = frame.shape[:2]
        combined_height = height * 2  # 上下に並べるので高さは2倍
        
        # 初期ウィンドウサイズを設定（画面に収まるように適切なサイズに調整）
        initial_width = min(1280, width)
        scale_factor = initial_width / width
        initial_height = int(combined_height * scale_factor)
        cv2.resizeWindow(self.window_name, initial_width, initial_height)
        
        # アスペクト比を計算
        aspect_ratio = width / combined_height
        
        # 初期ウィンドウサイズを保存
        self.last_resize_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # BGR -> RGB変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # フレーム処理
            qr_overlay, perspective_img = self.process_frame(frame_rgb)
            
            # RGB -> BGR変換（OpenCV表示用）
            qr_overlay = cv2.cvtColor(qr_overlay, cv2.COLOR_RGB2BGR)
            perspective_img = cv2.cvtColor(perspective_img, cv2.COLOR_RGB2BGR)
            
            # 上下に並べて表示するための結合画像を作成
            height, width, _ = frame.shape
            combined_img = np.zeros((height * 2, width, 3), dtype=np.uint8)
            combined_img[:height, :] = qr_overlay
            combined_img[height:, :] = perspective_img
            
            # 結合画像を表示
            cv2.imshow(self.window_name, combined_img)
            
            # アスペクト比を固定するために周期的にresizeWindowを呼び出す
            # ユーザーがウィンドウをリサイズした後、サイズを調整する
            current_time = time.time()
            if current_time - self.last_resize_time > 1.0:  # 1秒ごとにチェック
                self.last_resize_time = current_time
                
                try:
                    # 現在のウィンドウのサイズを取得
                    window_size = cv2.getWindowImageRect(self.window_name)
                    if window_size and len(window_size) == 4:
                        # getWindowImageRect returns [x, y, width, height]
                        _, _, current_width, current_height = window_size
                        
                        # アスペクト比が大きく変わっている場合は修正
                        current_ratio = current_width / current_height if current_height > 0 else 0
                        if abs(current_ratio - aspect_ratio) > 0.1:  # 許容誤差を広めに設定
                            # 高さに基づいて幅を調整
                            new_width = int(current_height * aspect_ratio)
                            cv2.resizeWindow(self.window_name, new_width, int(current_height))
                except Exception as e:
                    # 無視してプログラムを続行
                    pass
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # リソースの解放
        self.cap.release()
        cv2.destroyAllWindows()
    
    def __del__(self):
        """
        デストラクタ：リソースを解放
        """
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()


def main():
    """
    メイン関数
    """
    demo = QRPerspectiveDemo()
    demo.run()


if __name__ == "__main__":
    main()



