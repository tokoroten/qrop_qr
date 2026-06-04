package com.example.qropqr

import kotlin.math.atan
import kotlin.math.hypot
import kotlin.math.tan

/**
 * OpenCV fisheye(equidistant)モデルの歪み変換。tools/calib のキャリブ結果(K,D)を用いる。
 * 距離は不要：レンズ固有の「画素⇄画素」純幾何変換。
 *
 *   a,b: 歪み無し正規化   r=hypot(a,b)  th=atan(r)
 *   th_d = th*(1 + k1*th^2 + k2*th^4 + k3*th^6 + k4*th^8)
 *   歪みあり正規化 = (th_d/r)*(a,b)     画素 = K * 正規化
 */
class Fisheye(
    private val fx: Double, private val fy: Double,
    private val cx: Double, private val cy: Double,
    private val k1: Double, private val k2: Double, private val k3: Double, private val k4: Double,
) {
    /** 歪み無し正規化(a,b) → 歪みあり正規化(x',y')（前方向・閉形式）。 */
    private fun distortNorm(a: Double, b: Double): DoubleArray {
        val r = hypot(a, b)
        if (r < 1e-9) return doubleArrayOf(a, b)
        val th = atan(r)
        val t2 = th * th
        val thd = th * (1.0 + k1 * t2 + k2 * t2 * t2 + k3 * t2 * t2 * t2 + k4 * t2 * t2 * t2 * t2)
        val s = thd / r
        return doubleArrayOf(a * s, b * s)
    }

    /** 歪みあり画素(u,v) → 歪み無し正規化(a,b)（逆方向・Newton反復, OpenCV undistortPoints相当）。 */
    private fun undistortToNorm(u: Double, v: Double): DoubleArray {
        val xd = (u - cx) / fx
        val yd = (v - cy) / fy
        val thd = hypot(xd, yd)
        if (thd < 1e-9) return doubleArrayOf(xd, yd)
        var th = thd
        repeat(10) {
            val t2 = th * th
            val f = th * (1.0 + k1 * t2 + k2 * t2 * t2 + k3 * t2 * t2 * t2 + k4 * t2 * t2 * t2 * t2) - thd
            val fp = 1.0 + 3 * k1 * t2 + 5 * k2 * t2 * t2 + 7 * k3 * t2 * t2 * t2 + 9 * k4 * t2 * t2 * t2 * t2
            th -= f / fp
        }
        val scale = tan(th) / thd
        return doubleArrayOf(xd * scale, yd * scale)
    }

    /** 歪みあり画素 → 歪み補正後の画素（P=K）。QR隅を補正してホモグラフィを作るのに使う。 */
    fun undistortPixel(u: Double, v: Double): DoubleArray {
        val n = undistortToNorm(u, v)
        return doubleArrayOf(fx * n[0] + cx, fy * n[1] + cy)
    }

    /** 歪み補正後の画素(U,V) → 歪みあり画素(u,v)。クロップのサンプリング逆引き／オーバーレイ描画に使う。 */
    fun distortPixel(U: Double, V: Double): DoubleArray {
        val d = distortNorm((U - cx) / fx, (V - cy) / fy)
        return doubleArrayOf(fx * d[0] + cx, fy * d[1] + cy)
    }
}

/**
 * THINKLET LC01 のキャリブ結果（tools/calib/calibrate_fisheye.py の出力 calib.json）。
 * k1..k4 は無次元（角度の関数）なので解像度に依らず一定。fx,fy,cx,cy のみ実行時解像度へ線形スケール。
 */
object Calib {
    var enabled = true              // false にすると歪み補正を無効化（従来の純ホモグラフィ）
    // 既定値は実機キャリブ結果（焼き込み）。端末内キャリブ or 永続化ファイルがあれば上書きされる。
    var refW = 2048.0               // 推定時の解析解像度
    var refH = 1536.0
    var fx = 1080.6801066313897
    var fy = 1078.1127565569914
    var cx = 994.9940846792864
    var cy = 736.4125872987854
    var k1 = 0.35308315001651824
    var k2 = -0.26880705937503846
    var k3 = 0.1276706402798861
    var k4 = 0.023033044219524554

    /** 実行時フレーム解像度に合わせた Fisheye を返す。 */
    fun forSize(w: Int, h: Int): Fisheye {
        val sx = w / refW
        val sy = h / refH
        return Fisheye(fx * sx, fy * sy, cx * sx, cy * sy, k1, k2, k3, k4)
    }

    /** 端末内キャリブ結果で更新（refW/refH=撮影時の解析解像度）。 */
    fun update(fx: Double, fy: Double, cx: Double, cy: Double,
               k1: Double, k2: Double, k3: Double, k4: Double, refW: Double, refH: Double) {
        this.fx = fx; this.fy = fy; this.cx = cx; this.cy = cy
        this.k1 = k1; this.k2 = k2; this.k3 = k3; this.k4 = k4
        this.refW = refW; this.refH = refH; this.enabled = true
    }

    fun toJson(): String = org.json.JSONObject().apply {
        put("refW", refW); put("refH", refH); put("fx", fx); put("fy", fy); put("cx", cx); put("cy", cy)
        put("k1", k1); put("k2", k2); put("k3", k3); put("k4", k4)
    }.toString()

    fun fromJson(s: String) {
        val j = org.json.JSONObject(s)
        refW = j.getDouble("refW"); refH = j.getDouble("refH")
        fx = j.getDouble("fx"); fy = j.getDouble("fy"); cx = j.getDouble("cx"); cy = j.getDouble("cy")
        k1 = j.getDouble("k1"); k2 = j.getDouble("k2"); k3 = j.getDouble("k3"); k4 = j.getDouble("k4")
        enabled = true
    }
}
