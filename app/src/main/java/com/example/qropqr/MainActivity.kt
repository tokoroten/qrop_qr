package com.example.qropqr

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.Path
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraMetadata
import android.hardware.camera2.CaptureRequest
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import java.util.Locale
import android.util.Size
import android.view.KeyEvent
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.interop.Camera2CameraControl
import androidx.camera.camera2.interop.Camera2CameraInfo
import androidx.camera.camera2.interop.CaptureRequestOptions
import androidx.camera.camera2.interop.ExperimentalCamera2Interop
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import com.google.android.gms.tasks.Tasks
import com.google.mlkit.vision.barcode.BarcodeScannerOptions
import com.google.mlkit.vision.barcode.BarcodeScanning
import com.google.mlkit.vision.barcode.common.Barcode
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.japanese.JapaneseTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * Qrop QR — QR仕様(QRサイズ単位の相対矩形)で領域を切り出しOCR。
 * モーションブラー対策に Camera2 manual sensor で高速シャッター＋高ゲインを適用。
 * UI: 上=LIVE+認識枠 / 中=切出し画像 / 下=認識文字列。ML Kitは端末内バンドル(GMS非依存)。
 */
@OptIn(ExperimentalCamera2Interop::class)
class MainActivity : AppCompatActivity() {

    private val camExec = Executors.newSingleThreadExecutor()
    private val busy = AtomicBoolean(false)
    private var lastOcrMs = 0L
    private var lastShowMs = 0L

    // マルチQR用の状態（フィールド名で管理）
    private val fieldValues = java.util.concurrent.ConcurrentHashMap<String, String>()  // name -> 最新OCR値
    private val orientCache = HashMap<String, Pair<Int, Long>>()                         // name -> (brK, ts) QRごとの向き
    private val lastOcrByName = HashMap<String, Long>()                                  // name -> 最終OCR時刻（公平なRR）

    // アダプティブ露出
    private var camCtl: Camera2CameraControl? = null
    private var curExp = 8_000_000L
    private var isoFixed = 1550
    private var expLo = 500_000L
    private var expHi = 16_000_000L
    private var lastExpMs = 0L

    // 魚眼キャリブ撮影モード（音量↑＋↓の同時押しで実行時にON/OFF）
    @Volatile private var calibMode = false
    private var calibCount = 0
    private var lastCalibMs = 0L
    private val keysDown = HashSet<Int>()   // 物理キーの同時押し判定用
    private var chordFired = false          // chord発火済みフラグ（キーリピートの再発火抑制）

    private lateinit var previewView: ImageView
    private lateinit var cropView: ImageView

    // 読み上げ(TTS)
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    @Volatile private var lastSpoken = ""

    // 端末内蔵HTTPサーバ（OCR結果をブラウザでライブ確認＋蓄積閲覧）
    private val httpServer = OcrHttpServer(HTTP_PORT)
    private var httpUrl = ""   // HUDに表示する閲覧URL（同一LANのIP優先）

    private val barcodeScanner = BarcodeScanning.getClient(
        BarcodeScannerOptions.Builder().setBarcodeFormats(Barcode.FORMAT_QR_CODE).build()
    )
    private val ocrLatin = TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
    private val ocrJa = TextRecognition.getClient(JapaneseTextRecognizerOptions.Builder().build())

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        previewView = findViewById(R.id.preview)
        cropView = findViewById(R.id.crop)
        httpServer.start()
        Log.i(TAG, "HTTP: ${httpServer.urls().joinToString("  /  ")}")
        // HUD表示用：同一LANのIPがあれば優先、無ければUSB(localhost)案内
        httpUrl = httpServer.urls().firstOrNull { !it.startsWith("http://localhost") }
            ?: "http://localhost:$HTTP_PORT (USB: adb forward)"
        tts = TextToSpeech(this, { st ->
            ttsReady = (st == TextToSpeech.SUCCESS)
            Log.i(TAG, "TTS init=$ttsReady engine=${runCatching { tts?.defaultEngine }.getOrNull()}")
            if (ttsReady) {
                tts?.setLanguage(Locale.US)
                tts?.speak("Qrop QR Ready", TextToSpeech.QUEUE_FLUSH, null, "ready")
            }
        }, TTS_ENGINE)
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
            ui("CAMERA権限待ち（install -g 推奨）"); return
        }
        startCamera()
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) startCamera()
    }

    // THINKLETはタッチ操作が無いため、物理キーの「音量↑＋音量↓ 同時押し」を隠しコマンドとして
    // 魚眼キャリブ撮影モードのON/OFFに使う（再ビルド不要）。単独押しは通常の音量動作を残す。
    override fun onKeyDown(keyCode: Int, event: KeyEvent): Boolean {
        keysDown.add(keyCode)
        val chord = keysDown.contains(KeyEvent.KEYCODE_VOLUME_UP) && keysDown.contains(KeyEvent.KEYCODE_VOLUME_DOWN)
        if (chord && !chordFired) { chordFired = true; toggleCalibMode(); return true }
        if (chord) return true   // chord保持中は音量変化を抑制
        return super.onKeyDown(keyCode, event)
    }

    override fun onKeyUp(keyCode: Int, event: KeyEvent): Boolean {
        keysDown.remove(keyCode)
        if (!(keysDown.contains(KeyEvent.KEYCODE_VOLUME_UP) && keysDown.contains(KeyEvent.KEYCODE_VOLUME_DOWN))) chordFired = false
        return super.onKeyUp(keyCode, event)
    }

    /** キャリブ撮影モードのトグル。入る時はカウンタをリセットしTTSで案内。 */
    private fun toggleCalibMode() {
        calibMode = !calibMode
        if (calibMode) {
            calibCount = 0; lastCalibMs = 0L
            announce("キャリブレーションモードを開始します。チェスボードを傾けて見せてください")
        } else {
            announce("キャリブレーションを中止しました")
        }
        Log.i(TAG, "calibMode=$calibMode (volume up+down chord)")
    }

    /** 状態アナウンス（連呼抑制せず即読み上げ。日本語固定）。 */
    private fun announce(text: String) {
        val t = tts ?: return
        if (!ttsReady) return
        t.setLanguage(Locale.JAPANESE)
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "announce")
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
                // 解析解像度。センサは8MP(3264×2448)だが、レンズ/センサの実効解像力が頭打ちで
                // 8MPは「空の画素」が増えコストだけ上がる（実機検証で確認）。2048×1536 が速度と実効品質の最適点。
                val rs = ResolutionSelector.Builder().setResolutionStrategy(
                    ResolutionStrategy(Size(1920, 1080), ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER)
                ).build()
                val analysis = ImageAnalysis.Builder()
                    .setResolutionSelector(rs)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                analysis.setAnalyzer(camExec) { proxy -> onFrame(proxy) }
                provider.unbindAll()
                val camera = provider.bindToLifecycle(this, CameraSelector.DEFAULT_BACK_CAMERA, analysis)
                applyFastShutter(camera)
                Log.i(TAG, "camera bound")
            } catch (e: Exception) { Log.e(TAG, "camera bind failed", e) }
        }, ContextCompat.getMainExecutor(this))
    }

    /** Camera2 manual sensor。ゲイン最大固定＋露光時間を明るさで自動調整（上限でブラー抑制）。 */
    private fun applyFastShutter(camera: Camera) {
        try {
            val info = Camera2CameraInfo.from(camera.cameraInfo)
            val caps = info.getCameraCharacteristic(CameraCharacteristics.REQUEST_AVAILABLE_CAPABILITIES)
            val expRange = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_EXPOSURE_TIME_RANGE)
            val isoRange = info.getCameraCharacteristic(CameraCharacteristics.SENSOR_INFO_SENSITIVITY_RANGE)
            val manual = caps?.contains(CameraMetadata.REQUEST_AVAILABLE_CAPABILITIES_MANUAL_SENSOR) == true
            Log.i(TAG, "manualSensor=$manual expRange=$expRange isoRange=$isoRange")
            if (manual && expRange != null && isoRange != null) {
                expLo = max(expRange.lower, MIN_EXP_NS)
                expHi = min(expRange.upper, MAX_EXP_NS)
                isoFixed = isoRange.upper
                curExp = 12_000_000L.coerceIn(expLo, expHi)   // 暗所想定で長めから開始→自動短縮
                camCtl = Camera2CameraControl.from(camera.cameraControl)
                applyExposure()
                Log.i(TAG, "adaptive exposure: range[${expLo / 1000}-${expHi / 1000}us] ISO=$isoFixed start=${curExp / 1000}us")
            } else {
                val es = camera.cameraInfo.exposureState
                if (es.isExposureCompensationSupported) {
                    val step = es.exposureCompensationStep
                    val stepEv = step.numerator.toDouble() / step.denominator
                    val idx = Math.round(-2.0 / stepEv).toInt().coerceIn(es.exposureCompensationRange.lower, es.exposureCompensationRange.upper)
                    camera.cameraControl.setExposureCompensationIndex(idx)
                }
                Log.i(TAG, "manual非対応 → 露出補正(-2EV)で代替")
            }
        } catch (e: Exception) { Log.w(TAG, "applyFastShutter failed", e) }
    }

    private fun applyExposure() {
        camCtl?.setCaptureRequestOptions(
            CaptureRequestOptions.Builder()
                .setCaptureRequestOption(CaptureRequest.CONTROL_AE_MODE, CameraMetadata.CONTROL_AE_MODE_OFF)
                .setCaptureRequestOption(CaptureRequest.SENSOR_EXPOSURE_TIME, curExp)
                .setCaptureRequestOption(CaptureRequest.SENSOR_SENSITIVITY, isoFixed)
                .build()
        )
    }

    /** 明るさを測り、ブラー上限(MAX_EXP_NS)内で露光時間を自動調整（目標 平均輝度 ~110）。 */
    private fun maybeAdjustExposure(bmp: Bitmap, now: Long) {
        val ctl = camCtl ?: return
        if (now - lastExpMs < 350) return
        lastExpMs = now
        val b = meanBrightness(bmp)
        val newExp = when {
            b < 85 -> (curExp * 1.5).toLong()
            b > 145 -> (curExp * 0.7).toLong()
            else -> curExp
        }.coerceIn(expLo, expHi)
        if (newExp != curExp) { curExp = newExp; applyExposure() }
    }

    private fun meanBrightness(bmp: Bitmap): Double {
        val s = Bitmap.createScaledBitmap(bmp, 32, 24, true)
        val px = IntArray(32 * 24); s.getPixels(px, 0, 32, 0, 0, 32, 24); s.recycle()
        var sum = 0L
        for (p in px) sum += ((p shr 16 and 0xff) + (p shr 8 and 0xff) + (p and 0xff))
        return sum.toDouble() / (px.size * 3)
    }

    private fun drawExpHud(bmp: Bitmap) {
        val s = if (curExp > 0) 1_000_000_000L / curExp else 0
        Canvas(bmp).drawText("exp≈1/${s}s ISO=$isoFixed", 16f, bmp.height - 20f,
            Paint().apply { color = Color.GREEN; textSize = 40f; isAntiAlias = true; setShadowLayer(5f, 0f, 0f, Color.BLACK) })
    }

    private fun onFrame(proxy: ImageProxy) {
        try {
            if (!busy.compareAndSet(false, true)) return
            process(proxyToBitmap(proxy))
        } catch (e: Exception) {
            Log.w(TAG, "frame error", e)
        } finally {
            busy.set(false); proxy.close()
        }
    }

    /** 1QRぶんの描画・切り出し情報（QRごとに独立。複数QRを同時に扱う）。 */
    private class FieldDraw(
        val qrQuad: FloatArray, val spec: Spec, val fieldQuad: FloatArray,
        val hqUndist: Matrix?, val undField: FloatArray?
    )

    private fun process(bmp: Bitmap) {
        val now = SystemClock.elapsedRealtime()
        maybeAdjustExposure(bmp, now)
        if (calibMode) {
            runCalibCapture(bmp, now)
            if (now - lastShowMs > SHOW_MS) { lastShowMs = now; showPreview(bmp) }
            return
        }
        val barcodes = Tasks.await(barcodeScanner.process(InputImage.fromBitmap(bmp, 0)))
        val fe = if (Calib.enabled) Calib.forSize(bmp.width, bmp.height) else null

        // 検出した全QRを処理：QRごとに向きを判定し、フィールド領域を計算
        val fields = ArrayList<FieldDraw>()
        for (b in barcodes) {
            val s = parseSpec(b) ?: continue
            val cp = b.cornerPoints ?: continue
            if (cp.size != 4) continue
            val quad = floatArrayOf(
                cp[0].x.toFloat(), cp[0].y.toFloat(), cp[1].x.toFloat(), cp[1].y.toFloat(),
                cp[2].x.toFloat(), cp[2].y.toFloat(), cp[3].x.toFloat(), cp[3].y.toFloat()
            )
            // QRシンボル固有の向き(TL,TR,BR,BL)に並べ替え（QRごとに独立判定＝混線しない）
            val q = orderQuadBySymbol(bmp, quad, s.name, now)
            fields.add(buildField(bmp, q, s, fe))
        }

        if (fields.isNotEmpty()) {
            // OCRは1ティック1フィールドのラウンドロビン（検出/枠描画は毎フレーム、OCRだけ間引いて速度確保）。
            // 公平性のため「最も長くOCRしていないフィールド」を選ぶ。
            if (now - lastOcrMs > OCR_MS) {
                lastOcrMs = now
                val f = fields.minByOrNull { lastOcrByName[it.spec.name] ?: 0L }!!
                lastOcrByName[f.spec.name] = now
                ocrField(bmp, f, fe)
            }
            // すべてのフィールドの枠を毎フレーム描画
            for (f in fields) {
                if (fe != null && f.undField != null) drawOverlayCurved(bmp, f.qrQuad, f.undField, fe, f.fieldQuad, labelFor(f.spec.name))
                else drawOverlay(bmp, f.qrQuad, f.fieldQuad, labelFor(f.spec.name))
            }
        }
        drawHud(bmp, fields.size)
        if (fields.isEmpty()) drawHint(bmp, "QR付きフォームをかざしてください")
        drawExpHud(bmp)
        if (now - lastShowMs > SHOW_MS) { lastShowMs = now; showPreview(bmp) }
    }

    /** QRの並び替え済み4隅とspecから、フィールド領域（魚眼補正あり/なし）を構築。 */
    private fun buildField(bmp: Bitmap, q: FloatArray, spec: Spec, fe: Fisheye?): FieldDraw {
        if (fe != null) {
            // 魚眼補正: QRの4隅を歪み補正→補正空間で unit→補正画素のホモグラフィ→フィールド隅を求める。
            val und = FloatArray(8)
            for (i in 0..3) {
                val p = fe.undistortPixel(q[2 * i].toDouble(), q[2 * i + 1].toDouble())
                und[2 * i] = p[0].toFloat(); und[2 * i + 1] = p[1].toFloat()
            }
            val hq = Matrix().apply { setPolyToPoly(UNIT, 0, und, 0, 4) }
            val uf = floatArrayOf(spec.x, spec.y, spec.x + spec.w, spec.y, spec.x + spec.w, spec.y + spec.h, spec.x, spec.y + spec.h)
            hq.mapPoints(uf)
            // オーバーレイ用に、補正空間のフィールド隅を歪み画素へ戻す
            val fq = FloatArray(8)
            for (i in 0..3) {
                val p = fe.distortPixel(uf[2 * i].toDouble(), uf[2 * i + 1].toDouble())
                fq[2 * i] = p[0].toFloat(); fq[2 * i + 1] = p[1].toFloat()
            }
            return FieldDraw(q, spec, fq, hq, uf)
        }
        return FieldDraw(q, spec, fieldFromQuad(q, spec), null, null)
    }

    /** 1フィールドを切り出してOCR。結果はフィールド名で蓄積し、値が変わった時だけ読み上げ＆記録。 */
    private fun ocrField(bmp: Bitmap, f: FieldDraw, fe: Fisheye?) {
        val crop = if (f.hqUndist != null && fe != null) warpCropUndistort(bmp, f.hqUndist, f.spec, fe)
                   else warpCrop(bmp, f.fieldQuad, f.spec.w, f.spec.h)
        crop ?: return
        val name = f.spec.name
        val lang = f.spec.language
        val rec = if (lang.startsWith("ja")) ocrJa else ocrLatin
        rec.process(InputImage.fromBitmap(crop, 0)).addOnSuccessListener { res ->
            val text = res.text.trim().replace("\n", " ")
            if (text.isNotEmpty()) {
                val changed = fieldValues.put(name, text) != text
                cropView.setImageBitmap(crop)
                bottomText()?.text = fieldValues.entries.joinToString("\n") { "${it.key}: ${it.value}" }
                httpServer.publish(name, text, lang, jpegBytes(crop), System.currentTimeMillis(), f.spec.id)
                if (changed) speak(name, text, lang)
            }
            Log.i(TAG, "RESULT  {\"$name\":\"$text\"}")
        }
        saveJpeg(crop, File(getExternalFilesDir(null), "crop.jpg"))
    }

    /** オーバーレイのラベル。OCR済みなら "name: 値"、未OCRなら name。 */
    private fun labelFor(name: String): String = fieldValues[name]?.let { "$name: $it" } ?: name

    private data class Spec(val name: String, val language: String, val x: Float, val y: Float, val w: Float, val h: Float, val id: Int = 0)

    /**
     * QRペイロード（CQR2 バイナリ専用）を Spec に解釈。固定10Bヘッダ＋末尾name:
     *  [0] ver=1 / [1..2] id(uint16 LE) / [3] flags(bit0-1=lang 0:en 1:ja) /
     *  [4..6] x,y / [7..9] w,h  …各 12bit 符号付き固定小数 Q8.4(=値/16) を2個3バイトにパック /
     *  [10..] name(UTF-8, 残り全部=可変長, 長さ識別不要)
     * 12bitパック: (A,B) → [A>>4, (A&0xF)<<4 | B>>8, B&0xFF]。Q8.4=整数8bit+小数4bit, 範囲±128, 分解能1/16。
     * ML Kit の rawBytes（バイト透過）で読む。未リリースのため CSV/JSON 等の旧形式は非対応。
     */
    private fun parseSpec(b: Barcode): Spec? {
        val rb = b.rawBytes ?: return null
        if (rb.size < CQR2_HEADER || (rb[0].toInt() and 0xff) != CQR2_VER) return null
        return try {
            val id = (rb[1].toInt() and 0xff) or ((rb[2].toInt() and 0xff) shl 8)
            val lang = if ((rb[3].toInt() and 0x03) == 1) "ja_jp" else "en"
            fun u(o: Int) = rb[o].toInt() and 0xff
            fun s12(v: Int): Int = if (v >= 0x800) v - 0x1000 else v   // 12bit符号拡張
            // [o..o+2] にパックされた2個の12bit値(A,B)を復元（/16でQ8.4実値）
            val x = s12((u(4) shl 4) or (u(5) shr 4)) / 16f
            val y = s12(((u(5) and 0x0f) shl 8) or u(6)) / 16f
            val w = s12((u(7) shl 4) or (u(8) shr 4)) / 16f
            val h = s12(((u(8) and 0x0f) shl 8) or u(9)) / 16f
            val name = if (rb.size > CQR2_HEADER) String(rb, CQR2_HEADER, rb.size - CQR2_HEADER, Charsets.UTF_8).trim() else ""
            Spec(name.ifEmpty { "field" }, lang, x, y, w, h, id)
        } catch (e: Exception) { null }
    }

    // QRシンボルの向き判定用の再利用テンポラリ（camExec単一スレッドで逐次使用）。
    private val orMat = Matrix()
    private val orMv = FloatArray(9)

    /**
     * ML Kit cornerPoints は画像基準（常に画像TLから時計回り）でシンボルの向きを含まない。
     * 4隅のファインダパターン有無を“数十点のピクセル読み取り”だけで判定し、ファインダの無い隅=BR
     * を特定→4隅を [TL,TR,BR,BL]（シンボル基準）に並べ替える。
     *
     * 判定はQRごと（name でキャッシュ）に ORIENT_MS 間隔で更新。複数QRがあっても各QRが自分の
     * 向きだけを持つので、別QRの軸が混線してフィールドがズレることがない。
     */
    private fun orderQuadBySymbol(bmp: Bitmap, quad: FloatArray, name: String, now: Long): FloatArray {
        val cached = orientCache[name]
        val brK = if (cached != null && now - cached.second < ORIENT_MS) {
            cached.first
        } else {
            orMat.setPolyToPoly(UNIT, 0, quad, 0, 4)
            orMat.getValues(orMv)
            var k = 0; var best = Double.MAX_VALUE
            val sc = DoubleArray(4)
            for (i in 0..3) { sc[i] = finderness(bmp, i); if (sc[i] < best) { best = sc[i]; k = i } }
            if (DBG_ORIENT) Log.i(TAG, "ORIENT[$name] sc=[%.0f,%.0f,%.0f,%.0f] brK=%d".format(sc[0], sc[1], sc[2], sc[3], k))
            orientCache[name] = Pair(k, now)
            k
        }
        // ML Kit順は画像CW [TL,TR,BR,BL]。BRの対角がTL、CW隣接でTR/BLが決まる。
        val tlK = (brK + 2) % 4; val trK = (brK + 3) % 4; val blK = (brK + 1) % 4
        return floatArrayOf(
            quad[2 * tlK], quad[2 * tlK + 1], quad[2 * trK], quad[2 * trK + 1],
            quad[2 * brK], quad[2 * brK + 1], quad[2 * blK], quad[2 * blK + 1]
        )
    }

    /** 隅kのファインダらしさ。中心=暗, 中ﾘﾝｸﾞ=明, 外周=暗（1:1:3:1:1）をモジュール数不明でも多scaleで評価。 */
    private fun finderness(bmp: Bitmap, k: Int): Double {
        var best = -Double.MAX_VALUE
        for (f in FINDER_FRACS) {
            val center = sampleFinder(bmp, k, f, 0.5f, 0.5f)
            val light = (sampleFinder(bmp, k, f, 0.214f, 0.5f) + sampleFinder(bmp, k, f, 0.786f, 0.5f) +
                sampleFinder(bmp, k, f, 0.5f, 0.214f) + sampleFinder(bmp, k, f, 0.5f, 0.786f)) / 4.0
            val outer = (sampleFinder(bmp, k, f, 0.071f, 0.5f) + sampleFinder(bmp, k, f, 0.929f, 0.5f) +
                sampleFinder(bmp, k, f, 0.5f, 0.071f) + sampleFinder(bmp, k, f, 0.5f, 0.929f)) / 4.0
            val sc = (255 - center) + light + (255 - outer)
            if (sc > best) best = sc
        }
        return best
    }

    /** ファインダ内の正規化座標(a,b∈[0,1])を、隅kに合わせてunit→画像へ写し輝度を返す。 */
    private fun sampleFinder(bmp: Bitmap, k: Int, f: Float, a: Float, b: Float): Int {
        val uu = if (k == 1 || k == 2) 1f - a * f else a * f
        val vv = if (k == 2 || k == 3) 1f - b * f else b * f
        val den = orMv[6] * uu + orMv[7] * vv + orMv[8]
        val x = (orMv[0] * uu + orMv[1] * vv + orMv[2]) / den
        val y = (orMv[3] * uu + orMv[4] * vv + orMv[5]) / den
        val xi = x.toInt().coerceIn(0, bmp.width - 1)
        val yi = y.toInt().coerceIn(0, bmp.height - 1)
        val p = bmp.getPixel(xi, yi)
        return ((p shr 16 and 0xff) + (p shr 8 and 0xff) + (p and 0xff)) / 3
    }

    private fun fieldFromQuad(qrQuad: FloatArray, s: Spec): FloatArray {
        val unit = floatArrayOf(0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f)
        val m = Matrix(); m.setPolyToPoly(unit, 0, qrQuad, 0, 4)
        val field = floatArrayOf(s.x, s.y, s.x + s.w, s.y, s.x + s.w, s.y + s.h, s.x, s.y + s.h)
        m.mapPoints(field); return field
    }

    /**
     * 魚眼補正版クロップ。出力画素 →(ホモグラフィ)→ 補正画素 →(再歪み)→ 歪み画素 を辿り、
     * 元の歪み画像から直接バイリニア標本化する（全画面undistort不要・軽量）。
     */
    private fun warpCropUndistort(src: Bitmap, hq: Matrix, s: Spec, fe: Fisheye): Bitmap? {
        val ppu = 48f
        val outW = max(8, min(2000, (s.w * ppu).roundToInt()))
        val outH = max(8, min(2000, (s.h * ppu).roundToInt()))
        val m = FloatArray(9); hq.getValues(m)
        val sw = src.width; val sh = src.height
        val sp = IntArray(sw * sh); src.getPixels(sp, 0, sw, 0, 0, sw, sh)
        val out = IntArray(outW * outH)
        for (oy in 0 until outH) {
            val v = s.y + (oy + 0.5f) / outH * s.h
            for (ox in 0 until outW) {
                val u = s.x + (ox + 0.5f) / outW * s.w
                val den = m[6] * u + m[7] * v + m[8]
                val ux = (m[0] * u + m[1] * v + m[2]) / den   // 補正画素
                val uy = (m[3] * u + m[4] * v + m[5]) / den
                val d = fe.distortPixel(ux.toDouble(), uy.toDouble())  // 歪み画素
                out[oy * outW + ox] = bilinear(sp, sw, sh, d[0], d[1])
            }
        }
        return Bitmap.createBitmap(out, outW, outH, Bitmap.Config.ARGB_8888)
    }

    private fun bilinear(px: IntArray, w: Int, h: Int, fx: Double, fy: Double): Int {
        if (fx < 0.0 || fy < 0.0 || fx > w - 1.0 || fy > h - 1.0) return 0xFF000000.toInt()
        val x0 = fx.toInt(); val y0 = fy.toInt()
        val x1 = min(x0 + 1, w - 1); val y1 = min(y0 + 1, h - 1)
        val dx = fx - x0; val dy = fy - y0
        val p00 = px[y0 * w + x0]; val p10 = px[y0 * w + x1]
        val p01 = px[y1 * w + x0]; val p11 = px[y1 * w + x1]
        fun lerpCh(sh: Int): Int {
            val a = (p00 shr sh and 0xff) * (1 - dx) + (p10 shr sh and 0xff) * dx
            val b = (p01 shr sh and 0xff) * (1 - dx) + (p11 shr sh and 0xff) * dx
            return (a * (1 - dy) + b * dy).roundToInt().coerceIn(0, 255)
        }
        return (0xff shl 24) or (lerpCh(16) shl 16) or (lerpCh(8) shl 8) or lerpCh(0)
    }

    private fun warpCrop(src: Bitmap, quad: FloatArray, w: Float, h: Float): Bitmap? {
        val ppu = 48f
        val outW = max(8, min(2000, (w * ppu).roundToInt()))
        val outH = max(8, min(2000, (h * ppu).roundToInt()))
        val outRect = floatArrayOf(0f, 0f, outW.toFloat(), 0f, outW.toFloat(), outH.toFloat(), 0f, outH.toFloat())
        val m = Matrix(); if (!m.setPolyToPoly(quad, 0, outRect, 0, 4)) return null
        val out = Bitmap.createBitmap(outW, outH, Bitmap.Config.ARGB_8888)
        Canvas(out).drawBitmap(src, m, Paint(Paint.FILTER_BITMAP_FLAG or Paint.ANTI_ALIAS_FLAG))
        return out
    }

    private fun drawOverlay(bmp: Bitmap, qrQuad: FloatArray, fieldQuad: FloatArray, label: String) {
        val c = Canvas(bmp)
        c.drawPath(quadPath(qrQuad), Paint().apply { color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true })
        c.drawPath(quadPath(fieldQuad), Paint().apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 8f; isAntiAlias = true })
        if (label.isNotEmpty()) c.drawText(label, fieldQuad[0], fieldQuad[1] - 14f,
            Paint().apply { color = Color.YELLOW; textSize = 54f; isAntiAlias = true; setShadowLayer(6f, 0f, 0f, Color.BLACK) })
    }

    private fun quadPath(q: FloatArray) = Path().apply {
        moveTo(q[0], q[1]); lineTo(q[2], q[3]); lineTo(q[4], q[5]); lineTo(q[6], q[7]); close()
    }

    /**
     * 魚眼補正版オーバーレイ。フィールド枠は補正空間の矩形を各辺で細分し、歪み画素へ戻して
     * 曲線で描く（歪んだLive画像上で実際のフィールド境界に沿う）。QR枠は検出隅をそのまま。
     */
    private fun drawOverlayCurved(bmp: Bitmap, qrQuad: FloatArray, undField: FloatArray, fe: Fisheye, fieldQuad: FloatArray, label: String) {
        val c = Canvas(bmp)
        c.drawPath(quadPath(qrQuad), Paint().apply { color = Color.GREEN; style = Paint.Style.STROKE; strokeWidth = 6f; isAntiAlias = true })
        val seg = 12
        val path = Path()
        for (e in 0 until 4) {
            val ax = undField[2 * e]; val ay = undField[2 * e + 1]
            val bx = undField[2 * ((e + 1) % 4)]; val by = undField[2 * ((e + 1) % 4) + 1]
            for (i in 0..seg) {
                val t = i.toFloat() / seg
                val d = fe.distortPixel((ax + (bx - ax) * t).toDouble(), (ay + (by - ay) * t).toDouble())
                val px = d[0].toFloat(); val py = d[1].toFloat()
                if (e == 0 && i == 0) path.moveTo(px, py) else path.lineTo(px, py)
            }
        }
        path.close()
        c.drawPath(path, Paint().apply { color = Color.CYAN; style = Paint.Style.STROKE; strokeWidth = 8f; isAntiAlias = true })
        if (label.isNotEmpty()) c.drawText(label, fieldQuad[0], fieldQuad[1] - 14f,
            Paint().apply { color = Color.YELLOW; textSize = 54f; isAntiAlias = true; setShadowLayer(6f, 0f, 0f, Color.BLACK) })
    }

    /** キャリブ撮影：解析フレーム(オーバーレイ前)を一定間隔で external files/calib に保存。N枚で自動終了。 */
    private fun runCalibCapture(bmp: Bitmap, now: Long) {
        if (now - lastCalibMs > CALIB_INTERVAL_MS) {
            lastCalibMs = now
            val dir = File(getExternalFilesDir(null), "calib").apply { mkdirs() }
            saveJpeg(bmp, File(dir, "frame%03d.jpg".format(calibCount)))  // クリーンなフレーム
            calibCount++
            Log.i(TAG, "calib saved $calibCount/$CALIB_MAX -> ${dir.absolutePath}")
            if (calibCount >= CALIB_MAX) {
                calibMode = false
                announce("キャリブレーション撮影完了。${CALIB_MAX}枚保存しました")
            } else if (calibCount % 10 == 0) {
                announce("${calibCount}枚")
            }
        }
        Canvas(bmp).drawText("CALIB ${min(calibCount, CALIB_MAX)}/$CALIB_MAX  (音量±同時で中止)", 16f, 56f,
            Paint().apply { color = Color.YELLOW; textSize = 48f; isAntiAlias = true; setShadowLayer(6f, 0f, 0f, Color.BLACK) })
    }

    /** 上部の状態バー：QR検出数・保存件数・TTS可否・閲覧URL。デモで「どこを見るか」を明示。 */
    private fun drawHud(bmp: Bitmap, qrCount: Int) {
        val c = Canvas(bmp)
        val w = bmp.width.toFloat()
        c.drawRect(0f, 0f, w, 92f, Paint().apply { color = Color.argb(150, 0, 0, 0) })
        val (fld, rec) = httpServer.counts()
        c.drawText("Qrop QR    QR:$qrCount    保存 ${fld}項目/${rec}件    ${if (ttsReady) "TTS:on" else "TTS:off"}",
            16f, 40f, Paint().apply { color = Color.WHITE; textSize = 34f; isAntiAlias = true; setShadowLayer(4f, 0f, 0f, Color.BLACK) })
        c.drawText(httpUrl, 16f, 80f,
            Paint().apply { color = Color.CYAN; textSize = 30f; isAntiAlias = true; setShadowLayer(4f, 0f, 0f, Color.BLACK) })
    }

    /** 画面中央のガイダンス（QR未検出時）。 */
    private fun drawHint(bmp: Bitmap, text: String) {
        val p = Paint().apply { color = Color.WHITE; textSize = 50f; isAntiAlias = true; setShadowLayer(6f, 0f, 0f, Color.BLACK) }
        Canvas(bmp).drawText(text, (bmp.width - p.measureText(text)) / 2f, bmp.height * 0.55f, p)
    }

    private fun showPreview(bmp: Bitmap) {
        val shown = if (bmp.width > 1080) Bitmap.createScaledBitmap(bmp, 1080, bmp.height * 1080 / bmp.width, true) else bmp
        runOnUiThread { previewView.setImageBitmap(shown) }
    }

    private fun proxyToBitmap(proxy: ImageProxy): Bitmap {
        val plane = proxy.planes[0]
        val buf = plane.buffer.apply { rewind() }
        val wPad = plane.rowStride / plane.pixelStride
        val bmp = Bitmap.createBitmap(wPad, proxy.height, Bitmap.Config.ARGB_8888)
        bmp.copyPixelsFromBuffer(buf)
        return if (wPad != proxy.width) Bitmap.createBitmap(bmp, 0, 0, proxy.width, proxy.height) else bmp
    }

    private fun saveJpeg(bmp: Bitmap, file: File) {
        runCatching { FileOutputStream(file).use { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) } }
    }

    private fun jpegBytes(bmp: Bitmap): ByteArray {
        val bos = java.io.ByteArrayOutputStream()
        bmp.compress(Bitmap.CompressFormat.JPEG, 85, bos)
        return bos.toByteArray()
    }

    private fun bottomText(): TextView? = findViewById(R.id.text)
    private fun ui(msg: String) { runOnUiThread { bottomText()?.text = msg } }

    /** 認識値を読み上げ（同じ値の連呼は抑制）。カラム名は読まない（日本語TTSが英字を1字ずつ読むため）。 */
    private fun speak(name: String, text: String, lang: String) {
        val t = tts ?: return
        if (!ttsReady || text.isEmpty() || text == lastSpoken) return
        val locale = if (lang.startsWith("ja")) Locale.JAPANESE else Locale.US
        val r = t.setLanguage(locale)
        if (r == TextToSpeech.LANG_MISSING_DATA || r == TextToSpeech.LANG_NOT_SUPPORTED) {
            Log.w(TAG, "TTS: $locale 非対応（既定Picoは日本語不可。josee/mimi導入が必要）"); return
        }
        lastSpoken = text
        t.speak(text, TextToSpeech.QUEUE_FLUSH, null, "qr")
    }

    override fun onDestroy() { httpServer.stop(); tts?.stop(); tts?.shutdown(); camExec.shutdown(); super.onDestroy() }

    companion object {
        private const val TAG = "QropQR"
        private const val TTS_ENGINE = "ai.fd.josee.app.tts"  // Fairy Josee（オフライン日英TTS）。未導入時はTTS無効
        private const val OCR_MS = 600L
        private const val SHOW_MS = 80L
        private const val HTTP_PORT = 8080  // 端末内HTTPサーバ。adb forward tcp:8080 tcp:8080 でPCから閲覧可
        private const val CQR2_VER = 1      // CQR2 バイナリ形式のバージョン（rawBytes[0]＝magic兼用）
        private const val CQR2_HEADER = 10  // 固定ヘッダ長（ver1+id2+flags1+ xy3B + wh3B；座標は12bit Q8.4パック）
        private const val ORIENT_MS = 250L    // 向き軸を更新する間隔（毎フレームの分類はキャッシュ軸で軽量）
        private const val DBG_ORIENT = false  // 検証用：判定したBR隅をログ出力
        private const val MIN_EXP_NS = 250_000L     // 1/4000s（明るい場面の下限）
        private const val MAX_EXP_NS = 16_000_000L  // 1/62s（ブラー抑制の上限＝暗所でもこれ以上は開けない）

        // 魚眼キャリブ撮影モード：音量↑＋↓の同時押しで実行時にON（再ビルド不要）。OCRを止め解析フレームを連続保存。
        // 撮影後 `adb pull` → tools/calib で K,D を推定し Calib に焼き込む。
        private const val CALIB_MAX = 30
        private const val CALIB_INTERVAL_MS = 1200L

        // QRサイズの単位正方形（TL,TR,BR,BL）。ML Kit cornerPoints の並びに一致。
        private val UNIT = floatArrayOf(0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f)

        // ファインダ占有率 7/M（M=モジュール数, QR ver1-6: 21..41）。モジュール数不明でも多scaleで評価。
        private val FINDER_FRACS = floatArrayOf(7f / 21, 7f / 25, 7f / 29, 7f / 33, 7f / 37, 7f / 41)
    }
}
