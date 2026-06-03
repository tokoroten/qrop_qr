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
import org.json.JSONObject
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean
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
    @Volatile private var lastLabel = ""

    // アダプティブ露出
    private var camCtl: Camera2CameraControl? = null
    private var curExp = 8_000_000L
    private var isoFixed = 1550
    private var expLo = 500_000L
    private var expHi = 16_000_000L
    private var lastExpMs = 0L

    private lateinit var previewView: ImageView
    private lateinit var cropView: ImageView

    // 読み上げ(TTS)
    private var tts: TextToSpeech? = null
    private var ttsReady = false
    @Volatile private var lastSpoken = ""

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

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            try {
                val provider = future.get()
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

    private fun process(bmp: Bitmap) {
        val now = SystemClock.elapsedRealtime()
        maybeAdjustExposure(bmp, now)
        val barcodes = Tasks.await(barcodeScanner.process(InputImage.fromBitmap(bmp, 0)))
        var quad: FloatArray? = null
        var spec: Spec? = null
        for (b in barcodes) {
            val s = b.rawValue?.let { parseSpec(it) }
            val cp = b.cornerPoints
            if (s != null && cp?.size == 4) {
                quad = floatArrayOf(
                    cp[0].x.toFloat(), cp[0].y.toFloat(), cp[1].x.toFloat(), cp[1].y.toFloat(),
                    cp[2].x.toFloat(), cp[2].y.toFloat(), cp[3].x.toFloat(), cp[3].y.toFloat()
                )
                spec = s; break
            }
        }
        if (quad != null && spec != null) {
            val fieldQuad = fieldFromQuad(quad, spec)
            if (now - lastOcrMs > OCR_MS) {
                lastOcrMs = now
                val crop = warpCrop(bmp, fieldQuad, spec.w, spec.h)
                if (crop != null) {
                    val name = spec.name
                    val lang = spec.language
                    val rec = if (lang.startsWith("ja")) ocrJa else ocrLatin
                    rec.process(InputImage.fromBitmap(crop, 0)).addOnSuccessListener { res ->
                        val text = res.text.trim().replace("\n", " ")
                        if (text.isNotEmpty()) { lastLabel = "$name: $text"; speak(text, lang) }
                        Log.i(TAG, "RESULT  {\"$name\":\"$text\"}")
                        cropView.setImageBitmap(crop)
                        bottomText()?.text = if (lastLabel.isNotEmpty()) lastLabel else name
                    }
                    saveJpeg(crop, File(getExternalFilesDir(null), "crop.jpg"))
                }
            }
            drawOverlay(bmp, quad, fieldQuad, lastLabel)
        } else {
            drawStatus(bmp, "QRをかざしてください")
        }
        drawExpHud(bmp)
        if (now - lastShowMs > SHOW_MS) { lastShowMs = now; showPreview(bmp) }
    }

    private data class Spec(val name: String, val language: String, val x: Float, val y: Float, val w: Float, val h: Float)

    private fun parseSpec(raw: String): Spec? {
        if (raw.startsWith("CQR1,")) {
            return try {
                val p = raw.substring(5).split(",")
                if (p.size < 6) null
                else Spec(p[0].ifEmpty { "field" }, p[1].ifEmpty { "en" }, p[2].toFloat(), p[3].toFloat(), p[4].toFloat(), p[5].toFloat())
            } catch (e: Exception) { null }
        }
        return try {
            val j = JSONObject(raw)
            if (!j.has("x") || !j.has("y") || !j.has("w") || !j.has("h")) null
            else Spec(
                j.optString("name", j.optString("n", "field")), j.optString("language", j.optString("l", "en")),
                j.getDouble("x").toFloat(), j.getDouble("y").toFloat(), j.getDouble("w").toFloat(), j.getDouble("h").toFloat()
            )
        } catch (e: Exception) { null }
    }

    private fun fieldFromQuad(qrQuad: FloatArray, s: Spec): FloatArray {
        val unit = floatArrayOf(0f, 0f, 1f, 0f, 1f, 1f, 0f, 1f)
        val m = Matrix(); m.setPolyToPoly(unit, 0, qrQuad, 0, 4)
        val field = floatArrayOf(s.x, s.y, s.x + s.w, s.y, s.x + s.w, s.y + s.h, s.x, s.y + s.h)
        m.mapPoints(field); return field
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

    private fun drawStatus(bmp: Bitmap, text: String) {
        Canvas(bmp).drawText(text, 16f, 56f,
            Paint().apply { color = Color.WHITE; textSize = 48f; isAntiAlias = true; setShadowLayer(6f, 0f, 0f, Color.BLACK) })
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

    private fun bottomText(): TextView? = findViewById(R.id.text)
    private fun ui(msg: String) { runOnUiThread { bottomText()?.text = msg } }

    /** 認識値を読み上げ（同じ値の連呼は抑制）。日本語は対応エンジンが要る。 */
    private fun speak(text: String, lang: String) {
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

    override fun onDestroy() { tts?.stop(); tts?.shutdown(); camExec.shutdown(); super.onDestroy() }

    companion object {
        private const val TAG = "QropQR"
        private const val TTS_ENGINE = "ai.fd.josee.app.tts"  // Fairy Josee（オフライン日英TTS）。未導入時はTTS無効
        private const val OCR_MS = 600L
        private const val SHOW_MS = 80L
        private const val MIN_EXP_NS = 250_000L     // 1/4000s（明るい場面の下限）
        private const val MAX_EXP_NS = 16_000_000L  // 1/62s（ブラー抑制の上限＝暗所でもこれ以上は開けない）
    }
}
