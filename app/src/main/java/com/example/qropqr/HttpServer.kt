package com.example.qropqr

import android.util.Log
import org.json.JSONArray
import org.json.JSONObject
import java.io.BufferedReader
import java.io.InputStreamReader
import java.io.OutputStream
import java.net.Inet4Address
import java.net.NetworkInterface
import java.net.ServerSocket
import java.net.Socket
import java.text.SimpleDateFormat
import java.util.Collections
import java.util.Date
import java.util.Locale
import java.util.concurrent.CopyOnWriteArrayList
import java.util.concurrent.Executors

/**
 * 端末内蔵の軽量HTTPサーバ（依存ライブラリなし＝GMS非依存を維持）。
 * OCR結果を「ライブ確認＋蓄積テーブル＋JSON」で外部ブラウザに見せる。
 *
 * 見るには:
 *  - 同一LAN  : http://<端末IP>:8080
 *  - USBのみ : `adb forward tcp:8080 tcp:8080` → http://localhost:8080
 *
 * エンドポイント:
 *  GET /            ライブ確認＋蓄積テーブル（JSで自動更新）
 *  GET /records     蓄積テーブルのスナップショット（サーバ描画・印刷向け）
 *  GET /records.json 機械可読JSON（将来のDB連携の橋渡し）
 *  GET /state.json  ライブ画面用（最新値＋全レコード）
 *  GET /crop.jpg    最新の切り出し画像
 */
class OcrHttpServer(private val port: Int = 8080) {

    data class Rec(val ts: Long, val name: String, val value: String, val lang: String, val id: Int)

    private val records = CopyOnWriteArrayList<Rec>()
    private val lastByName = java.util.concurrent.ConcurrentHashMap<String, String>()  // name -> 直近記録値
    @Volatile private var latest: Rec? = null
    @Volatile private var cropJpeg: ByteArray? = null
    @Volatile private var running = false
    private var server: ServerSocket? = null
    private val pool = Executors.newCachedThreadPool()

    fun start() {
        if (running) return
        running = true
        pool.execute { acceptLoop() }
    }

    fun stop() {
        running = false
        runCatching { server?.close() }
        pool.shutdownNow()
    }

    /**
     * OCR成功ごとに呼ぶ。crop は最新ライブ表示用に常時更新。
     * 「そのフィールドの値が前回と変わった時」だけ1行追加（フィールド名ごとに抑制）。
     * 複数フィールドをラウンドロビンOCRしても、同じ値の再読み取りは記録されない。
     */
    fun publish(name: String, value: String, lang: String, jpeg: ByteArray?, ts: Long, id: Int = 0) {
        if (jpeg != null) cropJpeg = jpeg
        if (value.isEmpty()) return
        latest = Rec(ts, name, value, lang, id)
        if (lastByName.put(name, value) != value) {
            records.add(Rec(ts, name, value, lang, id))
            while (records.size > MAX_RECORDS) records.removeAt(0)
        }
    }

    /** 接続情報（起動ログ表示用）。site-local な IPv4 を列挙。 */
    fun urls(): List<String> {
        val out = ArrayList<String>()
        out.add("http://localhost:$port (adb forward tcp:$port tcp:$port 経由)")
        try {
            for (ni in Collections.list(NetworkInterface.getNetworkInterfaces())) {
                if (!ni.isUp || ni.isLoopback) continue
                for (addr in Collections.list(ni.inetAddresses)) {
                    if (addr is Inet4Address && addr.isSiteLocalAddress)
                        out.add("http://${addr.hostAddress}:$port")
                }
            }
        } catch (_: Exception) {}
        return out
    }

    private fun acceptLoop() {
        val s = bindWithRetry() ?: return
        server = s
        Log.i(TAG, "HTTP listening on :$port  ->  ${urls().joinToString("  /  ")}")
        while (running) {
            val sock = try { s.accept() } catch (e: Exception) {
                if (running) Log.w(TAG, "accept failed", e); break
            }
            pool.execute { handle(sock) }
        }
    }

    /**
     * bind をリトライ。アプリ再起動直後は前プロセスのリスナがまだ生きていて EADDRINUSE になり得る
     * （SO_REUSEADDR は TIME_WAIT は救えるが“生きているリスナ”は救えない）。前プロセスがポートを
     * 解放するまで数秒待ちながら再試行する。SO_REUSEADDR も併用。
     */
    private fun bindWithRetry(): ServerSocket? {
        var attempt = 0
        while (running && attempt < BIND_RETRIES) {
            try {
                val s = ServerSocket()
                s.reuseAddress = true
                s.bind(java.net.InetSocketAddress(port))
                return s
            } catch (e: Exception) {
                attempt++
                Log.w(TAG, "bind :$port failed (try $attempt/$BIND_RETRIES): ${e.message}")
                try { Thread.sleep(BIND_RETRY_MS) } catch (_: InterruptedException) { return null }
            }
        }
        Log.e(TAG, "HTTP server gave up binding :$port")
        return null
    }

    private fun handle(sock: Socket) {
        sock.use {
            try {
                val reader = BufferedReader(InputStreamReader(it.getInputStream()))
                val line = reader.readLine() ?: return
                while (true) { val h = reader.readLine() ?: break; if (h.isEmpty()) break } // ヘッダ読み飛ばし
                val parts = line.split(" ")
                val path = if (parts.size >= 2) parts[1].substringBefore('?') else "/"
                val out = it.getOutputStream()
                when (path) {
                    "/" -> writeText(out, 200, "text/html; charset=utf-8", INDEX_HTML)
                    "/state.json" -> writeText(out, 200, "application/json; charset=utf-8", stateJson())
                    "/records.json" -> writeText(out, 200, "application/json; charset=utf-8", recordsJson())
                    "/records" -> writeText(out, 200, "text/html; charset=utf-8", recordsHtml())
                    "/crop.jpg" -> {
                        val j = cropJpeg
                        if (j == null) writeText(out, 204, "text/plain", "")
                        else writeBytes(out, 200, "image/jpeg", j)
                    }
                    else -> writeText(out, 404, "text/plain; charset=utf-8", "not found")
                }
                out.flush()
            } catch (e: Exception) {
                Log.w(TAG, "handle failed", e)
            }
        }
    }

    private fun stateJson(): String {
        val o = JSONObject()
        latest?.let { o.put("latest", recJson(it)) }
        o.put("count", records.size)
        o.put("hasCrop", cropJpeg != null)
        val arr = JSONArray()
        for (i in records.indices.reversed()) arr.put(recJson(records[i]))  // 新しい順
        o.put("records", arr)
        return o.toString()
    }

    private fun recordsJson(): String {
        val o = JSONObject()
        val arr = JSONArray()
        for (r in records) arr.put(recJson(r))  // 古い順（取得順）
        o.put("count", records.size)
        o.put("records", arr)
        return o.toString()
    }

    private fun recJson(r: Rec) = JSONObject()
        .put("ts", r.ts).put("id", r.id).put("name", r.name).put("value", r.value).put("lang", r.lang)

    private fun recordsHtml(): String {
        val sb = StringBuilder()
        sb.append("<!doctype html><html lang=ja><head><meta charset=utf-8>")
        sb.append("<title>Qrop QR records</title><style>")
        sb.append("body{font-family:sans-serif;margin:20px}table{border-collapse:collapse}")
        sb.append("td,th{border:1px solid #ccc;padding:6px 10px;text-align:left}th{background:#f3f3f3}")
        sb.append("</style></head><body>")
        sb.append("<h2>Qrop QR records (").append(records.size).append(")</h2>")
        sb.append("<table><tr><th>#</th><th>time</th><th>field</th><th>value</th><th>lang</th></tr>")
        for (i in records.indices.reversed()) {
            val r = records[i]
            sb.append("<tr><td>").append(i + 1)
              .append("</td><td>").append(esc(fmtTs(r.ts)))
              .append("</td><td>").append(esc(r.name))
              .append("</td><td>").append(esc(r.value))
              .append("</td><td>").append(esc(r.lang)).append("</td></tr>")
        }
        sb.append("</table></body></html>")
        return sb.toString()
    }

    private fun fmtTs(ts: Long): String =
        runCatching { SimpleDateFormat("HH:mm:ss", Locale.US).format(Date(ts)) }.getOrDefault("")

    private fun esc(s: String) = s.replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace("\"", "&quot;")

    private fun writeText(out: OutputStream, code: Int, ctype: String, body: String) =
        writeBytes(out, code, ctype, body.toByteArray(Charsets.UTF_8))

    private fun writeBytes(out: OutputStream, code: Int, ctype: String, body: ByteArray) {
        val status = when (code) { 200 -> "OK"; 204 -> "No Content"; 404 -> "Not Found"; else -> "OK" }
        val head = "HTTP/1.1 $code $status\r\n" +
            "Content-Type: $ctype\r\n" +
            "Content-Length: ${body.size}\r\n" +
            "Access-Control-Allow-Origin: *\r\n" +
            "Cache-Control: no-store\r\n" +
            "Connection: close\r\n\r\n"
        out.write(head.toByteArray(Charsets.US_ASCII))
        if (body.isNotEmpty()) out.write(body)
    }

    companion object {
        private const val TAG = "QropQR"
        private const val MAX_RECORDS = 200
        private const val BIND_RETRIES = 12      // 再起動直後の EADDRINUSE 対策（前プロセスのポート解放待ち）
        private const val BIND_RETRY_MS = 400L

        // ライブ確認＋蓄積テーブル。/state.json を ~700ms 間隔でポーリングして自動更新（$は使わない＝Kotlin文字列展開回避）。
        private val INDEX_HTML = """
<!doctype html><html lang="ja"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Qrop QR</title>
<style>
 body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#111;color:#eee}
 header{padding:12px 16px;background:#1b1b1b;border-bottom:1px solid #333;display:flex;align-items:center;gap:10px}
 header h1{font-size:16px;margin:0;font-weight:600}
 .wrap{display:flex;flex-wrap:wrap;gap:16px;padding:16px}
 .card{background:#1b1b1b;border:1px solid #333;border-radius:10px;padding:14px}
 .live{flex:1;min-width:320px}.log{flex:1;min-width:320px}
 #crop{max-width:100%;border-radius:8px;background:#000;display:block}
 .val{font-size:30px;font-weight:700;margin:10px 0 4px;word-break:break-word}
 .meta{color:#9aa;font-size:13px}
 table{border-collapse:collapse;width:100%;font-size:14px}
 th,td{border-bottom:1px solid #2a2a2a;padding:7px 8px;text-align:left}
 th{color:#9aa;font-weight:600}.empty{color:#777;padding:20px 0}
 .dot{width:9px;height:9px;border-radius:50%;background:#3c3;display:inline-block}
</style></head><body>
<header><span class="dot"></span><h1>Qrop QR &mdash; live</h1><span class="meta" id="cnt"></span></header>
<div class="wrap">
 <div class="card live"><img id="crop" alt="crop">
   <div class="val" id="val">&mdash;</div><div class="meta" id="vmeta"></div></div>
 <div class="card log">
   <table><thead><tr><th>#</th><th>time</th><th>field</th><th>value</th><th>lang</th></tr></thead>
   <tbody id="rows"></tbody></table>
   <div class="empty" id="empty">まだ読み取りがありません</div></div>
</div>
<script>
function esc(s){var d=document.createElement('div');d.textContent=(s==null?'':String(s));return d.innerHTML;}
function ftime(ts){try{return new Date(ts).toLocaleTimeString();}catch(e){return '';}}
function tick(){
 fetch('/state.json',{cache:'no-store'}).then(function(r){return r.json();}).then(function(s){
  var l=s.latest;
  document.getElementById('crop').src='/crop.jpg?t='+(l?l.ts:0);
  document.getElementById('val').textContent=l?l.value:'—';
  document.getElementById('vmeta').textContent=l?(l.name+'  /  '+l.lang+'  /  '+ftime(l.ts)):'';
  document.getElementById('cnt').textContent=(s.count||0)+' records';
  var rows=s.records||[];
  document.getElementById('empty').style.display=rows.length?'none':'block';
  var html='';
  for(var i=0;i<rows.length;i++){var r=rows[i];
   html+='<tr><td>'+(rows.length-i)+'</td><td>'+esc(ftime(r.ts))+'</td><td>'+esc(r.name)+'</td><td>'+esc(r.value)+'</td><td>'+esc(r.lang)+'</td></tr>';}
  document.getElementById('rows').innerHTML=html;
 }).catch(function(){}).then(function(){setTimeout(tick,700);});
}
tick();
</script></body></html>
""".trimIndent()
    }
}
