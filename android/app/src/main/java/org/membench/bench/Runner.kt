package org.membench.bench

import android.content.Context
import android.os.SystemClock
import java.io.File
import java.io.InputStream
import java.security.MessageDigest
import java.time.Instant
import java.util.UUID
import kotlin.math.roundToInt
import org.json.JSONArray
import org.json.JSONObject

const val NEW_TOKENS = 128
const val ANSWER_MAX = 200  // "natural answers": stop at the end of the answer, at most this many tokens (the Mac's cap)
const val BRIGHTNESS = 0.5f  // fixed screen brightness during every run (screen power is part of the battery reading)
val GRID_NEW = listOf(16, 64, 256)
const val GRID_PASSES = 3
const val GRID_MIN_S = 3.0  // each grid cell repeats until at least this much measured time
const val COOLDOWN_MIN_S = 15 * 60.0  // same as the iOS app
// Energy runs: start at or below ENERGY_START_MAX % on battery, stop cleanly at ENERGY_STOP %. 90 → 30 holds a full grid
// (~35-70 min at ~7 W uses 20-40% of a ~19 Wh battery); the charge counter is linear well inside this range.
const val ENERGY_START_MAX = 90
const val ENERGY_STOP = 30
const val COOLDOWN_MAX_S = 60 * 60.0

/** Between repeats (as in the iOS app): at least [COOLDOWN_MIN_S], then thermal status "none" and, since Android also
 *  reports it, the battery back within 1 °C of where the series started (NaN = not reported: thermal status only).
 *  true = cooled, false = gave up at [COOLDOWN_MAX_S], null = keep waiting. */
fun cooldownDone(elapsedS: Double, thermal: String, tempC: Double, startTempC: Double): Boolean? = when {
    elapsedS >= COOLDOWN_MAX_S -> false
    elapsedS >= COOLDOWN_MIN_S && thermal == "none" && !(tempC > startTempC + 1.0) -> true
    else -> null
}

/** Energy-grid cell order, identical to the iOS app's BenchRunner.gridOrder: budgets rotate every cell and output
 *  lengths shift every row, so each budget meets each output length once per pass; pass p starts nb × (p − 1) cells later. */
fun gridOrder(budgets: List<Int>, news: List<Int>, pass: Int): List<Pair<Int, Int>> {
    val nb = budgets.size
    val n = nb * news.size
    val cells = (0 until n).map { budgets[it % nb] to news[(it + it / nb) % news.size] }
    return (0 until n).map { cells[(it + nb * (pass - 1)) % n] }
}

/** Interleaved order, identical to the iOS app's BenchRunner.runOrder: question by question, the budgets rotated one
 *  step per question, so each budget is spread evenly over the run and over the phone warming up. */
fun runOrder(items: List<Runner.Item>): List<Runner.Item> =
    items.groupBy { it.qid }.toSortedMap().values.withIndex().flatMap { (i, q) ->
        val b = q.sortedBy { it.budget }
        b.indices.map { b[(it + i) % b.size] }
    }

/** 1-based question number of `qid` in run order (questions are run in sorted qid order). */
fun questionNumber(items: List<Runner.Item>, qid: String) = items.map { it.qid }.distinct().sorted().indexOf(qid) + 1

/** The run order from question `first` on, so a run that needs two charges continues where the first one stopped,
 *  with the same rotation as an uninterrupted run. */
fun fromQuestion(order: List<Runner.Item>, first: Int): List<Runner.Item> =
    order.filter { questionNumber(order, it.qid) >= first }

/** Same protocol as the iOS BenchRunner: load → 1 discarded warm-up (first 1K item) → items up to `maxBudget` in
 *  [runOrder] → the first 2K item in a loop for `sustainedS`. Or the energy grid (Mac calibration, Result 14).
 *  The JSON is rewritten after every call, so a crash keeps everything before it. The battery is sampled every 250 ms. */
class Runner(private val ctx: Context, private val threads: Int, private val say: (String) -> Unit) {
    private val probe = Probe(ctx)

    class Item(val qid: String, val budget: Int, val promptTok: Int, val ids: IntArray)
    private class Stop(val reason: String) : Exception(reason)

    /** Why an energy run (on battery, between [ENERGY_START_MAX]% and [ENERGY_STOP]%) can't start now; null = OK. */
    fun energyGuard(): String? {
        val b = probe.battery()
        val pct = (b.level * 100).roundToInt()
        return when {
            b.plugged -> "Unplug the phone: energy can't be measured while it charges."
            pct > ENERGY_START_MAX -> "Battery at $pct%: energy is measured between $ENERGY_START_MAX% and $ENERGY_STOP%. Use the phone until it shows $ENERGY_START_MAX% or less."
            pct <= ENERGY_STOP -> "Battery at $pct%: energy is measured between $ENERGY_START_MAX% and $ENERGY_STOP%. Charge above $ENERGY_START_MAX%, unplug, wait for $ENERGY_START_MAX%."
            else -> null
        }
    }

    /** `count` complete runs, each with its own file; from the second on it first waits for the phone to cool down,
     *  so every run starts from the same thermal state without anyone touching the phone. */
    fun benchSeries(modelFile: File, maxBudget: Int, sustainedS: Double, energyRun: Boolean, count: Int,
                    natural: Boolean = false, firstQuestion: Int = 1) {
        val startTempC = probe.batteryTempC()
        for (i in 1..count) {
            val (cooldownS, cooled) = if (i > 1) coolDown(i, count, startTempC) else 0.0 to true
            val series = if (count > 1) JSONObject().put("index", i).put("of", count).put("cooldown_s", cooldownS).put("cooled", cooled) else null
            bench(modelFile, maxBudget, sustainedS, energyRun, series, natural, firstQuestion)
        }
    }

    private fun coolDown(i: Int, n: Int, startTempC: Double): Pair<Double, Boolean> {
        val t0 = SystemClock.elapsedRealtime()
        var lastSay = Double.NEGATIVE_INFINITY
        while (true) {
            val e = (SystemClock.elapsedRealtime() - t0) / 1000.0
            cooldownDone(e, probe.thermal(), probe.batteryTempC(), startTempC)?.let {
                say("cool-down %.0f min (%s) → run %d/%d".format(e / 60, if (it) "cool" else "gave up", i, n))
                return e to it
            }
            if (e - lastSay >= 300) {
                lastSay = e
                say("cooling down before run %d/%d: %.0f min, %s, battery %.1f °C (start %.1f)".format(
                    i, n, e / 60, probe.thermal(), probe.batteryTempC(), startTempC))
            }
            Thread.sleep(30_000)
        }
    }

    /** `natural`: items stop at the end of the answer (at most [ANSWER_MAX] tokens), so the judge on the Mac can score
     *  them and each call's energy is a real query's; warm-up and sustained calls stay at a forced [NEW_TOKENS].
     *  `firstQuestion` > 1 continues a run that the battery window stopped (see `next_question` in its file). */
    fun bench(modelFile: File, maxBudget: Int, sustainedS: Double, energyRun: Boolean, series: JSONObject? = null,
              natural: Boolean = false, firstQuestion: Int = 1) =
        session(modelFile, energyRun, if (firstQuestion > 1) "-q$firstQuestion" else "", sustainedS, series) { items ->
            run.put("items", JSONObject().put("new_tokens", if (natural) ANSWER_MAX else NEW_TOKENS).put("stop_at_eos", natural)
                .put("first_question", firstQuestion))
            val order = fromQuestion(runOrder(items.filter { it.budget <= maxBudget }), firstQuestion)
            for ((phase, item) in listOf("warmup" to items[0]) + order.map { "item" to it }) {
                if (phase == "item") run.put("next_question", questionNumber(items, item.qid))
                val c = if (phase == "item" && natural) call(phase, item, ANSWER_MAX, stopAtEos = true) else call(phase, item)
                say("%s %d/%d %d ttft %.2fs prefill %.0f decode %.1f tok/s, %d tokens, peak %s %s".format(phase,
                    calls.length() - 1, order.size, item.budget, c.getDouble("ttft_s"), c.getDouble("prefill_tok_s"),
                    c.optDouble("decode_tok_s"), c.getJSONArray("out_ids").length(), gb(c.getLong("peak_footprint")),
                    c.getString("thermal_state")))
            }
            run.remove("next_question")
            val item = items.first { it.budget == 2000 }
            val iterations = run.getJSONObject("sustained").getJSONArray("iterations")
            val s0 = SystemClock.elapsedRealtime()
            while (SystemClock.elapsedRealtime() - s0 < sustainedS * 1000) {
                val start = SystemClock.elapsedRealtime()
                val c = call("sustained", item, into = iterations)
                val b = probe.battery()
                for (k in listOf("phase", "qid", "budget", "prompt_tok", "out_ids")) c.remove(k)  // already in `sustained`
                c.put("t_s", (start - s0) / 1000.0).put("battery_level", b.level.toDouble()).put("battery_state", b.state)
                save()
                say("sustained %.0f/%.0f s ttft %.2fs decode %.1f tok/s %s".format((start - s0) / 1000.0 + c.getDouble("ttft_s"),
                    sustainedS, c.getDouble("ttft_s"), c.getDouble("decode_tok_s"), c.getString("thermal_state")))
            }
        }

    /** Energy grid: one prompt per budget (1K/2K/4K/8K) × forced output 16/64/256 tokens, for the given passes.
     *  A single pass keeps its own rotated order, so passes 1, 2 and 3 run on separate charges add up to the full grid. */
    fun grid(modelFile: File, energyRun: Boolean, passes: List<Int>) =
        session(modelFile, energyRun, if (passes.size == 1) "-grid-p${passes[0]}" else "-grid", 0.0) { items ->
        run.put("grid", JSONObject().put("passes", passes.size).put("pass_numbers", JSONArray(passes))
            .put("min_cell_s", GRID_MIN_S).put("new_tokens", JSONArray(GRID_NEW)))
        call("warmup", items[0])
        val prompts = items.groupBy { it.budget }.mapValues { it.value.first() }
        for (pass in passes) for ((budget, n) in gridOrder(prompts.keys.sorted(), GRID_NEW, pass)) {
            val item = prompts.getValue(budget)
            var measured = 0.0
            var rep = 0
            while (measured < GRID_MIN_S) {
                val c = call("grid", item, n).put("pass", pass).put("n_new", n).put("rep", rep++)
                measured += c.getDouble("ttft_s") + (n - 1) / c.getDouble("decode_tok_s")
            }
            val e = calls.getJSONObject(calls.length() - 1).optJSONObject("energy")
            say("grid pass %d %d tok × %d new: %d reps, %.1f s, last decode %.1f tok/s%s".format(pass, item.promptTok, n, rep,
                measured, calls.getJSONObject(calls.length() - 1).getDouble("decode_tok_s"),
                e?.optDouble("decode_j")?.takeUnless { it.isNaN() }?.let { ", decode %.1f J".format(it) } ?: ""))
        }
    }

    /** Idle baseline for the energy runs (like the iPhone's): model loaded, nothing running, screen on at [BRIGHTNESS],
     *  on battery, for `seconds`. Mean idle W = idle.energy.decode_j / decode_s (the whole window is the "decode" phase). */
    fun idle(modelFile: File, seconds: Double) = session(modelFile, true, "-idle", 0.0) { measureIdle(seconds) }

    private inner class Session(val file: File, val run: JSONObject, val energyRun: Boolean) {
        val calls: JSONArray = run.getJSONArray("calls")
        var model = 0L
        val rows = ArrayList<LongArray>()
        val battery = Ticker(SAMPLE_MS) { probe.batteryRow().let { synchronized(rows) { rows.add(it) } } }
        val peaks = PeakSampler()
        private val windows = ArrayList<Pair<JSONObject, LongArray>>()  // call record → [start, first token, end] in ms

        /** Energy per call is recomputed from the whole trace at every save, so the last call gets samples after its end. */
        fun save() {
            val trace = synchronized(rows) { rows.toList() }
            for ((rec, w) in windows) rec.put("energy", JSONObject(callEnergy(trace, w[0], w[1], w[2])))
            run.getJSONObject("energy")
                .put("rows", JSONArray(trace.map { r -> JSONArray(r.map { if (it == NA) JSONObject.NULL else it }) }))
                .put("counters", JSONObject(mapOf("charge_uah" to CHARGE, "current_ua" to CURRENT, "voltage_mv" to VOLT,
                    "energy_nwh" to ENERGY).mapValues { JSONObject(counterStatus(trace, it.value)) }))
            val tmp = File(file.path + ".tmp")
            tmp.writeText(run.toString())
            tmp.renameTo(file)
        }

        /** One fresh-KV call → the iOS call record (Android extras: thermal_headroom, energy). */
        fun call(phase: String, item: Item, nNew: Int = NEW_TOKENS, into: JSONArray = calls, stopAtEos: Boolean = false): JSONObject {
            outOfWindow()?.let { throw Stop(it) }
            run.put("in_progress", "$phase ${item.qid}@${item.budget}")
            save()
            peaks.takePeak()
            val buf = LongArray(nNew + 1)
            val wall = System.currentTimeMillis()
            val out = Llama.generate(model, item.ids, nNew, threads, buf, stopAtEos)
            val times = buf.copyOf(out.size + 1)  // an answer that stopped early leaves the rest of `buf` at 0
            val (ttft, prefill, decode) = rates(times, item.promptTok)
            val rec = JSONObject().put("phase", phase).put("qid", item.qid).put("budget", item.budget)
                .put("prompt_tok", item.promptTok).put("started_at", iso(wall)).put("ttft_s", ttft)
                .put("prefill_tok_s", prefill).putOpt("decode_tok_s", decode.takeUnless { it.isNaN() })  // NaN: one-token answer
                .put("peak_footprint", peaks.takePeak())
                .put("available_memory_after", probe.available()).put("thermal_state", probe.thermal())
                .putOpt("thermal_headroom", probe.headroom().takeUnless { it.isNaN() }?.toDouble())
                .put("out_ids", JSONArray(out))
            into.put(rec)
            windows.add(rec to longArrayOf(times[0] / 1_000_000, times[1] / 1_000_000, times.last() / 1_000_000))
            save()
            return rec
        }

        fun measureIdle(seconds: Double) {
            val rec = JSONObject().put("target_s", seconds)
            run.put("idle", rec)
            val w = SystemClock.elapsedRealtime().let { longArrayOf(it, it, it) }  // start = first token = start: all "decode"
            windows.add(rec to w)
            while (true) {
                outOfWindow()?.let { throw Stop(it) }
                w[2] = SystemClock.elapsedRealtime()
                save()
                if (w[2] - w[0] >= seconds * 1000) break
                say("idle %.0f / %.0f s".format((w[2] - w[0]) / 1000.0, seconds))
                Thread.sleep(30_000)
            }
            rec.optJSONObject("energy")?.let { say("idle %.2f W".format(it.optDouble("decode_j") / it.getDouble("decode_s"))) }
        }

        /** An energy run stops when the charger is connected or the battery drops below [ENERGY_STOP]%. */
        private fun outOfWindow(): String? {
            if (!energyRun) return null
            val b = probe.battery()
            return when {
                b.plugged -> "charger connected"
                b.level * 100 < ENERGY_STOP -> "battery below $ENERGY_STOP%"
                else -> null
            }
        }
    }

    private fun session(modelFile: File, energyRun: Boolean, tag: String, sustainedS: Double, series: JSONObject? = null,
                        body: Session.(List<Item>) -> Unit) {
        var s: Session? = null
        try {
            val device = probe.device()  // before loading: available memory without the model
            val promptBytes = ctx.assets.open("prompts.json").use { it.readBytes() }
            val prompts = JSONObject(String(promptBytes))
            val items = prompts.getJSONArray("items").let { a ->
                (0 until a.length()).map { a.getJSONObject(it) }.map { o ->
                    val ids = o.getJSONArray("ids")
                    Item(o.getString("qid"), o.getInt("budget"), o.getInt("prompt_tok"), IntArray(ids.length()) { ids.getInt(it) })
                }
            }.sortedWith(compareBy({ it.budget }, { it.qid }))
            val sustainedItem = items.first { it.budget == 2000 }
            val sha = modelSha(modelFile)
            if (energyRun && !probe.airplaneMode()) say("WARNING: airplane mode is off (radios add to the energy reading)")

            val name = "bench-${device.getString("machine")}-${modelFile.nameWithoutExtension}$tag-${System.currentTimeMillis() / 1000}.json"
            val run = JSONObject().put("schema", "membench/1").put("run_id", UUID.randomUUID().toString())
                .put("status", "running").put("device", device)
                .put("runtime", JSONObject().put("llama_cpp", "b11836").put("backend", "cpu").put("threads", threads).put("fast_cores", Probe.fastCores())
                    .put("prefill_step", 2048).put("n_ctx", "prompt + new tokens").put("load_mode", "read into RAM (no mmap)")
                    .put("flash_attn", "auto").put("kv_type", "f16").put("sampler", "greedy").put("new_tokens", NEW_TOKENS)
                    .put("ignore_eos", true).put("fresh_kv", true).put("memory_metric", "VmRSS").put("system_info", Llama.systemInfo()))
                .put("prompts", JSONObject().put("sha256", hex(MessageDigest.getInstance("SHA-256").digest(promptBytes)))
                    .put("source", prompts.getString("source")))
                .put("calls", JSONArray())
                .putOpt("series", series)  // same keys as iOS: index, of, cooldown_s, cooled
                .put("sustained", JSONObject().put("qid", sustainedItem.qid).put("budget", sustainedItem.budget)
                    .put("target_s", sustainedS).put("iterations", JSONArray()))
                .put("energy", JSONObject().put("energy_run", energyRun).put("sample_ms", SAMPLE_MS)
                    .put("boot_to_wall_ms", System.currentTimeMillis() - SystemClock.elapsedRealtime())
                    .put("brightness", BRIGHTNESS.toDouble()).put("airplane_mode", probe.airplaneMode()).put("columns", JSONArray(COLUMNS)))
            s = Session(File(ctx.getExternalFilesDir(null), name.replace(Regex("[^A-Za-z0-9._,-]"), "_")), run, energyRun)

            say("loading ${modelFile.name} (${threads} threads) …")
            val t0 = SystemClock.elapsedRealtimeNanos()
            s.model = Llama.load(modelFile.path)
            val loadS = (SystemClock.elapsedRealtimeNanos() - t0) / 1e9
            run.put("model", JSONObject().put("id", modelFile.name).put("sha256", sha).put("size_bytes", modelFile.length())
                .put("load_s", loadS).put("footprint_after_load", Probe.rss()))
            run.put("started_at", iso(System.currentTimeMillis()))
            say("loaded in %.1f s, RSS %s".format(loadS, gb(Probe.rss())))

            try {
                s.body(items)
                run.put("status", "complete")
            } catch (e: Stop) {
                run.put("status", "complete").put("stopped", e.reason)
                say("STOPPED: ${e.reason} (energy runs measure on battery between $ENERGY_START_MAX% and $ENERGY_STOP%)" +
                    (run.optInt("next_question").takeIf { it > 0 }?.let { ". To continue: charge, then Start at question $it" } ?: ""))
            } catch (e: Throwable) {
                run.put("status", "failed").put("error", e.toString())
            }
            run.remove("in_progress")
            run.put("finished_at", iso(System.currentTimeMillis()))
            s.save()
            say("${run.getString("status")}: ${s.file.name}")
        } catch (e: Throwable) {
            say("FAILED before the run started: $e")
        } finally {
            s?.run {
                battery.stop()
                peaks.stop()
                if (model != 0L) Llama.free(model)
            }
        }
    }

    /** SHA-256 of the model, cached next to it in `<file>.sha256` (written at import, or here on the first run). */
    private fun modelSha(f: File): String {
        val cache = File(f.path + ".sha256")
        if (cache.exists()) return cache.readText().trim()
        say("hashing ${f.name} (once) …")
        return f.inputStream().use { sha256(it) }.also { cache.writeText(it) }
    }

    companion object {
        /** Newest .gguf in the app's private models folder (imported) or its external one (adb push). */
        fun model(ctx: Context): File? = listOf(File(ctx.filesDir, "models"), File(ctx.getExternalFilesDir(null), "models"))
            .flatMap { it.listFiles { f -> f.name.endsWith(".gguf") }?.toList() ?: emptyList() }
            .maxByOrNull { it.lastModified() }

        /** Streams `input` (into `copyTo` too, if given) and returns its SHA-256. */
        fun sha256(input: InputStream, copyTo: File? = null): String {
            val md = MessageDigest.getInstance("SHA-256")
            val out = copyTo?.outputStream()
            out.use {
                val buf = ByteArray(1 shl 20)
                while (true) {
                    val n = input.read(buf)
                    if (n < 0) break
                    md.update(buf, 0, n)
                    it?.write(buf, 0, n)
                }
            }
            return hex(md.digest())
        }

        private fun hex(b: ByteArray) = b.joinToString("") { "%02x".format(it) }
        fun iso(ms: Long): String = Instant.ofEpochMilli(ms).toString()
    }
}
