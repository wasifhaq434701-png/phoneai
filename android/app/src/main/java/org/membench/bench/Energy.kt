package org.membench.bench

import kotlin.math.abs

// Pure maths (no Android types), unit-tested in EnergyTest.

/** Per-call rates as on iOS and experiments/phone_ref.py. `times` = [start, token 1, …, token n] in ns.
 *  TTFT includes prefill and the first token; the decode rate is over tokens 2…n. */
fun rates(times: LongArray, promptTok: Int): Triple<Double, Double, Double> {
    val ttft = (times[1] - times[0]) / 1e9
    val decode = (times.size - 2) / ((times.last() - times[1]) / 1e9)
    return Triple(ttft, promptTok / ttft, decode)
}

/** Battery trace row: one sample every SAMPLE_MS. Missing/unsupported values are NA. Time is the boot clock in ms. */
const val T = 0
const val CHARGE = 1   // µAh, BATTERY_PROPERTY_CHARGE_COUNTER
const val CURRENT = 2  // µA, BATTERY_PROPERTY_CURRENT_NOW (sign differs between phones)
const val VOLT = 3     // mV, ACTION_BATTERY_CHANGED EXTRA_VOLTAGE
const val ENERGY = 4   // nWh, BATTERY_PROPERTY_ENERGY_COUNTER
const val LEVEL = 5    // %
const val PLUGGED = 6  // EXTRA_PLUGGED (0 = on battery)
val COLUMNS = listOf("t_ms", "charge_uah", "current_ua", "voltage_mv", "energy_nwh", "level_pct", "plugged")
const val NA = Long.MIN_VALUE
const val SAMPLE_MS = 250L

/** Value at time t, linear between the two nearest points, clamped to the first/last point. Points sorted by time. */
fun interp(pts: List<Pair<Long, Double>>, t: Long): Double {
    if (t <= pts.first().first) return pts.first().second
    if (t >= pts.last().first) return pts.last().second
    val i = pts.indexOfFirst { it.first >= t }
    val (t0, y0) = pts[i - 1]
    val (t1, y1) = pts[i]
    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
}

/** ∫ y dt over [a, b] in y·seconds: trapezoid over the points inside, interpolated at both ends. */
fun integrate(pts: List<Pair<Long, Double>>, a: Long, b: Long): Double {
    val xs = listOf(a to interp(pts, a)) + pts.filter { it.first > a && it.first < b } + listOf(b to interp(pts, b))
    return xs.zipWithNext { (t0, y0), (t1, y1) -> (y0 + y1) / 2 * (t1 - t0) / 1000.0 }.sum()
}

private fun series(rows: List<LongArray>, f: (LongArray) -> Double?) =
    rows.mapNotNull { r -> f(r)?.let { r[T] to it } }

/** Battery power in W = |I| × V (the sign of CURRENT_NOW is not standard across phones). */
fun power(r: LongArray): Double? =
    if (r[CURRENT] == NA || r[VOLT] == NA || r[VOLT] <= 0) null else abs(r[CURRENT]) * 1e-6 * r[VOLT] * 1e-3

/** Energy of one call from the battery trace, prefill = start → first token, decode = first token → last token.
 *  Windows under 1 s are flagged: CURRENT_NOW updates too slowly to resolve them. */
fun callEnergy(rows: List<LongArray>, start: Long, first: Long, end: Long): Map<String, Any> {
    val out = mutableMapOf<String, Any>(
        "t_start_ms" to start, "t_first_ms" to first, "t_end_ms" to end,
        "prefill_s" to (first - start) / 1000.0, "decode_s" to (end - first) / 1000.0,
        "prefill_under_1s" to (first - start < 1000), "decode_under_1s" to (end - first < 1000))
    fun phases(name: String, pts: List<Pair<Long, Double>>, f: (Long, Long) -> Double) {
        if (pts.size < 2) return
        out["prefill_$name"] = f(start, first)
        out["decode_$name"] = f(first, end)
    }
    val watts = series(rows, ::power)
    phases("j", watts) { a, b -> integrate(watts, a, b) }
    val charge = series(rows) { r -> if (r[CHARGE] == NA) null else r[CHARGE].toDouble() }
    phases("uah", charge) { a, b -> interp(charge, a) - interp(charge, b) }
    val counter = series(rows) { r -> if (r[ENERGY] == NA) null else r[ENERGY].toDouble() }
    phases("counter_j", counter) { a, b -> (interp(counter, a) - interp(counter, b)) * 3.6e-6 }  // 1 nWh = 3.6 µJ
    return out
}

/** Does the phone report this counter, and how often does the value change? */
fun counterStatus(rows: List<LongArray>, c: Int): Map<String, Any> {
    val v = rows.filter { it[c] != NA && it[c] != 0L }
    val changes = v.zipWithNext().filter { (a, b) -> a[c] != b[c] }.map { it.second[T] }
    val gaps = changes.zipWithNext { a, b -> (b - a) / 1000.0 }.sorted()
    return buildMap {
        put("reported", v.isNotEmpty())
        put("changes", changes.size)
        if (gaps.isNotEmpty()) put("median_update_s", gaps[gaps.size / 2])
    }
}
