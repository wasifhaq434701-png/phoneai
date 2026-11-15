package org.membench.bench

import android.app.ActivityManager
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.PowerManager
import android.os.SystemClock
import android.provider.Settings
import java.io.File
import org.json.JSONObject

/** Device readings. Android has no per-app memory limit like iOS jetsam: available memory is system-wide. */
class Probe(private val ctx: Context) {
    private val am = ctx.getSystemService(ActivityManager::class.java)
    private val pm = ctx.getSystemService(PowerManager::class.java)
    private val bm = ctx.getSystemService(BatteryManager::class.java)

    private fun memInfo() = ActivityManager.MemoryInfo().also { am.getMemoryInfo(it) }
    fun available(): Long = memInfo().availMem

    fun thermal(): String = when (pm.currentThermalStatus) {
        PowerManager.THERMAL_STATUS_NONE -> "none"
        PowerManager.THERMAL_STATUS_LIGHT -> "light"
        PowerManager.THERMAL_STATUS_MODERATE -> "moderate"
        PowerManager.THERMAL_STATUS_SEVERE -> "severe"
        PowerManager.THERMAL_STATUS_CRITICAL -> "critical"
        PowerManager.THERMAL_STATUS_EMERGENCY -> "emergency"
        PowerManager.THERMAL_STATUS_SHUTDOWN -> "shutdown"
        else -> "unknown"
    }

    /** Forecast 0 s ahead: 1.0 = throttling starts. NaN on phones without support (stored as null). */
    fun headroom(): Float = pm.getThermalHeadroom(0)

    private fun batteryIntent(): Intent? = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))

    data class Battery(val level: Float, val state: String, val plugged: Boolean)

    /** Battery temperature in °C; NaN if the phone doesn't report it. */
    fun batteryTempC(): Double = batteryIntent()?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, Int.MIN_VALUE)
        ?.takeIf { it != Int.MIN_VALUE }?.div(10.0) ?: Double.NaN

    fun battery(): Battery {
        val i = batteryIntent()
        val level = (i?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1).toFloat() /
            (i?.getIntExtra(BatteryManager.EXTRA_SCALE, 100) ?: 100)
        val state = when (i?.getIntExtra(BatteryManager.EXTRA_STATUS, -1)) {
            BatteryManager.BATTERY_STATUS_CHARGING -> "charging"
            BatteryManager.BATTERY_STATUS_DISCHARGING -> "unplugged"
            BatteryManager.BATTERY_STATUS_NOT_CHARGING -> "not_charging"
            BatteryManager.BATTERY_STATUS_FULL -> "full"
            else -> "unknown"
        }
        return Battery(level, state, (i?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) ?: 0) != 0)
    }

    /** One battery trace row (see Energy.kt). Unsupported properties come back as MIN_VALUE, or 0 on some phones:
     *  both are stored as missing (a real 0 µA under load doesn't happen). */
    fun batteryRow(): LongArray {
        val i = batteryIntent()
        fun prop(id: Int) = bm.getLongProperty(id).let { if (it == Long.MIN_VALUE || it == Int.MIN_VALUE.toLong() || it == 0L) NA else it }
        return longArrayOf(
            SystemClock.elapsedRealtime(),
            prop(BatteryManager.BATTERY_PROPERTY_CHARGE_COUNTER),
            prop(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW),
            i?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, -1)?.takeIf { it > 0 }?.toLong() ?: NA,
            prop(BatteryManager.BATTERY_PROPERTY_ENERGY_COUNTER),
            i?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1)?.toLong() ?: NA,
            i?.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0)?.toLong() ?: NA)
    }

    fun airplaneMode() = Settings.Global.getInt(ctx.contentResolver, Settings.Global.AIRPLANE_MODE_ON, 0) == 1

    /** Same keys as the iOS `device` object, plus Android-only ones. */
    fun device(): JSONObject {
        val b = battery()
        val m = memInfo()
        return JSONObject()
            .put("machine", Build.MODEL)
            .put("os", "Android ${Build.VERSION.RELEASE} (SDK ${Build.VERSION.SDK_INT})")
            .put("ram_bytes", m.totalMem)
            .put("available_memory_at_start", m.availMem)
            .put("low_power_mode", pm.isPowerSaveMode)
            .put("battery_level", b.level.toDouble())
            .put("battery_state", b.state)
            .put("thermal_state_at_start", thermal())
            .put("manufacturer", Build.MANUFACTURER)
            .put("soc", "${Build.SOC_MANUFACTURER} ${Build.SOC_MODEL}")
            .put("cpu_cores", Runtime.getRuntime().availableProcessors())
            .put("airplane_mode", airplaneMode())
    }

    /** Label for the conditions box on screen. */
    fun summary(): String {
        val d = device()
        return "Device: ${d["machine"]} (${d["soc"]}), ${d["os"]}\n" +
            "RAM ${gb(d.getLong("ram_bytes"))}, available ${gb(d.getLong("available_memory_at_start"))}\n" +
            "Battery ${(d.getDouble("battery_level") * 100).toInt()}%, ${d["battery_state"]}\n" +
            "Thermal ${d["thermal_state_at_start"]}, power saver ${if (d.getBoolean("low_power_mode")) "ON — turn it off" else "off"}, " +
            "airplane mode ${if (d.getBoolean("airplane_mode")) "on" else "OFF"}"
    }

    companion object {
        /** Cores whose max clock is at least 70% of the fastest one: leaves out the little cores, which slow
         *  llama.cpp's threads down (all cores on a phone with one cluster). */
        fun fastCores(): Int {
            val max = (0 until Runtime.getRuntime().availableProcessors()).map {
                File("/sys/devices/system/cpu/cpu$it/cpufreq/cpuinfo_max_freq").let { f -> if (f.canRead()) f.readText().trim().toLong() else 0L }
            }
            return max.count { it >= 0.7 * max.max() }.coerceAtLeast(1)
        }

        /** Resident set size of this process (bytes), from /proc/self/status. */
        fun rss(): Long = File("/proc/self/status").useLines { lines ->
            lines.first { it.startsWith("VmRSS:") }.filter { it.isDigit() }.toLong() * 1024
        }
    }
}

fun gb(bytes: Long) = "%.2f GB".format(bytes / 1e9)

/** Calls `f` every `periodMs` on its own thread until stopped. */
class Ticker(private val periodMs: Long, private val f: () -> Unit) {
    @Volatile private var on = true
    private val thread = Thread {
        while (on) {
            f()
            Thread.sleep(periodMs)
        }
    }.apply { isDaemon = true; start() }

    fun stop() { on = false; thread.join() }
}

/** Highest RSS seen, sampled every 100 ms. */
class PeakSampler {
    private var peak = 0L
    private val ticker = Ticker(100) { sample() }

    @Synchronized private fun sample() { peak = maxOf(peak, Probe.rss()) }

    /** Peak since the last call (including a reading now), then resets. */
    @Synchronized fun takePeak(): Long { sample(); return peak.also { peak = 0 } }

    fun stop() = ticker.stop()
}
