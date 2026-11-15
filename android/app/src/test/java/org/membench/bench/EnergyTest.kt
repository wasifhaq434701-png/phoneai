package org.membench.bench

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class EnergyTest {
    private fun row(t: Long, chargeUah: Long, currentUa: Long, mv: Long, nwh: Long = NA) =
        longArrayOf(t, chargeUah, currentUa, mv, nwh, 75, 0)

    @Test fun rates() {
        // start 0, first token at 2 s, 4 more tokens over 1 s
        val (ttft, prefill, decode) = rates(longArrayOf(0, 2_000_000_000, 2_250_000_000, 2_500_000_000, 2_750_000_000, 3_000_000_000), 1000)
        assertEquals(2.0, ttft, 1e-9)
        assertEquals(500.0, prefill, 1e-9)
        assertEquals(4.0, decode, 1e-9)
    }

    @Test fun integrateInterpolatesWindowEnds() {
        val ramp = listOf(0L to 0.0, 1000L to 10.0, 2000L to 10.0)
        assertEquals(5.0, interp(ramp, 500), 1e-9)
        assertEquals(10.0, interp(ramp, 5000), 1e-9)  // clamped after the last sample
        assertEquals(3.75 + 5.0, integrate(ramp, 500, 1500), 1e-9)  // 0.5 s ramp 5→10 + 0.5 s at 10
    }

    @Test fun callEnergy() {
        // −1 A (discharging, negative sign) at 4 V = 4 W; 1 A for 1 h = 1e6 µAh → 2 s = 555.6 µAh
        val rows = (0..16).map { row(it * 250L, 5_000_000 - (it * 250L * 1_000_000 / 3_600_000), -1_000_000, 4000, 20_000_000_000 - it * 250L * 4 * 1_000_000_000 / 3_600_000) }
        val e = callEnergy(rows, 500, 1000, 3000)
        assertEquals(2.0, e["prefill_j"] as Double, 1e-9)  // 0.5 s × 4 W
        assertEquals(8.0, e["decode_j"] as Double, 1e-9)   // 2 s × 4 W
        assertEquals(555.6, e["decode_uah"] as Double, 1.0)
        assertEquals(8.0, e["decode_counter_j"] as Double, 0.01)
        assertTrue(e["prefill_under_1s"] as Boolean)
        assertFalse(e["decode_under_1s"] as Boolean)
    }

    @Test fun unsupportedCounters() {
        val rows = (0..8).map { row(it * 250L, NA, if (it < 4) -500_000 else -900_000, 3900) }
        val e = callEnergy(rows, 0, 1000, 2000)
        assertFalse("prefill_uah" in e)
        assertFalse("decode_counter_j" in e)
        assertEquals(mapOf("reported" to false, "changes" to 0), counterStatus(rows, CHARGE))
        assertEquals(1, counterStatus(rows, CURRENT)["changes"])
    }
}
