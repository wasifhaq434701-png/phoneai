package org.membench.bench

import org.junit.Assert.assertEquals
import org.junit.Test

class RunOrderTest {
    // same fixture and expected order as BenchTests.testRunOrderInterleavesBudgets (iOS)
    @Test fun interleavesBudgets() {
        val items = listOf("c", "a", "b").flatMap { q -> listOf(8000, 1000, 4000, 2000).map { Runner.Item(q, it, 0, IntArray(0)) } }
        val got = runOrder(items).map { "${it.qid}${it.budget / 1000}" }
        assertEquals(listOf("a1", "a2", "a4", "a8", "b2", "b4", "b8", "b1", "c4", "c8", "c1", "c2"), got)
    }

    @Test fun resumeKeepsTheUninterruptedOrder() {
        val items = listOf("c", "a", "b").flatMap { q -> listOf(8000, 1000, 4000, 2000).map { Runner.Item(q, it, 0, IntArray(0)) } }
        val full = runOrder(items).map { "${it.qid}${it.budget / 1000}" }
        val resumed = fromQuestion(runOrder(items), 2).map { "${it.qid}${it.budget / 1000}" }
        assertEquals(full.drop(4), resumed)  // question 2 ("b") on, same rotation as without the stop
        assertEquals(full, fromQuestion(runOrder(items), 1).map { "${it.qid}${it.budget / 1000}" })
        assertEquals(3, questionNumber(items, "c"))
    }

    @Test fun gridOrderCoversEveryCellOncePerPass() {
        // same expectations as BenchTests.testGridOrderCoversEveryCellOncePerPass (iOS)
        val p1 = gridOrder(listOf(1, 2, 4, 8), listOf(16, 64, 256), 1).map { "${it.first}x${it.second}" }
        assertEquals(listOf("1x16", "2x64", "4x256", "8x16", "1x256", "2x16", "4x64", "8x256", "1x64", "2x256", "4x16", "8x64"), p1)
        val p2 = gridOrder(listOf(1, 2, 4, 8), listOf(16, 64, 256), 2).map { "${it.first}x${it.second}" }
        assertEquals("1x256", p2.first())
        assertEquals(p1.toSet(), p2.toSet())
    }

    @Test fun cooldownWaitsForMinimumThenCoolPhone() {
        assertEquals(null, cooldownDone(60.0, "none", 30.0, 30.0))           // too early
        assertEquals(null, cooldownDone(20 * 60.0, "light", 30.0, 30.0))     // still throttling
        assertEquals(null, cooldownDone(20 * 60.0, "none", 32.0, 30.0))      // battery still warm
        assertEquals(true, cooldownDone(15 * 60.0, "none", 31.0, 30.0))
        assertEquals(true, cooldownDone(15 * 60.0, "none", Double.NaN, 30.0)) // no temperature: thermal status only
        assertEquals(false, cooldownDone(60 * 60.0, "severe", 40.0, 30.0))   // give up, record it
    }
}
