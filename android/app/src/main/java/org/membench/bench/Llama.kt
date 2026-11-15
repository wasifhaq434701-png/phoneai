package org.membench.bench

/** llama.cpp through bench.cpp. */
object Llama {
    init { System.loadLibrary("bench") }

    /** Returns a model handle; throws if the file can't be loaded. */
    external fun load(path: String): Long
    external fun free(model: Long)
    external fun systemInfo(): String

    /** Greedy, fresh KV, at most nNew tokens (exactly nNew unless stopAtEos). `times` (size nNew + 1) receives
     *  [start, token 1, …] in elapsedRealtimeNanos; entries past the returned tokens stay 0. */
    external fun generate(model: Long, ids: IntArray, nNew: Int, threads: Int, times: LongArray, stopAtEos: Boolean): IntArray
}
