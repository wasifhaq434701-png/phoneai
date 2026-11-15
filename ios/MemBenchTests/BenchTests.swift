import XCTest
@testable import MemBench

final class BenchTests: XCTestCase {
    func testRates() {
        // prefill 2 s for 1,000 tokens, then 3 more tokens 0.1 s apart → 10 tok/s decode
        let r = rates(start: 10, tokenTimes: [12, 12.1, 12.2, 12.3], promptTokens: 1000)
        XCTAssertEqual(r.ttft, 2, accuracy: 1e-9)
        XCTAssertEqual(r.prefill, 500, accuracy: 1e-6)
        XCTAssertEqual(r.decode, 10, accuracy: 1e-6)
    }

    func testSchemaKeysAndRoundTrip() throws {
        var run = BenchRun(
            device: .init(machine: "iPhone13,3", os: "iOS 27.0", ramBytes: 6, availableMemoryAtStart: 5,
                          lowPowerMode: false, batteryLevel: 0.5, batteryState: "charging", thermalStateAtStart: "nominal"),
            model: .init(id: "mlx-community/Qwen3-4B-4bit", revision: "abc", loadS: 1, footprintAfterLoad: 2),
            runtime: .init(kvBits: nil, mlxCacheLimit: 3),
            prompts: .init(sha256: "00", source: "lme"),
            sustained: .init(qid: "q", budget: 2000, targetS: 1800))
        run.calls.append(.init(phase: "item", qid: "q", budget: 2000, promptTok: 2156, startedAt: Date(), ttftS: 1, prefillTokS: 2,
                               decodeTokS: 3, peakFootprint: 4, mlxPeak: 5, availableMemoryAfter: 6,
                               thermalState: "fair", batteryLevel: 0.5, outIds: [1, 2]))
        run.sustained.iterations.append(.init(tS: 0, startedAt: Date(), ttftS: 1, decodeTokS: 2, thermalState: "nominal",
                                              batteryLevel: 0.5, batteryState: "charging"))
        run.series = .init(index: 2, of: 3, cooldownS: 900, cooled: true)
        let data = try BenchRun.encoder.encode(run)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(Set(try XCTUnwrap(json["series"] as? [String: Any]).keys), ["index", "of", "cooldown_s", "cooled"])  // = Android

        // Keys Python (experiments/phone_ref.py) and the future Android app rely on.
        XCTAssertEqual(json["schema"] as? String, "membench/1")
        let runtime = try XCTUnwrap(json["runtime"] as? [String: Any])
        XCTAssertEqual(Set(runtime.keys), ["mlx_swift", "mlx_swift_lm", "prefill_step", "mlx_cache_limit", "sampler",
                                           "new_tokens", "ignore_eos", "fresh_kv"])  // kv_bits nil → absent
        let call = try XCTUnwrap((json["calls"] as? [[String: Any]])?.first)
        XCTAssertEqual(Set(call.keys), ["phase", "qid", "budget", "prompt_tok", "started_at", "ttft_s", "prefill_tok_s", "decode_tok_s",
                                        "peak_footprint", "mlx_peak", "available_memory_after", "thermal_state", "battery_level",
                                        "out_ids"])
        let iteration = try XCTUnwrap(((json["sustained"] as? [String: Any])?["iterations"] as? [[String: Any]])?.first)
        XCTAssertEqual(Set(iteration.keys), ["t_s", "started_at", "ttft_s", "decode_tok_s", "thermal_state", "battery_level", "battery_state"])
        XCTAssertNotNil((json["device"] as? [String: Any])?["available_memory_at_start"])
        XCTAssertNotNil((json["model"] as? [String: Any])?["footprint_after_load"])

        let back = try BenchRun.decoder.decode(BenchRun.self, from: data)
        XCTAssertEqual(back.calls[0].outIds, [1, 2])
        XCTAssertEqual(back.model.revision, "abc")
        XCTAssertNil(back.runtime.kvBits)
    }

    func testRunOrderInterleavesBudgets() {
        // same fixture and expected order as RunOrderTest.kt (Android)
        let items = ["c", "a", "b"].flatMap { q in [8000, 1000, 4000, 2000].map {
            PromptFile.Item(qid: q, budget: $0, promptTok: 0, ids: []) } }
        let got = BenchRunner.runOrder(items).map { "\($0.qid)\($0.budget / 1000)" }
        XCTAssertEqual(got, ["a1", "a2", "a4", "a8", "b2", "b4", "b8", "b1", "c4", "c8", "c1", "c2"])
    }

    func testCooldownWaitsForMinimumThenNominal() {
        XCTAssertNil(BenchRunner.cooldownDone(elapsedS: 60, thermal: "nominal"))       // too early, even if nominal
        XCTAssertNil(BenchRunner.cooldownDone(elapsedS: 20 * 60, thermal: "fair"))     // long enough, still warm
        XCTAssertEqual(BenchRunner.cooldownDone(elapsedS: 15 * 60, thermal: "nominal"), true)
        XCTAssertEqual(BenchRunner.cooldownDone(elapsedS: 60 * 60, thermal: "serious"), false)  // give up, record it
    }

    func testGridOrderCoversEveryCellOncePerPass() {
        // same expectations as RunOrderTest.kt (Android)
        let p1 = BenchRunner.gridOrder(budgets: [1, 2, 4, 8], news: [16, 64, 256], pass: 1).map { "\($0.budget)x\($0.nNew)" }
        XCTAssertEqual(p1, ["1x16", "2x64", "4x256", "8x16", "1x256", "2x16", "4x64", "8x256", "1x64", "2x256", "4x16", "8x64"])
        let p2 = BenchRunner.gridOrder(budgets: [1, 2, 4, 8], news: [16, 64, 256], pass: 2).map { "\($0.budget)x\($0.nNew)" }
        XCTAssertEqual(p2.first, "1x256")
        XCTAssertEqual(Set(p2), Set(p1))
    }

    func testPromptFileDecodes() throws {
        let file = #"{"model":"m","protocol":"p","source":"s","items":[{"qid":"a","budget":1000,"mem_tok":9,"prompt_tok":3,"prompt":"x","ids":[1,2,3]}]}"#
        let p = try BenchRun.decoder.decode(PromptFile.self, from: Data(file.utf8))
        XCTAssertEqual(p.items[0].promptTok, 3)
        XCTAssertEqual(p.items[0].ids, [1, 2, 3])
    }
}
