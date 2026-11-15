import CryptoKit
import Foundation
import MLX
import UIKit

/// Phase 0 protocol: load → 1 discarded warm-up → all items in `runOrder` → the first 2K item in a loop
/// for `sustainedS`. The JSON is rewritten after every call, so a jetsam kill keeps everything before it.
/// Weights and prompts.json are pushed into Documents from the Mac with ios/push_assets.sh.
@Observable @MainActor
final class BenchRunner {
    var log: [String] = []
    var running = false
    var resultURL: URL?

    static let modelsDir = URL.documentsDirectory.appending(component: "models")
    static let promptsURL = URL.documentsDirectory.appending(component: "prompts.json")
    /// MLX keeps freed buffers for reuse; capped so the footprint reflects the model, not the pool (Mac: uncapped).
    static let cacheLimit = 256 * 1024 * 1024
    static let newTokens = 128

    static func models() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: modelsDir.path)) ?? []).sorted()
    }

    /// `repeats` runs, each a complete run with its own file; from the second on, it first waits for the phone to cool
    /// down (`cooldownDone`) so every run starts from the same thermal state, without anyone touching the phone.
    func start(model: String, kvBits: Int?, sustainedS: Double, maxBudget: Int, repeats: Int = 1) {
        running = true
        log = []
        UIApplication.shared.isIdleTimerDisabled = true
        Task.detached(priority: .userInitiated) {
            for i in 1...repeats {
                var series = BenchRun.Series(index: i, of: repeats, cooldownS: 0, cooled: true)
                if i > 1 { (series.cooldownS, series.cooled) = await self.coolDown(before: i, of: repeats) }
                await self.execute(model: model, kvBits: kvBits, sustainedS: sustainedS, maxBudget: maxBudget,
                                   series: repeats > 1 ? series : nil)
            }
            await MainActor.run {
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    func say(_ line: String) { log.append(line) }

    static let cooldownMinS = 15 * 60.0, cooldownMaxS = 60 * 60.0

    /// Between repeats: wait at least `cooldownMinS` (iOS reports only 4 coarse thermal levels and no temperature, so a
    /// phone can still be warm at "nominal"), then until "nominal". true = cooled, false = gave up at `cooldownMaxS`,
    /// nil = keep waiting.
    nonisolated static func cooldownDone(elapsedS: Double, thermal: String) -> Bool? {
        if elapsedS >= cooldownMaxS { return false }
        return elapsedS >= cooldownMinS && thermal == "nominal" ? true : nil
    }

    private nonisolated func coolDown(before i: Int, of n: Int) async -> (Double, Bool) {
        let t0 = ProcessInfo.processInfo.systemUptime
        var lastSay = -Double.infinity
        while true {
            let e = ProcessInfo.processInfo.systemUptime - t0
            if let cooled = Self.cooldownDone(elapsedS: e, thermal: Probe.thermal()) {
                await say(String(format: "cool-down %.0f min (%@) → run %d/%d", e / 60, cooled ? "nominal" : "gave up", i, n))
                return (e, cooled)
            }
            if e - lastSay >= 300 {
                lastSay = e
                await say(String(format: "cooling down before run %d/%d: %.0f min, %@", i, n, e / 60, Probe.thermal()))
            }
            try? await Task.sleep(for: .seconds(30))
        }
    }

    /// One fresh-KV call of `maxTokens` tokens (EOS ignored): wall-clock start (to align with a power trace), rates, output ids and
    /// memory peaks (footprint sampled every 100 ms).
    nonisolated static func measure(_ engine: MLXEngine, _ item: PromptFile.Item, kvBits: Int?, sampler: PeakSampler,
                                    maxTokens: Int = newTokens) throws
        -> (startedAt: Date, rates: (ttft: Double, prefill: Double, decode: Double), ids: [Int], peakFootprint: UInt64,
            mlxPeak: UInt64)
    {
        let now = { ProcessInfo.processInfo.systemUptime }
        Memory.clearCache()
        Memory.peakMemory = 0
        _ = sampler.takePeak()
        var times: [Double] = [], ids: [Int] = []
        let startedAt = Date()
        let start = now()
        try engine.generate(promptIDs: item.ids, maxTokens: maxTokens, kvBits: kvBits) {
            times.append(now())
            ids.append($0)
            return true
        }
        return (startedAt, rates(start: start, tokenTimes: times, promptTokens: item.promptTok), ids, sampler.takePeak(),
                UInt64(Memory.peakMemory))
    }

    /// Interleaved order, identical on every device (Android app, Jetson script): question by question, the budgets
    /// rotated one step per question, so each budget is spread evenly over the run and over the phone warming up.
    /// Smallest-budget-first (the 23 Oct runs) put every 8K call on the hottest phone.
    nonisolated static func runOrder(_ items: [PromptFile.Item]) -> [PromptFile.Item] {
        Dictionary(grouping: items, by: \.qid).sorted { $0.key < $1.key }.enumerated().flatMap { i, q in
            let b = q.value.sorted { $0.budget < $1.budget }
            return b.indices.map { b[($0 + i) % b.count] }
        }
    }

    /// `maxBudget`: run only items up to this budget (0 = warm-up + sustained only).
    private nonisolated func execute(model name: String, kvBits: Int?, sustainedS: Double, maxBudget: Int,
                                     series: BenchRun.Series? = nil) async {
        let device = await Probe.device()  // before loading: available memory = the app's real limit
        let url = URL.documentsDirectory.appending(
            component: "bench-\(device.machine)-\(name)\(kvBits.map { "-kv\($0)" } ?? "")-\(Int(Date().timeIntervalSince1970)).json")
        func save(_ run: BenchRun) throws { try BenchRun.encoder.encode(run).write(to: url, options: .atomic) }
        let now = { ProcessInfo.processInfo.systemUptime }

        do {
            let promptData = try Data(contentsOf: Self.promptsURL)
            let prompts = try BenchRun.decoder.decode(PromptFile.self, from: promptData)
            let items = prompts.items.sorted { ($0.budget, $0.qid) < ($1.budget, $1.qid) }
            guard let sustainedItem = items.first(where: { $0.budget == 2000 }) else { throw CocoaError(.fileReadCorruptFile) }

            let dir = Self.modelsDir.appending(component: name)
            let source = (try? String(contentsOf: dir.appending(component: "source.txt"), encoding: .utf8))?
                .trimmingCharacters(in: .whitespacesAndNewlines).split(separator: "@").map(String.init) ?? [name]
            Memory.cacheLimit = Self.cacheLimit
            let engine = MLXEngine()
            await say("loading \(source[0]) …")
            let t0 = now()
            try await engine.load(directory: dir)
            let loadS = now() - t0

            var run = BenchRun(
                device: device,
                model: .init(id: source[0], revision: source.count > 1 ? source[1] : nil, loadS: loadS,
                             footprintAfterLoad: Probe.footprint()),
                runtime: .init(kvBits: kvBits, mlxCacheLimit: Self.cacheLimit),
                prompts: .init(sha256: SHA256.hash(data: promptData).map { String(format: "%02x", $0) }.joined(),
                               source: prompts.source),
                sustained: .init(qid: sustainedItem.qid, budget: sustainedItem.budget, targetS: sustainedS),
                series: series)
            await say(String(format: "loaded in %.1f s, footprint %@", loadS, gb(run.model.footprintAfterLoad)))
            await MainActor.run { resultURL = url }

            let sampler = PeakSampler()
            let call = { (item: PromptFile.Item) in try Self.measure(engine, item, kvBits: kvBits, sampler: sampler) }

            do {
                for (phase, item) in [("warmup", items[0])] + Self.runOrder(items.filter { $0.budget <= maxBudget }).map({ ("item", $0) }) {
                    run.inProgress = "\(phase) \(item.qid)@\(item.budget)"
                    try save(run)
                    let r = try call(item)
                    let battery = await Probe.battery().level
                    run.calls.append(.init(
                        phase: phase, qid: item.qid, budget: item.budget, promptTok: item.promptTok, startedAt: r.startedAt,
                        ttftS: r.rates.ttft, prefillTokS: r.rates.prefill, decodeTokS: r.rates.decode,
                        peakFootprint: r.peakFootprint, mlxPeak: r.mlxPeak,
                        availableMemoryAfter: Probe.available(), thermalState: Probe.thermal(), batteryLevel: battery,
                        outIds: r.ids))
                    let c = run.calls[run.calls.count - 1]
                    await say(String(format: "%@ %d/%d %d ttft %.2fs prefill %.0f decode %.1f tok/s peak %@ %@",
                                     phase, run.calls.count - 1, items.count, item.budget, c.ttftS, c.prefillTokS,
                                     c.decodeTokS, gb(c.peakFootprint), c.thermalState))
                }

                let s0 = now()
                while now() - s0 < sustainedS {
                    run.inProgress = "sustained \(run.sustained.iterations.count)"
                    let start = now()
                    let r = try call(sustainedItem)
                    let b = await Probe.battery()
                    run.sustained.iterations.append(.init(
                        tS: start - s0, startedAt: r.startedAt, ttftS: r.rates.ttft, decodeTokS: r.rates.decode,
                        thermalState: Probe.thermal(), batteryLevel: b.level, batteryState: b.state))
                    try save(run)
                    await say(String(format: "sustained %.0f/%.0f s ttft %.2fs decode %.1f tok/s %@", start - s0 + r.rates.ttft,
                                     sustainedS, r.rates.ttft, r.rates.decode, Probe.thermal()))
                }
                run.status = "complete"
            } catch {
                run.status = "failed"
                run.error = "\(error)"
            }
            run.inProgress = nil
            run.finishedAt = Date()
            try save(run)
            await say("\(run.status): \(url.lastPathComponent)")
            engine.unload()
        } catch {
            await say("FAILED before the run started: \(error)")
        }
    }
}

func gb(_ bytes: UInt64) -> String { String(format: "%.2f GB", Double(bytes) / 1e9) }
