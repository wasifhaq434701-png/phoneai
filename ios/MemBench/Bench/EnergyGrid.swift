import Foundation
import UIKit

/// Energy grid for the phone's power telemetry (experiments/iphone_power.py, read from the Mac): one prompt per budget ×
/// forced output 16/64/256 tokens, each cell repeated for at least `gridMinCellS` (the telemetry publishes every ~20 s,
/// so a cell spans ≥5 updates), `gridPasses` passes in `gridOrder`. Cell start and end are unix times on the phone's
/// clock, the same clock as the telemetry. Output: Documents/grid-<machine>-<time>.json.
struct GridFile: Codable {
    var machine, model: String
    var passes: Int
    var minCellS: Double
    var newTokens: [Int]
    var inProgress: String?
    var cells: [Cell] = []

    struct Cell: Codable {
        var pass, budget, promptTok, nNew, reps: Int
        var startT, endT: Double  // unix seconds
        var ttftS, decodeTokS: Double  // medians over the reps
        var thermalState, batteryState: String
    }
}

extension BenchRunner {
    static let gridNewTokens = [16, 64, 256]
    static let gridPasses = 3
    static let gridMinCellS = 100.0

    /// Cell order, same as the Android app: budgets rotate every cell and output lengths shift every row, so each
    /// budget meets each output length once per pass; pass p starts `budgets.count × (p − 1)` cells later.
    nonisolated static func gridOrder(budgets: [Int], news: [Int], pass: Int) -> [(budget: Int, nNew: Int)] {
        let nb = budgets.count, n = nb * news.count
        let cells = (0..<n).map { (budgets[$0 % nb], news[($0 + $0 / nb) % news.count]) }
        return (0..<n).map { cells[($0 + nb * (pass - 1)) % n] }
    }

    func grid(model: String) {
        running = true
        log = []
        UIApplication.shared.isIdleTimerDisabled = true
        Task.detached(priority: .userInitiated) {
            await self.runGrid(model: model)
            await MainActor.run {
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    private nonisolated func runGrid(model name: String) async {
        let url = URL.documentsDirectory.appending(component: "grid-\(Probe.machine)-\(Int(Date().timeIntervalSince1970)).json")
        func save(_ f: GridFile) throws { try BenchRun.encoder.encode(f).write(to: url, options: .atomic) }
        let median = { (x: [Double]) in x.sorted()[x.count / 2] }
        do {
            let items = try BenchRun.decoder.decode(PromptFile.self, from: Data(contentsOf: Self.promptsURL)).items
                .sorted { ($0.budget, $0.qid) < ($1.budget, $1.qid) }
            var prompt: [Int: PromptFile.Item] = [:]
            for i in items where prompt[i.budget] == nil { prompt[i.budget] = i }  // first question of each budget
            let engine = MLXEngine()
            await say("loading \(name) …")
            try await engine.load(directory: Self.modelsDir.appending(component: name))
            var file = GridFile(machine: Probe.machine, model: name, passes: Self.gridPasses, minCellS: Self.gridMinCellS,
                                newTokens: Self.gridNewTokens)
            await MainActor.run { resultURL = url }
            let sampler = PeakSampler()
            _ = try Self.measure(engine, items[0], kvBits: nil, sampler: sampler)  // warm-up

            for pass in 1...Self.gridPasses {
                for (budget, n) in Self.gridOrder(budgets: prompt.keys.sorted(), news: Self.gridNewTokens, pass: pass) {
                    let item = prompt[budget]!
                    file.inProgress = "pass \(pass) \(budget) × \(n)"
                    try save(file)
                    var ttft: [Double] = [], decode: [Double] = []
                    let start = Date().timeIntervalSince1970
                    while Date().timeIntervalSince1970 - start < Self.gridMinCellS {
                        let r = try Self.measure(engine, item, kvBits: nil, sampler: sampler, maxTokens: n)
                        ttft.append(r.rates.ttft)
                        decode.append(r.rates.decode)
                    }
                    let battery = await Probe.battery().state
                    file.cells.append(.init(pass: pass, budget: budget, promptTok: item.promptTok, nNew: n, reps: ttft.count,
                                            startT: start, endT: Date().timeIntervalSince1970, ttftS: median(ttft),
                                            decodeTokS: median(decode), thermalState: Probe.thermal(), batteryState: battery))
                    let c = file.cells[file.cells.count - 1]
                    await say(String(format: "pass %d %d × %d: %d reps in %.0f s, ttft %.2f s, decode %.1f tok/s, %@, %@", pass,
                                     budget, n, c.reps, c.endT - c.startT, c.ttftS, c.decodeTokS, c.thermalState, c.batteryState))
                }
            }
            file.inProgress = nil
            try save(file)
            await say("grid complete: \(url.lastPathComponent)")
            engine.unload()
        } catch {
            await say("grid FAILED: \(error)")
        }
    }
}
