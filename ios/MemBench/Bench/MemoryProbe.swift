import Foundation
import MLX
import UIKit

/// 8K on the phone: the first 8K item under each setting (prefill step × MLX cache limit), after a 1K warm-up.
/// An MLX error (Metal out of memory, GPU watchdog) is recorded for that call and the probe moves on.
/// Output: Documents/probe8k-<machine>-<time>.json (diagnostic, not the bench schema).
struct ProbeFile: Codable {
    var machine: String
    var footprintBeforeLoad, footprintAfterLoad: UInt64
    var inProgress: String?
    var calls: [Call] = []

    struct Call: Codable {
        var prefillStep, cacheLimit: Int
        var error: String?
        var ttftS, decodeTokS: Double?
        var peakFootprint, mlxPeak: UInt64
        var thermalState: String
    }
}

extension BenchRunner {
    /// Small chunks bound the longest GPU submissions (A14: ~0.3 s per 2048-row matmul); 2048 is the failing control.
    static let probeSettings = [(512, 256 << 20), (512, 256 << 20), (512, 0), (512, 0), (2048, 256 << 20)]

    func probe(model: String) {
        running = true
        log = []
        UIApplication.shared.isIdleTimerDisabled = true
        Task.detached(priority: .userInitiated) {
            await self.runProbe(model: model)
            await MainActor.run {
                self.running = false
                UIApplication.shared.isIdleTimerDisabled = false
            }
        }
    }

    private nonisolated func runProbe(model name: String) async {
        let url = URL.documentsDirectory.appending(component: "probe8k-\(Probe.machine)-\(Int(Date().timeIntervalSince1970)).json")
        func save(_ f: ProbeFile) throws { try BenchRun.encoder.encode(f).write(to: url, options: .atomic) }
        do {
            let items = try BenchRun.decoder.decode(PromptFile.self, from: Data(contentsOf: Self.promptsURL)).items
            guard let warmup = items.first(where: { $0.budget == 1000 }), let item = items.first(where: { $0.budget == 8000 })
            else { throw CocoaError(.fileReadCorruptFile) }
            let before = Probe.footprint()
            let engine = MLXEngine()
            try await engine.load(directory: Self.modelsDir.appending(component: name))
            var file = ProbeFile(machine: Probe.machine, footprintBeforeLoad: before, footprintAfterLoad: Probe.footprint())
            await MainActor.run { resultURL = url }
            await say("footprint before load \(gb(before)), after \(gb(file.footprintAfterLoad))")

            let sampler = PeakSampler()
            engine.prefillStep = 512
            _ = try Self.measure(engine, warmup, kvBits: nil, sampler: sampler)
            for (step, limit) in Self.probeSettings {
                engine.prefillStep = step
                Memory.cacheLimit = limit
                file.inProgress = "step \(step) cache \(limit >> 20) MB"
                try save(file)
                var call = ProbeFile.Call(prefillStep: step, cacheLimit: limit, peakFootprint: 0, mlxPeak: 0, thermalState: "")
                do {
                    let r = try Self.measure(engine, item, kvBits: nil, sampler: sampler)
                    (call.ttftS, call.decodeTokS, call.peakFootprint, call.mlxPeak) = (r.rates.ttft, r.rates.decode, r.peakFootprint, r.mlxPeak)
                } catch {
                    call.error = "\(error)"
                    (call.peakFootprint, call.mlxPeak) = (sampler.takePeak(), UInt64(Memory.peakMemory))
                }
                call.thermalState = Probe.thermal()
                file.calls.append(call)
                await say(call.error.map { "step \(step) cache \(limit >> 20) MB: FAILED \($0)" } ?? String(
                    format: "step %d cache %d MB: ttft %.0f s, decode %.1f tok/s, peak %@ (MLX %@), %@", step, limit >> 20,
                    call.ttftS ?? 0, call.decodeTokS ?? 0, gb(call.peakFootprint), gb(call.mlxPeak), call.thermalState))
            }
            file.inProgress = nil
            try save(file)
            await say("probe complete: \(url.lastPathComponent)")
        } catch {
            await say("probe FAILED: \(error)")
        }
    }
}
