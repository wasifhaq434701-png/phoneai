import SwiftUI

/// Phase 0 benchmark screen (becomes the hidden developer screen in Phase 1).
struct BenchView: View {
    @State private var runner = BenchRunner()
    @State private var models = BenchRunner.models()
    @State private var model = BenchRunner.models().first ?? ""
    @State private var kv8 = false
    @State private var minutes = 30
    @State private var maxBudget = 8000
    @State private var repeats = 3
    @State private var device = Probe.device()
    private let hasPrompts = FileManager.default.fileExists(atPath: BenchRunner.promptsURL.path)

    var body: some View {
        NavigationStack {
            Form {
                Section("Conditions (pull to refresh)") {
                    LabeledContent("Device", value: "\(device.machine), \(device.os)")
                    LabeledContent("RAM", value: gb(device.ramBytes))
                    LabeledContent("Available to app", value: gb(device.availableMemoryAtStart))
                    LabeledContent("Low Power Mode", value: device.lowPowerMode ? "ON — turn it off" : "off")
                    LabeledContent("Battery", value: "\(Int(device.batteryLevel * 100))%, \(device.batteryState)")
                    LabeledContent("Thermal state", value: device.thermalStateAtStart)
                }
                Section("Run") {
                    Picker("Model", selection: $model) {
                        ForEach(models, id: \.self) { Text($0) }
                    }
                    Toggle("8-bit KV cache", isOn: $kv8)
                    Picker("Items", selection: $maxBudget) {
                        Text("None (sustained only)").tag(0)
                        ForEach([1000, 2000, 4000], id: \.self) { Text("Up to \($0 / 1000)K").tag($0) }
                        Text("All (up to 8K)").tag(8000)
                    }
                    Stepper("Sustained run: \(minutes) min", value: $minutes, in: 0...120, step: 5)
                    Stepper("Repeats: \(repeats)", value: $repeats, in: 1...5)
                    Button(runner.running ? "Running…" : "Start") {
                        runner.start(model: model, kvBits: kv8 ? 8 : nil, sustainedS: Double(minutes * 60), maxBudget: maxBudget,
                                     repeats: repeats)
                    }
                    .disabled(runner.running || model.isEmpty || !hasPrompts || device.lowPowerMode)
                    Button("8K probe (6 calls, ~35 min)") { runner.probe(model: model) }
                        .disabled(runner.running || model.isEmpty || !hasPrompts)
                    Button("Energy grid (12 cells × ≥100 s × 3 passes, ~65 min)") { runner.grid(model: model) }
                        .disabled(runner.running || model.isEmpty || !hasPrompts)
                    Text("Energy grid: plugged in at the 80% charge limit (not charging), airplane mode, untouched; record power from the Mac with experiments/iphone_power.py.")
                        .font(.footnote)
                    if repeats > 1 {
                        Text("Between repeats the app waits until the phone is back to nominal, at least 15 min (at most 60). Leave it plugged in and untouched.")
                            .font(.footnote)
                    }
                    if !hasPrompts || models.isEmpty {
                        Text("Push weights and prompts.json from the Mac: ios/push_assets.sh").font(.footnote)
                    }
                    if let url = runner.resultURL {
                        ShareLink("Share results JSON", item: url)
                    }
                }
                Section("Log") {
                    ForEach(runner.log.indices.reversed(), id: \.self) {
                        Text(runner.log[$0]).font(.caption.monospaced())
                    }
                }
            }
            .navigationTitle("MemBench")
            .refreshable {
                device = Probe.device()
                models = BenchRunner.models()
                if model.isEmpty { model = models.first ?? "" }
            }
        }
    }
}
