import Foundation

/// Results file `membench/1`. Keys are snake_case on disk; an absent optional key means null.
/// Shared with the future Android app — change only by bumping `schema`.
struct BenchRun: Codable {
    var schema = "membench/1"
    var runId = UUID().uuidString
    var status = "running"  // running | complete | failed
    var startedAt = Date()
    var finishedAt: Date?
    var inProgress: String?  // "phase qid@budget" while a call runs: if the app is killed, this names the culprit
    var error: String?
    var device: Device
    var model: Model
    var runtime: Runtime
    var prompts: Prompts
    var calls: [Call] = []
    var sustained: Sustained
    var series: Series?  // set when the run is one of several started together

    struct Device: Codable {
        var machine, os: String
        var ramBytes, availableMemoryAtStart: UInt64
        var lowPowerMode: Bool
        var batteryLevel: Float
        var batteryState, thermalStateAtStart: String
    }

    struct Model: Codable {
        var id: String
        var revision: String?
        var loadS: Double
        var footprintAfterLoad: UInt64
    }

    struct Runtime: Codable {
        var mlxSwift = "0.32.3", mlxSwiftLm = "3.32.3"  // = the exact pins in project.pbxproj
        var prefillStep = MLXEngine.macPrefillStep
        var kvBits: Int?
        var mlxCacheLimit: Int
        var sampler = "greedy"
        var newTokens = 128
        var ignoreEos = true, freshKv = true
    }

    struct Prompts: Codable {
        var sha256, source: String
    }

    struct Call: Codable {
        var phase: String  // warmup | item
        var qid: String
        var budget, promptTok: Int
        var startedAt: Date  // wall clock, to align calls with an energy trace
        var ttftS, prefillTokS, decodeTokS: Double
        var peakFootprint, mlxPeak, availableMemoryAfter: UInt64
        var thermalState: String
        var batteryLevel: Float  // after the call: coarse energy when no power trace is available
        var outIds: [Int]
    }

    /// Run `index` of `of`, after `cooldownS` of waiting; `cooled` false = the phone never got back to "nominal".
    struct Series: Codable {
        var index, of: Int
        var cooldownS: Double
        var cooled: Bool
    }

    struct Sustained: Codable {
        var qid: String
        var budget: Int
        var targetS: Double
        var iterations: [Iteration] = []
    }

    struct Iteration: Codable {
        var tS: Double
        var startedAt: Date
        var ttftS, decodeTokS: Double
        var thermalState: String
        var batteryLevel: Float
        var batteryState: String
    }

    static let encoder: JSONEncoder = {
        let e = JSONEncoder()
        e.keyEncodingStrategy = .convertToSnakeCase
        e.dateEncodingStrategy = .iso8601
        e.outputFormatting = [.sortedKeys]
        return e
    }()

    static let decoder: JSONDecoder = {
        let d = JSONDecoder()
        d.keyDecodingStrategy = .convertFromSnakeCase
        d.dateDecodingStrategy = .iso8601
        return d
    }()
}

/// One item of data/exp5/phone/prompts.json (exported by the Mac pipeline).
struct PromptFile: Decodable {
    var model, source: String
    var items: [Item]

    struct Item: Decodable {
        var qid: String
        var budget, promptTok: Int
        var ids: [Int]
    }
}

/// Per-call rates. `start` = clock just before prefill; `tokenTimes` = clock when each new token reached the CPU.
/// TTFT includes prefill and the first token; decode rate is over tokens 2…n, as in experiments/phone_ref.py.
func rates(start: Double, tokenTimes: [Double], promptTokens: Int) -> (ttft: Double, prefill: Double, decode: Double) {
    let ttft = tokenTimes[0] - start
    let decode = Double(tokenTimes.count - 1) / (tokenTimes[tokenTimes.count - 1] - tokenTimes[0])
    return (ttft, Double(promptTokens) / ttft, decode)
}
