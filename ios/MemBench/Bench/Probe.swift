import Foundation
import UIKit
import os

/// Device readings for the benchmark. iOS gives no temperature: thermal state + throughput over time is the proxy.
enum Probe {
    /// phys_footprint: the number jetsam compares against the per-app limit.
    static func footprint() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    /// Bytes left before this app hits its jetsam limit.
    static func available() -> UInt64 { UInt64(os_proc_available_memory()) }

    static let machine: String = {
        var u = utsname()
        uname(&u)
        return withUnsafeBytes(of: u.machine) { String(decoding: $0.prefix { $0 != 0 }, as: UTF8.self) }
    }()

    static func thermal() -> String {
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: "nominal"
        case .fair: "fair"
        case .serious: "serious"
        case .critical: "critical"
        @unknown default: "unknown"
        }
    }

    /// Needs `UIDevice.current.isBatteryMonitoringEnabled = true` (set at app launch).
    @MainActor static func battery() -> (level: Float, state: String) {
        let d = UIDevice.current
        let state = switch d.batteryState {
        case .charging: "charging"
        case .full: "full"
        case .unplugged: "unplugged"
        default: "unknown"
        }
        return (d.batteryLevel, state)
    }

    @MainActor static func device() -> BenchRun.Device {
        let b = battery()
        return .init(machine: machine, os: "iOS \(UIDevice.current.systemVersion)",
                     ramBytes: ProcessInfo.processInfo.physicalMemory, availableMemoryAtStart: available(),
                     lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                     batteryLevel: b.level, batteryState: b.state, thermalStateAtStart: thermal())
    }
}

/// Highest phys_footprint seen, sampled every 100 ms on a background queue.
final class PeakSampler: @unchecked Sendable {
    private let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
    private let lock = NSLock()
    private var peak: UInt64 = 0

    init() {
        timer.schedule(deadline: .now(), repeating: .milliseconds(100))
        timer.setEventHandler { [weak self] in self?.sample() }
        timer.resume()
    }

    deinit { timer.cancel() }

    private func sample() {
        let f = Probe.footprint()
        lock.withLock { peak = max(peak, f) }
    }

    /// Peak since the last call (includes a reading taken now), then resets.
    func takePeak() -> UInt64 {
        sample()
        return lock.withLock {
            defer { peak = 0 }
            return peak
        }
    }
}
