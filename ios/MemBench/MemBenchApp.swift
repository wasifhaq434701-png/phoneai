import SwiftUI

@main
struct MemBenchApp: App {
    init() {
        UIDevice.current.isBatteryMonitoringEnabled = true
        // Weights are re-downloadable: keep them out of iCloud/device backups.
        var models = BenchRunner.modelsDir
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try? models.setResourceValues(values)
    }

    var body: some Scene {
        WindowGroup { BenchView() }
    }
}
