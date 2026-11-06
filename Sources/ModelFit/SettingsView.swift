import SwiftUI
import ModelFitCore

struct SettingsView: View {
    @AppStorage(PrefKey.kvCache) private var kvRaw = KVCacheType.f16.rawValue
    @AppStorage(PrefKey.customGPULimit) private var customGPU = false
    @AppStorage(PrefKey.gpuPercent) private var gpuPercent = 75.0
    @AppStorage(PrefKey.overheadMB) private var overheadMB = 512

    var body: some View {
        let automatic = MacProfile.current(policy: .automatic)
        let custom = MacProfile.current(policy: .fraction(gpuPercent / 100))
        Form {
            Section {
                Picker("KV cache precision", selection: $kvRaw) {
                    ForEach(KVCacheType.allCases) { Text($0.label).tag($0.rawValue) }
                }
                Stepper(value: $overheadMB, in: 0 ... 8192, step: 128) {
                    LabeledContent("Runtime allowance", value: "\(overheadMB) MB")
                }
                Text("Memory for the inference runtime and Metal context, added to every estimate.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } header: {
                Text("Estimates")
            }

            Section {
                LabeledContent("Automatic limit", value: "\(Format.bytes(automatic.gpuWorkingSet)), \(automatic.gpuLimitSource.label)")
                Toggle("Use a custom limit", isOn: $customGPU)
                if customGPU {
                    Slider(value: $gpuPercent, in: 50 ... 95, step: 1) {
                        Text("Share of memory")
                    } minimumValueLabel: {
                        Text("50%")
                    } maximumValueLabel: {
                        Text("95%")
                    }
                    LabeledContent("Custom limit", value: "\(Int(gpuPercent))% = \(Format.bytes(custom.gpuWorkingSet))")
                }
                Text("macOS lets the GPU wire only part of unified memory. If you raised it with `sudo sysctl iogpu.wired_limit_mb=…`, set the same share here.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            } header: {
                Text("GPU memory limit")
            }

            Section {
                Button("Restore Defaults") {
                    kvRaw = KVCacheType.f16.rawValue
                    customGPU = false
                    gpuPercent = 75
                    overheadMB = 512
                }
            }
        }
        .formStyle(.grouped)
        .frame(width: 480)
        .fixedSize(horizontal: false, vertical: true)
    }
}
