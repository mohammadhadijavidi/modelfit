import Foundation
#if canImport(Metal)
import Metal
#endif

public enum GPULimitPolicy: Sendable, Equatable {
    /// Use Metal's `recommendedMaxWorkingSetSize`, or the rule of thumb if unavailable.
    case automatic
    /// A fixed fraction of physical memory, e.g. after raising `iogpu.wired_limit_mb`.
    case fraction(Double)
}

public enum GPULimitSource: Sendable, Equatable {
    case metal
    case ruleOfThumb
    case custom(Double)
    case cpuOnly

    public var label: String {
        switch self {
        case .metal: "reported by Metal"
        case .ruleOfThumb: "rule of thumb"
        case let .custom(f): "custom, \(Format.percent(f)) of RAM"
        case .cpuOnly: "CPU inference, 75% of RAM"
        }
    }
}

/// The machine a model is being judged against.
public struct MacProfile: Sendable, Equatable {
    public var chipName: String
    public var physicalMemory: Int64
    public var gpuWorkingSet: Int64
    public var gpuLimitSource: GPULimitSource
    public var isAppleSilicon: Bool
    public var isSample: Bool

    public init(chipName: String, physicalMemory: Int64, gpuWorkingSet: Int64, gpuLimitSource: GPULimitSource,
                isAppleSilicon: Bool = true, isSample: Bool = false) {
        self.chipName = chipName
        self.physicalMemory = physicalMemory
        self.gpuWorkingSet = gpuWorkingSet
        self.gpuLimitSource = gpuLimitSource
        self.isAppleSilicon = isAppleSilicon
        self.isSample = isSample
    }

    /// Approximates macOS's default GPU wired-memory cap on Apple Silicon:
    /// about two thirds of RAM up to 36 GB, about three quarters above.
    public static func ruleOfThumbLimit(physicalMemory: Int64) -> Int64 {
        physicalMemory <= 36 << 30 ? physicalMemory * 2 / 3 : physicalMemory * 3 / 4
    }

    /// A deterministic 16 GB machine, used for screenshots and labelled as a sample.
    public static let sample16GB = MacProfile(
        chipName: "Sample Mac", physicalMemory: 16 << 30,
        gpuWorkingSet: ruleOfThumbLimit(physicalMemory: 16 << 30),
        gpuLimitSource: .ruleOfThumb, isAppleSilicon: true, isSample: true
    )

    public static func current(policy: GPULimitPolicy = .automatic) -> MacProfile {
        let physical = Int64(clamping: ProcessInfo.processInfo.physicalMemory)
        let hw = Hardware.shared
        let name = hw.chipName ?? (hw.isAppleSilicon ? "Apple Silicon Mac" : "Intel Mac")
        let limit: Int64
        let source: GPULimitSource
        switch policy {
        case let .fraction(f):
            limit = Int64(Double(physical) * min(max(f, 0.1), 1.0))
            source = .custom(f)
        case .automatic:
            if !hw.isAppleSilicon {
                limit = physical * 3 / 4
                source = .cpuOnly
            } else if let metal = hw.metalWorkingSet, metal > 0, metal <= physical {
                limit = metal
                source = .metal
            } else {
                limit = ruleOfThumbLimit(physicalMemory: physical)
                source = .ruleOfThumb
            }
        }
        return MacProfile(chipName: name, physicalMemory: physical, gpuWorkingSet: limit,
                          gpuLimitSource: source, isAppleSilicon: hw.isAppleSilicon)
    }
}

/// Facts about this machine that don't change while the app runs, read once.
struct Hardware: Sendable {
    let chipName: String?
    let isAppleSilicon: Bool
    let metalWorkingSet: Int64?

    static let shared = Hardware()

    private init() {
        chipName = Self.sysctlString("machdep.cpu.brand_string")
        isAppleSilicon = (Self.sysctlInt("hw.optional.arm64") ?? 0) == 1
        #if canImport(Metal)
        if let device = MTLCreateSystemDefaultDevice(), device.hasUnifiedMemory {
            metalWorkingSet = Int64(clamping: device.recommendedMaxWorkingSetSize)
        } else {
            metalWorkingSet = nil
        }
        #else
        metalWorkingSet = nil
        #endif
    }

    static func sysctlString(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var buffer = [UInt8](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return nil }
        let text = String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self).trimmingCharacters(in: .whitespaces)
        return text.isEmpty ? nil : text
    }

    static func sysctlInt(_ name: String) -> Int64? {
        var value: Int64 = 0
        var size = MemoryLayout<Int64>.size
        guard sysctlbyname(name, &value, &size, nil, 0) == 0 else { return nil }
        return value
    }
}
