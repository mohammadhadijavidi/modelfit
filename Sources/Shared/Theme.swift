import SwiftUI
import ModelFitCore

extension Color {
    init(hex: UInt32, opacity: Double = 1) {
        self.init(
            .sRGB,
            red: Double((hex >> 16) & 0xFF) / 255,
            green: Double((hex >> 8) & 0xFF) / 255,
            blue: Double(hex & 0xFF) / 255,
            opacity: opacity
        )
    }
}

/// ModelFit's palette, resolved from the SwiftUI color scheme rather than NSAppearance so
/// that offscreen renders (ImageRenderer) match the app exactly in both modes.
///
/// The three memory components each get one fixed hue used everywhere: weights are
/// "wafer blue", the KV cache is "cache teal", runtime overhead is a neutral slate so it
/// never competes with what the model itself needs.
struct Theme {
    let isDark: Bool

    init(_ scheme: ColorScheme) { isDark = scheme == .dark }

    var canvas: Color { isDark ? Color(hex: 0x17181C) : Color(hex: 0xF3F4F7) }
    var sidebar: Color { isDark ? Color(hex: 0x1F2026) : Color(hex: 0xE9EAEF) }
    var card: Color { isDark ? Color(hex: 0x22242B) : .white }
    var cardStroke: Color { isDark ? Color.white.opacity(0.07) : Color.black.opacity(0.06) }
    var inset: Color { isDark ? Color.white.opacity(0.045) : Color.black.opacity(0.035) }
    var track: Color { isDark ? Color.white.opacity(0.06) : Color.black.opacity(0.05) }
    var hairline: Color { isDark ? Color.white.opacity(0.09) : Color.black.opacity(0.08) }
    var selection: Color { isDark ? Color(hex: 0x5468FF, opacity: 0.28) : Color(hex: 0x5468FF, opacity: 0.16) }

    var weights: Color { isDark ? Color(hex: 0x7085FF) : Color(hex: 0x4F64F5) }
    var weightsLight: Color { isDark ? Color(hex: 0x95A4FF) : Color(hex: 0x7487FF) }
    var kv: Color { isDark ? Color(hex: 0x2CCDB4) : Color(hex: 0x10A893) }
    var kvLight: Color { isDark ? Color(hex: 0x6BE8D3) : Color(hex: 0x3CC8B2) }
    var runtime: Color { isDark ? Color(hex: 0x6E7180) : Color(hex: 0xB2B5C2) }

    var comfortable: Color { isDark ? Color(hex: 0x3CCB82) : Color(hex: 0x1C9A5C) }
    var tight: Color { isDark ? Color(hex: 0xF4B23E) : Color(hex: 0xC27A00) }
    var wontFit: Color { isDark ? Color(hex: 0xFF6E66) : Color(hex: 0xD93A33) }

    func color(for verdict: FitVerdict) -> Color {
        switch verdict {
        case .comfortable: comfortable
        case .tight: tight
        case .wontFit: wontFit
        }
    }

    func color(for format: ModelFormat) -> Color {
        switch format {
        case .gguf: weights
        case .safetensors: isDark ? Color(hex: 0xB38CFF) : Color(hex: 0x8A5CF0)
        case .mlxFolder: kv
        case .huggingFaceFolder: isDark ? Color(hex: 0xFFAA5C) : Color(hex: 0xE07A1F)
        }
    }

    /// Categorical colors for the tensor-type mix, most bytes first.
    func mixColor(_ index: Int) -> Color {
        let dark: [UInt32] = [0x7085FF, 0x2CCDB4, 0xB38CFF, 0xF4B23E, 0xFF7FAE, 0x8E93A6]
        let light: [UInt32] = [0x4F64F5, 0x10A893, 0x8A5CF0, 0xC98A12, 0xE0558E, 0x8E93A6]
        let list = isDark ? dark : light
        return Color(hex: list[min(index, list.count - 1)])
    }
}

/// Type scale. One family (SF Pro), with SF Rounded reserved for measured quantities so
/// numbers read as instruments, not prose.
enum Typo {
    static let display = Font.system(size: 30, weight: .semibold)
    static let headline = Font.system(size: 22, weight: .semibold)
    static let title = Font.system(size: 15, weight: .semibold)
    static let body = Font.system(size: 13)
    static let caption = Font.system(size: 11.5)
    static let label = Font.system(size: 11, weight: .medium)

    static func number(_ size: CGFloat, weight: Font.Weight = .semibold) -> Font {
        .system(size: size, weight: weight, design: .rounded).monospacedDigit()
    }

    static let formula = Font.system(size: 12, design: .monospaced)
}

/// Environment flag for offscreen renders (no hover, no animation, expanded states).
private struct RenderingKey: EnvironmentKey {
    static let defaultValue = false
}

extension EnvironmentValues {
    var isRenderingAsset: Bool {
        get { self[RenderingKey.self] }
        set { self[RenderingKey.self] = newValue }
    }
}
