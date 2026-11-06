import Foundation

/// Locale-stable formatting used across the app, Quick Look and tests.
///
/// Sizes use binary units (1 GB = 1024³ bytes), which is how macOS reports
/// physical memory in About This Mac and Activity Monitor.
public enum Format {
    public static let gib: Double = 1024 * 1024 * 1024
    public static let mib: Double = 1024 * 1024

    public static func bytes(_ value: Int64) -> String {
        let (number, unit) = bytesParts(value)
        return "\(number) \(unit)"
    }

    /// The numeric part and unit separately, for typographic layouts.
    public static func bytesParts(_ value: Int64) -> (String, String) {
        let v = Double(max(0, value))
        if v >= gib {
            let g = v / gib
            return (trimmed(fixed(g, decimals: g < 10 ? 2 : (g < 100 ? 1 : 0))), "GB")
        }
        if v >= mib {
            let m = v / mib
            return (trimmed(fixed(m, decimals: m < 10 ? 1 : 0)), "MB")
        }
        if v >= 1024 {
            return (fixed(v / 1024, decimals: 0), "KB")
        }
        return ("\(Int64(v))", "B")
    }

    /// Gigabytes with a fixed number of decimals, no unit: "4.58".
    public static func gigabytes(_ value: Int64, decimals: Int = 2) -> String {
        fixed(Double(value) / gib, decimals: decimals)
    }

    public static func parameters(_ count: Int64) -> String {
        let v = Double(count)
        switch v {
        case 1e12...: return fixed(v / 1e12, decimals: 2) + "T"
        case 1e11...: return fixed(v / 1e9, decimals: 0) + "B"
        case 1e10...: return fixed(v / 1e9, decimals: 1) + "B"
        case 1e9...: return fixed(v / 1e9, decimals: 2) + "B"
        case 1e8...: return fixed(v / 1e6, decimals: 0) + "M"
        case 1e6...: return fixed(v / 1e6, decimals: 1) + "M"
        case 1e3...: return fixed(v / 1e3, decimals: 1) + "K"
        default: return "\(count)"
        }
    }

    /// "2K", "128K", "1M", or a grouped integer when not a multiple of 1024.
    public static func context(_ tokens: Int) -> String {
        if tokens >= 1024, tokens % (1024 * 1024) == 0 { return "\(tokens / (1024 * 1024))M" }
        if tokens >= 1024, tokens % 1024 == 0 { return "\(tokens / 1024)K" }
        return integer(Int64(tokens))
    }

    /// Integer with comma grouping, independent of the user's locale: 131,072.
    public static func integer(_ value: Int64) -> String {
        let negative = value < 0
        var digits = Array(String(value.magnitude))
        var out: [Character] = []
        while digits.count > 3 {
            out.insert(contentsOf: [","] + digits.suffix(3), at: 0)
            digits.removeLast(3)
        }
        out.insert(contentsOf: digits, at: 0)
        return (negative ? "-" : "") + String(out)
    }

    public static func integer(_ value: Int) -> String { integer(Int64(value)) }

    public static func fixed(_ value: Double, decimals: Int) -> String {
        String(format: "%.\(max(0, decimals))f", value)
    }

    /// "16.0" becomes "16", "1.00" becomes "1"; "4.50" is left alone.
    static func trimmed(_ s: String) -> String {
        guard let dot = s.firstIndex(of: ".") else { return s }
        return s[s.index(after: dot)...].allSatisfy { $0 == "0" } ? String(s[..<dot]) : s
    }

    public static func percent(_ fraction: Double) -> String {
        if fraction > 0, fraction < 0.01 { return "<1%" }
        return fixed(fraction * 100, decimals: 0) + "%"
    }
}

extension Int64 {
    /// Addition that clamps instead of trapping, for sums over untrusted headers.
    func addingSaturating(_ other: Int64) -> Int64 {
        let (r, o) = addingReportingOverflow(other)
        return o ? (other > 0 ? .max : .min) : r
    }
}
