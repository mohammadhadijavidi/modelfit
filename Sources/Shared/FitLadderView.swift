import SwiftUI
import ModelFitCore

/// One stacked capsule: weights, KV cache, then runtime overhead, on a shared scale.
struct MemoryBar: View {
    let estimate: MemoryEstimate
    let scaleMax: Double
    var height: CGFloat = 20
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        GeometryReader { geo in
            let unit = geo.size.width / max(scaleMax, 1)
            ZStack(alignment: .leading) {
                Capsule().fill(theme.track)
                HStack(spacing: 1.5) {
                    segment(width: CGFloat(estimate.weights) * unit,
                            fill: LinearGradient(colors: [theme.weights, theme.weightsLight], startPoint: .leading, endPoint: .trailing))
                    if estimate.kvCache > 0 {
                        segment(width: CGFloat(estimate.kvCache) * unit,
                                fill: LinearGradient(colors: [theme.kv, theme.kvLight], startPoint: .leading, endPoint: .trailing))
                    }
                    segment(width: CGFloat(estimate.overhead) * unit,
                            fill: LinearGradient(colors: [theme.runtime, theme.runtime.opacity(0.8)], startPoint: .leading, endPoint: .trailing))
                }
                .clipShape(Capsule())
                .overlay(alignment: .top) {
                    // A soft top highlight gives the bar a little physical depth.
                    Capsule()
                        .fill(LinearGradient(colors: [.white.opacity(theme.isDark ? 0.10 : 0.28), .clear], startPoint: .top, endPoint: .center))
                        .allowsHitTesting(false)
                }
            }
        }
        .frame(height: height)
    }

    private func segment(width: CGFloat, fill: LinearGradient) -> some View {
        Rectangle().fill(fill).frame(width: max(0, width))
    }
}

/// Dashed GPU-limit line and solid physical-memory line drawn over a column of bars.
struct LimitLines: View {
    let profile: MacProfile
    let scaleMax: Double
    let leadingInset: CGFloat
    let trailingInset: CGFloat
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        GeometryReader { geo in
            let width = max(1, geo.size.width - leadingInset - trailingInset)
            let gpuX = leadingInset + width * CGFloat(Double(profile.gpuWorkingSet) / scaleMax)
            let ramX = leadingInset + width * CGFloat(Double(profile.physicalMemory) / scaleMax)
            Path { p in
                p.move(to: CGPoint(x: gpuX, y: 0))
                p.addLine(to: CGPoint(x: gpuX, y: geo.size.height))
            }
            .stroke(Color.primary.opacity(0.55), style: StrokeStyle(lineWidth: 1.25, lineCap: .round, dash: [3, 4]))
            Path { p in
                p.move(to: CGPoint(x: ramX, y: 0))
                p.addLine(to: CGPoint(x: ramX, y: geo.size.height))
            }
            .stroke(Color.primary.opacity(0.28), lineWidth: 1)
        }
        .allowsHitTesting(false)
        .accessibilityHidden(true)
    }
}

/// Labels for the limit lines: GPU label ends at its line, memory label starts at its line,
/// so they can never overlap.
struct LimitLabels: View {
    let profile: MacProfile
    let scaleMax: Double
    let leadingInset: CGFloat
    let trailingInset: CGFloat

    var body: some View {
        GeometryReader { geo in
            let width = max(1, geo.size.width - leadingInset - trailingInset)
            let gpuX = leadingInset + width * CGFloat(Double(profile.gpuWorkingSet) / scaleMax)
            let ramX = leadingInset + width * CGFloat(Double(profile.physicalMemory) / scaleMax)
            ZStack(alignment: .topLeading) {
                HStack(spacing: 4) {
                    Text("GPU limit")
                    Text(Format.bytes(profile.gpuWorkingSet)).fontWeight(.semibold).monospacedDigit()
                }
                .fixedSize()
                .frame(width: 220, alignment: .trailing)
                .offset(x: gpuX - 220 - 6)
                HStack(spacing: 4) {
                    Text(Format.bytes(profile.physicalMemory)).fontWeight(.semibold).monospacedDigit()
                    Text("memory")
                }
                .fixedSize()
                .offset(x: ramX + 6)
            }
            .font(Typo.caption)
            .foregroundStyle(.secondary)
        }
        .frame(height: 15)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("GPU limit \(Format.bytes(profile.gpuWorkingSet)), physical memory \(Format.bytes(profile.physicalMemory))")
    }
}

/// The signature visualization: memory needed at each context length against this Mac's
/// GPU limit and physical memory.
struct FitLadderView: View {
    let estimates: [MemoryEstimate]
    let profile: MacProfile
    var compact = false
    @Environment(\.colorScheme) private var scheme
    @Environment(\.accessibilityReduceMotion) private var reduceMotion

    private var labelWidth: CGFloat { compact ? 42 : 64 }
    private var trailingWidth: CGFloat { compact ? 140 : 196 }
    private var spacing: CGFloat { compact ? 12 : 16 }
    private var barHeight: CGFloat { compact ? 14 : 20 }

    var scaleMax: Double {
        max(Double(profile.physicalMemory), Double(estimates.map(\.total).max() ?? 0)) * 1.05
    }

    var body: some View {
        let lead = labelWidth + spacing
        let trail = trailingWidth + spacing
        VStack(alignment: .leading, spacing: compact ? 6 : 10) {
            LimitLabels(profile: profile, scaleMax: scaleMax, leadingInset: lead, trailingInset: trail)
            VStack(spacing: compact ? 12 : 18) {
                ForEach(estimates) { row($0) }
            }
            .padding(.vertical, compact ? 2 : 4)
            .overlay(LimitLines(profile: profile, scaleMax: scaleMax, leadingInset: lead, trailingInset: trail))
        }
        .animation(reduceMotion ? nil : .spring(response: 0.45, dampingFraction: 0.85), value: estimates.map(\.total))
    }

    private func row(_ e: MemoryEstimate) -> some View {
        let theme = Theme(scheme)
        let (number, unit) = Format.bytesParts(e.total)
        return HStack(alignment: compact ? .center : .top, spacing: spacing) {
            Text(Format.context(e.contextLength))
                .font(Typo.number(compact ? 13 : 17))
                .frame(width: labelWidth, alignment: .leading)
                .padding(.top, compact ? 0 : -1)

            VStack(alignment: .leading, spacing: 6) {
                MemoryBar(estimate: e, scaleMax: scaleMax, height: barHeight)
                if !compact {
                    breakdown(e, theme: theme)
                }
            }

            HStack(alignment: .firstTextBaseline, spacing: 8) {
                HStack(alignment: .firstTextBaseline, spacing: 2) {
                    Text(number).font(Typo.number(compact ? 13 : 16))
                    Text(unit).font(Typo.caption).foregroundStyle(.secondary)
                }
                Spacer(minLength: 4)
                VerdictPill(verdict: e.verdict, compact: compact)
            }
            .frame(width: trailingWidth)
            .padding(.top, compact ? 0 : -1)
        }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(Format.context(e.contextLength)) context needs about \(Format.bytes(e.total)). \(e.verdict.label).")
    }

    private func breakdown(_ e: MemoryEstimate, theme: Theme) -> some View {
        let plus = Text(" + ").foregroundStyle(.tertiary)
        return (
            Text(Format.gigabytes(e.weights)).foregroundStyle(theme.weights)
                + plus
                + Text(Format.gigabytes(e.kvCache)).foregroundStyle(theme.kv)
                + plus
                + Text(Format.gigabytes(e.overhead)).foregroundStyle(.secondary)
                + Text(" GB").foregroundStyle(.tertiary)
        )
        .font(.system(size: 11, weight: .medium).monospacedDigit())
        .fixedSize()
    }
}

/// Legend for the three memory components.
struct MemoryLegend: View {
    let kvLabel: String
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        HStack(spacing: 16) {
            LegendItem(color: theme.weights, label: "Weights")
            LegendItem(color: theme.kv, label: "KV cache (\(kvLabel))")
            LegendItem(color: theme.runtime, label: "Compute + runtime")
        }
    }
}
