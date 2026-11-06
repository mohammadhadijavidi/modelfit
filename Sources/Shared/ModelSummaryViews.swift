import SwiftUI
import ModelFitCore

/// Model name, location and key facts as chips.
struct ModelHeader: View {
    let analysis: ModelAnalysis
    var compact = false

    var body: some View {
        HStack(alignment: .top, spacing: compact ? 12 : 16) {
            FormatBadge(format: analysis.format, size: compact ? 34 : 46)
            VStack(alignment: .leading, spacing: compact ? 6 : 9) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(analysis.displayName)
                        .font(compact ? Typo.headline : Typo.display)
                        .lineLimit(1)
                        .truncationMode(.middle)
                        .textSelection(.enabled)
                    Text(subtitle)
                        .font(Typo.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                chips
            }
            Spacer(minLength: 0)
        }
    }

    private var subtitle: String {
        if analysis.isSample {
            return "Sample header: synthetic file with this model's published architecture"
        }
        return analysis.url.path(percentEncoded: false)
    }

    @ViewBuilder private var chips: some View {
        let a = analysis
        FlowLayout(spacing: 6, lineSpacing: 6) {
            Chip(symbol: a.format.symbolName, text: a.formatDetail)
            if let arch = a.architecture.architecture {
                Chip(symbol: "square.stack.3d.down.right", text: arch)
            }
            Chip(symbol: "number", text: "\(Format.parameters(a.parameterCount)) params")
            Chip(symbol: "dial.medium", text: bitsText(a))
            Chip(symbol: "internaldrive", text: Format.bytes(a.weightBytes))
            if a.isSample {
                Chip(symbol: "flask", text: "Sample header", tint: .orange)
            }
        }
    }

    private func bitsText(_ a: ModelAnalysis) -> String {
        guard let bpw = a.bitsPerWeight else { return a.quantizationLabel }
        return "\(a.quantizationLabel), \(Format.fixed(bpw, decimals: 2)) bpw"
    }
}

/// The one-line answer, tinted by the best verdict.
struct VerdictHeadline: View {
    let summary: FitSummary
    var compact = false
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let color = Theme(scheme).color(for: summary.verdict)
        HStack(alignment: .top, spacing: compact ? 10 : 14) {
            Image(systemName: summary.verdict.symbolName)
                .font(.system(size: compact ? 18 : 24, weight: .semibold))
                .foregroundStyle(color)
                .frame(width: compact ? 22 : 30)
                .padding(.top, compact ? 1 : 2)
            VStack(alignment: .leading, spacing: 4) {
                Text(summary.headline)
                    .font(compact ? Typo.title : Typo.headline)
                    .fixedSize(horizontal: false, vertical: true)
                if !summary.detail.isEmpty {
                    Text(summary.detail)
                        .font(compact ? Typo.caption : Typo.body)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            }
        }
        .accessibilityElement(children: .combine)
    }
}

/// Compact analysis for Quick Look (Space in Finder).
struct CompactPreviewView: View {
    let analysis: ModelAnalysis
    let profile: MacProfile
    let settings: EstimationSettings
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        let estimates = MemoryEstimator.ladder(analysis, settings: settings, profile: profile)
        let summary = FitSummary(estimates: estimates, profile: profile)
        VStack(alignment: .leading, spacing: 18) {
            ModelHeader(analysis: analysis, compact: true)
            Divider().overlay(theme.hairline)
            VerdictHeadline(summary: summary, compact: true)
            VStack(alignment: .leading, spacing: 12) {
                FitLadderView(estimates: estimates, profile: profile, compact: true)
                MemoryLegend(kvLabel: settings.kvCacheType.label)
            }
            .padding(16)
            .background(theme.card, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
            .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(theme.cardStroke))

            if let warning = analysis.warnings.first {
                Label(warning, systemImage: "exclamationmark.triangle.fill")
                    .font(Typo.caption)
                    .foregroundStyle(theme.tight)
                    .lineLimit(2)
            }
            Spacer(minLength: 0)
            HStack(spacing: 6) {
                Image(systemName: "info.circle")
                Text("Estimates for ") + Text(profile.isSample ? "a sample 16 GB Mac" : "this Mac").fontWeight(.semibold)
                    + Text(": \(Format.bytes(profile.physicalMemory)) memory, GPU limit \(Format.bytes(profile.gpuWorkingSet)). Open in ModelFit for the formula.")
            }
            .font(Typo.caption)
            .foregroundStyle(.secondary)
            .lineLimit(2)
        }
        .padding(22)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(theme.canvas)
    }
}

/// Error state shared by the app and Quick Look.
struct ModelErrorView: View {
    let error: ModelFitError
    let fileName: String
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        VStack(spacing: 14) {
            ZStack {
                Circle().fill(theme.wontFit.opacity(0.12)).frame(width: 64, height: 64)
                Image(systemName: "doc.badge.ellipsis")
                    .font(.system(size: 26, weight: .medium))
                    .foregroundStyle(theme.wontFit)
            }
            Text(error.title).font(Typo.headline)
            Text(error.detail)
                .font(Typo.body)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
                .fixedSize(horizontal: false, vertical: true)
            Text(error.recoverySuggestion)
                .font(Typo.caption)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(32)
        .accessibilityElement(children: .combine)
    }
}
