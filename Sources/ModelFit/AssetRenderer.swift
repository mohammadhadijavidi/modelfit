import AppKit
import ImageIO
import SwiftUI
import UniformTypeIdentifiers
import ModelFitCore

/// `ModelFit --render-assets <dir>`: renders the real app views with sample-header models
/// into PNGs (dark and light, 2x) for the README, then exits. Used on CI.
///
/// NavigationSplitView and List are AppKit-backed and don't draw offscreen, so the window
/// chrome is a faithful SwiftUI composition; every content view is the same type the app shows.
@MainActor
enum AssetRenderer {
    static let windowSize = CGSize(width: 1240, height: 860)
    static let sidebarWidth: CGFloat = 250

    static func run(output: URL) -> Int32 {
        _ = NSApplication.shared
        NSApp.setActivationPolicy(.prohibited)
        do {
            try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
            let temp = FileManager.default.temporaryDirectory.appendingPathComponent("modelfit-samples", isDirectory: true)
            let urls = try SampleModels.writeAll(to: temp)
            let store = ModelStore()
            for url in urls {
                let id = UUID()
                store.items.append(ModelItem(id: id, url: url, name: url.lastPathComponent, state: .loading))
                store.apply(Result { try ModelLoader.analyze(url: url) }.mapError { ($0 as? ModelFitError) ?? .notAModel(file: url.lastPathComponent) }, to: id)
            }
            let brokenURL = temp.appendingPathComponent("Phi-3-mini-Q4_K_M.gguf")
            try SampleModels.writeTruncated(to: brokenURL)
            let brokenID = UUID()
            let brokenStore = ModelStore()
            brokenStore.items = store.items
            brokenStore.items.append(ModelItem(id: brokenID, url: brokenURL, name: brokenURL.lastPathComponent, state: .loading))
            brokenStore.apply(Result { try ModelLoader.analyze(url: brokenURL) }.mapError { ($0 as? ModelFitError) ?? .notAModel(file: "?") }, to: brokenID)

            let analyses = store.analyses
            guard analyses.count == 4, let llama = analyses.first, let qwen = analyses.dropFirst().first else {
                print("Expected 4 sample analyses, got \(analyses.count)")
                return 2
            }
            let inputs = FitInputs(profile: .sample16GB, settings: EstimationSettings())
            try report(store: store, broken: brokenStore, inputs: inputs).write(to: output.appendingPathComponent("report.txt"), atomically: true, encoding: .utf8)

            for scheme in [ColorScheme.dark, .light] {
                let tag = scheme == .dark ? "dark" : "light"
                let rows = store.items.map { SidebarRowModel(item: $0, inputs: inputs) }

                try render(WindowFrame(rows: rows, selected: .model(store.items[0].id), profile: inputs.profile,
                                       title: llama.displayName, subtitle: llama.formatDetail) {
                    DetailContent(analysis: llama, inputs: inputs, kvType: .constant(.f16), showCalculation: .constant(false))
                        .padding(28)
                }, size: windowSize, scheme: scheme, to: output.appendingPathComponent("window-detail-\(tag).png"))

                try render(WindowFrame(rows: rows, selected: .model(store.items[0].id), profile: inputs.profile,
                                       title: llama.displayName, subtitle: llama.formatDetail, height: nil) {
                    DetailContent(analysis: llama, inputs: inputs, kvType: .constant(.f16), showCalculation: .constant(true))
                        .padding(28)
                }, size: CGSize(width: windowSize.width, height: 2500), scheme: scheme, fitHeight: true,
                to: output.appendingPathComponent("detail-full-\(tag).png"))

                let q8 = FitInputs(profile: .sample16GB, settings: EstimationSettings(kvCacheType: .q8_0))
                try render(WindowFrame(rows: store.items.map { SidebarRowModel(item: $0, inputs: q8) },
                                       selected: .model(store.items[1].id), profile: q8.profile,
                                       title: qwen.displayName, subtitle: qwen.formatDetail) {
                    DetailContent(analysis: qwen, inputs: q8, kvType: .constant(.q8_0), showCalculation: .constant(false))
                        .padding(28)
                }, size: windowSize, scheme: scheme, to: output.appendingPathComponent("window-mlx-\(tag).png"))

                try render(WindowFrame(rows: rows, selected: .compare, profile: inputs.profile, title: "Compare", subtitle: "4 models") {
                    CompareContent(models: analyses, inputs: inputs, context: .constant(8192)).padding(28)
                }, size: windowSize, scheme: scheme, to: output.appendingPathComponent("window-compare-\(tag).png"))

                try render(WindowFrame(rows: [], selected: nil, profile: inputs.profile, title: "ModelFit", subtitle: nil) {
                    EmptyStateView(profile: inputs.profile) {}.frame(height: windowSize.height - 53)
                }, size: windowSize, scheme: scheme, to: output.appendingPathComponent("window-empty-\(tag).png"))

                let brokenRows = brokenStore.items.map { SidebarRowModel(item: $0, inputs: inputs) }
                if case let .failed(error)? = brokenStore.items.last?.state {
                    try render(WindowFrame(rows: brokenRows, selected: .model(brokenID), profile: inputs.profile,
                                           title: brokenURL.lastPathComponent, subtitle: nil) {
                        ErrorScreen(error: error, fileName: brokenURL.lastPathComponent, reveal: {}, remove: {})
                            .frame(height: windowSize.height - 53)
                    }, size: windowSize, scheme: scheme, to: output.appendingPathComponent("window-error-\(tag).png"))
                }

                try render(CompactPreviewView(analysis: llama, profile: inputs.profile, settings: inputs.settings),
                           size: CGSize(width: 680, height: 520), scheme: scheme,
                           to: output.appendingPathComponent("quicklook-\(tag).png"))
            }
            print("Rendered assets to \(output.path)")
            return 0
        } catch {
            print("Render failed: \(error)")
            return 1
        }
    }

    static func render<V: View>(_ view: V, size: CGSize, scheme: ColorScheme, fitHeight: Bool = false, to url: URL) throws {
        NSApp.appearance = NSAppearance(named: scheme == .dark ? .darkAqua : .aqua)
        let content = view
            .frame(width: size.width, height: fitHeight ? nil : size.height, alignment: .top)
            .environment(\.colorScheme, scheme)
            .environment(\.isRenderingAsset, true)
        let renderer = ImageRenderer(content: content)
        renderer.scale = 2
        guard let image = renderer.cgImage else { throw RenderError.noImage(url.lastPathComponent) }
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw RenderError.write(url.lastPathComponent)
        }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw RenderError.write(url.lastPathComponent) }
        print("wrote \(url.lastPathComponent) \(image.width)x\(image.height)")
    }

    enum RenderError: Error { case noImage(String), write(String) }

    /// Plain-text numbers for every sample, so CI output can be checked without images.
    static func report(store: ModelStore, broken: ModelStore, inputs: FitInputs) -> String {
        var lines = ["ModelFit render report (sample headers, sample 16 GB Mac)", ""]
        for a in store.analyses {
            lines.append("\(a.displayName) [\(a.formatDetail)]")
            lines.append("  params \(a.parameterCount) (\(Format.parameters(a.parameterCount))), weights \(a.weightBytes) B (\(Format.bytes(a.weightBytes))), quant \(a.quantizationLabel), bpw \(a.bitsPerWeight.map { Format.fixed($0, decimals: 3) } ?? "-")")
            lines.append("  tensors \(a.tensorCount), files \(a.fileCount), header bytes read \(a.headerBytesRead), sample \(a.isSample)")
            for e in MemoryEstimator.ladder(a, settings: inputs.settings, profile: inputs.profile) {
                lines.append("  \(Format.context(e.contextLength)): total \(Format.bytes(e.total)) = w \(Format.bytes(e.weights)) + kv \(Format.bytes(e.kvCache)) + compute \(Format.bytes(e.compute)) + runtime \(Format.bytes(e.runtime)) -> \(e.verdict.label)")
            }
            lines.append("  summary: \(FitSummary(estimates: MemoryEstimator.ladder(a, settings: inputs.settings, profile: inputs.profile), profile: inputs.profile).headline)")
            if !a.warnings.isEmpty { lines.append("  warnings: \(a.warnings)") }
            lines.append("")
        }
        if case let .failed(e)? = broken.items.last?.state { lines.append("Truncated sample error: \(e.title) / \(e.detail)") }
        let live = MacProfile.current()
        lines.append("Runner Mac: \(live.chipName), \(Format.bytes(live.physicalMemory)), GPU limit \(Format.bytes(live.gpuWorkingSet)) (\(live.gpuLimitSource.label))")
        return lines.joined(separator: "\n")
    }
}

/// A faithful SwiftUI drawing of the app window: unified sidebar with traffic lights,
/// toolbar title, and the detail column.
struct WindowFrame<Detail: View>: View {
    let rows: [SidebarRowModel]
    let selected: SidebarSelection?
    let profile: MacProfile
    let title: String
    let subtitle: String?
    var height: CGFloat? = 860
    @ViewBuilder var detail: Detail
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        HStack(alignment: .top, spacing: 0) {
            sidebar(theme)
                .frame(width: 250)
                .frame(maxHeight: .infinity, alignment: .top)
                .background(theme.sidebar)
            Rectangle().fill(theme.hairline).frame(width: 1)
            VStack(spacing: 0) {
                toolbar(theme)
                Rectangle().fill(theme.hairline).frame(height: 1)
                detail
                    .fixedSize(horizontal: false, vertical: true)
                    .frame(maxWidth: .infinity, alignment: .top)
                Spacer(minLength: 0)
            }
            .background(theme.canvas)
        }
        .frame(height: height, alignment: .top)
        .clipped()
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay(RoundedRectangle(cornerRadius: 12, style: .continuous).strokeBorder(theme.isDark ? Color.white.opacity(0.14) : Color.black.opacity(0.14)))
    }

    private func sidebar(_ theme: Theme) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 8) {
                ForEach([0xFF5F57, 0xFEBC2E, 0x28C840] as [UInt32], id: \.self) { c in
                    Circle().fill(Color(hex: c)).frame(width: 12, height: 12)
                }
            }
            .padding(.leading, 20)
            .frame(height: 52)

            if rows.filter({ if case .loaded = $0.status { return true } else { return false } }).count >= 2 {
                row(selected: selected == .compare, theme: theme) {
                    Label("Compare", systemImage: "chart.bar.xaxis").font(.system(size: 13))
                }
                .padding(.bottom, 10)
            }
            Text("Models")
                .font(.system(size: 11, weight: .semibold))
                .foregroundStyle(.secondary)
                .padding(.horizontal, 20)
                .padding(.bottom, 4)
            if rows.isEmpty {
                Text("Drop .gguf, .safetensors or model folders here.")
                    .font(.system(size: 11.5))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 20)
            }
            ForEach(rows) { r in
                row(selected: selected == .model(r.id), theme: theme) { SidebarRow(model: r) }
            }
            Spacer(minLength: 16)
            MacFooter(profile: profile).padding(10)
        }
    }

    private func row<Content: View>(selected: Bool, theme: Theme, @ViewBuilder content: () -> Content) -> some View {
        content()
            .padding(.horizontal, 10)
            .padding(.vertical, 3)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(selected ? theme.selection : .clear, in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .padding(.horizontal, 10)
    }

    private func toolbar(_ theme: Theme) -> some View {
        HStack(spacing: 18) {
            VStack(alignment: .leading, spacing: 1) {
                Text(title).font(.system(size: 13, weight: .semibold)).lineLimit(1)
                if let subtitle {
                    Text(subtitle).font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }
            Spacer()
            ForEach(["folder", "minus.circle", "plus"], id: \.self) { symbol in
                Image(systemName: symbol)
                    .font(.system(size: 14, weight: .regular))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 20)
        .frame(height: 52)
    }
}
