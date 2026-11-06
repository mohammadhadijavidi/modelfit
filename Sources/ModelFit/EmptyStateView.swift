import SwiftUI
import ModelFitCore

/// First launch: an invitation to drop a model.
struct EmptyStateView: View {
    let profile: MacProfile
    var open: () -> Void
    @Environment(\.colorScheme) private var scheme

    var body: some View {
        let theme = Theme(scheme)
        VStack(spacing: 26) {
            ZStack {
                RoundedRectangle(cornerRadius: 30, style: .continuous)
                    .strokeBorder(theme.weights.opacity(0.45), style: StrokeStyle(lineWidth: 2, dash: [8, 7]))
                    .frame(width: 156, height: 156)
                RoundedRectangle(cornerRadius: 22, style: .continuous)
                    .fill(LinearGradient(colors: [theme.weights.opacity(0.18), theme.kv.opacity(0.12)], startPoint: .topLeading, endPoint: .bottomTrailing))
                    .frame(width: 112, height: 112)
                VStack(spacing: 7) {
                    ForEach(Array([0.42, 0.62, 0.86].enumerated()), id: \.offset) { index, fraction in
                        HStack(spacing: 2) {
                            Capsule().fill(theme.weights).frame(width: 30, height: 9)
                            Capsule().fill(index == 2 ? theme.wontFit : theme.kv).frame(width: 46 * fraction, height: 9)
                        }
                        .frame(width: 72, alignment: .leading)
                    }
                }
                Image(systemName: "arrow.down")
                    .font(.system(size: 15, weight: .bold))
                    .foregroundStyle(.white)
                    .frame(width: 32, height: 32)
                    .background(theme.weights, in: Circle())
                    .offset(x: 56, y: 56)
            }
            .accessibilityHidden(true)

            VStack(spacing: 10) {
                Text("Drop a model to see if it fits")
                    .font(Typo.display)
                Text("GGUF files, .safetensors files, or MLX and Hugging Face model folders. Only the header is read, so even a 40 GB model opens instantly.")
                    .font(.system(size: 14))
                    .foregroundStyle(.secondary)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 470)
                    .fixedSize(horizontal: false, vertical: true)
            }

            Button(action: open) {
                Label("Open Model…", systemImage: "plus")
            }
            .buttonStyle(ProminentButtonStyle())
            .keyboardShortcut("o")

            VStack(spacing: 6) {
                Label("Press Space on a .gguf or .safetensors file in Finder for a Quick Look preview.", systemImage: "eye")
                MacSummaryLine(profile: profile)
            }
            .font(Typo.caption)
            .foregroundStyle(.tertiary)
            .multilineTextAlignment(.center)
        }
        .padding(40)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }
}
