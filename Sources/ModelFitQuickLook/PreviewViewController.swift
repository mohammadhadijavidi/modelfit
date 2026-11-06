import Cocoa
import Quartz
import SwiftUI
import ModelFitCore

/// Quick Look preview for .gguf and .safetensors: press Space on a model in Finder.
///
/// Only the header is read, so even multi-gigabyte files preview instantly. The extension
/// is sandboxed and can read only the previewed file; split parts or a neighbouring
/// config.json are used when readable and skipped otherwise.
final class PreviewViewController: NSViewController, QLPreviewingController {
    override var nibName: NSNib.Name? { nil }

    override func loadView() {
        view = NSView(frame: NSRect(x: 0, y: 0, width: 680, height: 520))
        preferredContentSize = NSSize(width: 680, height: 520)
    }

    func preparePreviewOfFile(at url: URL) async throws {
        let result: Result<ModelAnalysis, ModelFitError> = await Task.detached(priority: .userInitiated) {
            do {
                guard let first = try ModelLoader.analyze(url: url).first else {
                    return .failure(.notAModel(file: url.lastPathComponent))
                }
                return .success(first)
            } catch let error as ModelFitError {
                return .failure(error)
            } catch {
                return .failure(.unreadable(file: url.lastPathComponent, reason: error.localizedDescription))
            }
        }.value

        let root: AnyView
        switch result {
        case let .success(analysis):
            root = AnyView(CompactPreviewView(analysis: analysis, profile: .current(), settings: EstimationSettings()))
        case let .failure(error):
            root = AnyView(
                ModelErrorView(error: error, fileName: url.lastPathComponent)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            )
        }
        let host = NSHostingView(rootView: root)
        host.translatesAutoresizingMaskIntoConstraints = false
        view.subviews.forEach { $0.removeFromSuperview() }
        view.addSubview(host)
        NSLayoutConstraint.activate([
            host.leadingAnchor.constraint(equalTo: view.leadingAnchor),
            host.trailingAnchor.constraint(equalTo: view.trailingAnchor),
            host.topAnchor.constraint(equalTo: view.topAnchor),
            host.bottomAnchor.constraint(equalTo: view.bottomAnchor),
        ])
    }
}
