import Foundation

/// The one-sentence answer shown above the memory ladder.
public struct FitSummary: Sendable, Equatable {
    public var headline: String
    public var detail: String
    /// The best verdict across contexts, used for tinting.
    public var verdict: FitVerdict
    /// Largest context that fits at all (tight or better).
    public var largestFittingContext: Int?

    public init(estimates: [MemoryEstimate], profile: MacProfile) {
        let comfortable = estimates.filter { $0.verdict == .comfortable }.map(\.contextLength)
        let tight = estimates.filter { $0.verdict == .tight }.map(\.contextLength)
        let wont = estimates.filter { $0.verdict == .wontFit }.map(\.contextLength)
        largestFittingContext = (comfortable + tight).max()
        verdict = estimates.map(\.verdict).min() ?? .wontFit

        func list(_ c: [Int]) -> String {
            let labels = c.map(Format.context)
            if labels.count <= 1 { return labels.first ?? "" }
            return labels.dropLast().joined(separator: ", ") + " and " + labels.last!
        }
        var tail: [String] = []
        if !tight.isEmpty { tail.append("Tight at \(list(tight)).") }
        if !wont.isEmpty { tail.append("Won't fit at \(list(wont)).") }

        if estimates.isEmpty {
            headline = "No estimate available"
            detail = "This model's header doesn't describe enough to estimate memory."
        } else if wont.isEmpty, tight.isEmpty {
            headline = "Fits comfortably at every context length"
            detail = "Up to \(Format.context(comfortable.max() ?? 0)) tokens, with headroom left for other apps."
        } else if let best = comfortable.max() {
            headline = "Runs comfortably up to \(Format.context(best)) context"
            detail = tail.joined(separator: " ")
        } else if let best = tight.max() {
            headline = "Fits, but only just, up to \(Format.context(best))"
            detail = (["Quit memory-hungry apps first."] + (wont.isEmpty ? [] : ["Won't fit at \(list(wont))."])).joined(separator: " ")
        } else {
            let need = estimates.first.map { Format.bytes($0.total) } ?? "?"
            headline = "Too big for this Mac"
            detail = "Needs about \(need) even at \(Format.context(estimates.first?.contextLength ?? 0)); the GPU limit here is \(Format.bytes(profile.gpuWorkingSet))."
        }
    }
}
