# Contributing to ModelFit

Thanks for helping. ModelFit is a small, dependency-free macOS app, and it should stay that way.

## Setup

```bash
brew install xcodegen
xcodegen generate
open ModelFit.xcodeproj
```

Run the tests with `xcodebuild -scheme ModelFit -destination 'platform=macOS' test`.

## Guidelines

- All parsing and memory math lives in `Sources/ModelFitCore` and must be covered by tests in
  `Tests/ModelFitCoreTests`. Tests build tiny synthetic files with `GGUFWriter` and
  `SafetensorsWriter`; never commit real model weights.
- Parsers must never read tensor data and must never crash on damaged input: return a
  `ModelFitError` instead. Add a corruption test for any new field you parse.
- Every number shown as an estimate needs its formula in "How we calculate"
  (`CalculationExplainer`).
- Swift 6 language mode with strict concurrency; no new warnings; no third-party dependencies.
- UI: SF Symbols only, no emojis; check both light and dark mode; respect Reduce Motion.

## Good first issues

- More architectures with non-standard KV layouts (sliding-window-aware estimates, hybrids).
- A thumbnail extension for model files in Finder.
- Localizations.

Open an issue first for larger changes so we can agree on the approach.
