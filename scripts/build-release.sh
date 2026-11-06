#!/usr/bin/env bash
# Builds ModelFit.app in Release, ad-hoc signs it, and zips it to dist/ModelFit.zip.
# Usage: scripts/build-release.sh   (run from the repo root on macOS with Xcode and XcodeGen)
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
BUILD="$ROOT/build/release"
DIST="$ROOT/dist"

command -v xcodegen >/dev/null || { echo "xcodegen is required: brew install xcodegen" >&2; exit 1; }
xcodegen generate --quiet

rm -rf "$BUILD" "$DIST"
mkdir -p "$DIST"

xcodebuild \
  -project ModelFit.xcodeproj \
  -scheme ModelFit \
  -configuration Release \
  -destination 'generic/platform=macOS' \
  -derivedDataPath "$BUILD/DerivedData" \
  CODE_SIGN_IDENTITY=- \
  CODE_SIGN_STYLE=Manual \
  DEVELOPMENT_TEAM= \
  build

APP="$BUILD/DerivedData/Build/Products/Release/ModelFit.app"
[ -d "$APP" ] || { echo "ModelFit.app was not produced" >&2; exit 1; }

# Ad-hoc sign the whole bundle. --preserve-metadata keeps the Quick Look extension's
# sandbox entitlements, which macOS requires before it will load the extension.
codesign --force --deep --preserve-metadata=entitlements,requirements,flags -s - "$APP"
codesign --verify --deep --strict --verbose=2 "$APP"

ditto -c -k --keepParent "$APP" "$DIST/ModelFit.zip"
echo "Built $DIST/ModelFit.zip ($(du -h "$DIST/ModelFit.zip" | cut -f1))"
