#!/bin/sh
# Build "Projects Search.app" into ./build. No Xcode project: swiftc + a hand-made bundle, ad-hoc signed.
set -e
cd "$(dirname "$0")"
APP="build/Projects Search.app"
rm -rf "$APP"; mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
xcrun coremlcompiler compile ../model/leaf_S32.mlpackage "$APP/Contents/Resources" >/dev/null
python3 - "$APP/Contents/Resources/vocab.txt" <<'PY'
import json, sys
v = json.load(open("../model/tokenizer.json"))["model"]["vocab"]
open(sys.argv[1], "w").write("\n".join(t for t, _ in sorted(v.items(), key=lambda x: x[1])) + "\n")
PY
cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>Projects Search</string>
  <key>CFBundleDisplayName</key><string>Projects Search</string>
  <key>CFBundleIdentifier</key><string>com.austegard.projects-search</string>
  <key>CFBundleExecutable</key><string>ProjectsSearch</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>0.1.0</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>LSMinimumSystemVersion</key><string>15.0</string>
  <key>LSUIElement</key><true/>
  <key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
swiftc -swift-version 5 -O -target arm64-apple-macos15.0 -o "$APP/Contents/MacOS/ProjectsSearch" main.swift
codesign --force --sign - "$APP" 2>/dev/null
echo "built $APP"
