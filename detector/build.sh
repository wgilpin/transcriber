#!/bin/zsh
set -e
cd "$(dirname "$0")"
for t in meetingd captureprobe capture; do
  swiftc -O $t.swift -o $t \
    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist
  echo "built ./$t"
done

# Capture needs a real bundle launched by launchd, or TCC will never prompt
# and the process tap silently returns zeros.
bundle() {
  local app=$1 exe=$2 ident=$3
  rm -rf $app
  mkdir -p $app/Contents/MacOS
  sed -e "s/__EXE__/$exe/" -e "s/__ID__/$ident/" AppInfo.plist.tmpl > $app/Contents/Info.plist
  cp $exe $app/Contents/MacOS/$exe
  codesign --force --sign - --identifier $ident $app >/dev/null 2>&1
  echo "bundled $app"
}
bundle Capture.app capture com.bgilpin.capture
bundle CaptureProbe.app captureprobe com.bgilpin.captureprobe
