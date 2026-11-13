#!/bin/zsh
set -e
cd "$(dirname "$0")"
for t in meetingd captureprobe capture; do
  swiftc -O $t.swift -o $t \
    -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist
  echo "built ./$t"
done

swiftc -O menubar.swift -o MeetingNotes
echo "built ./MeetingNotes"

# Capture needs a real bundle launched by launchd, or TCC will never prompt
# and the process tap silently returns zeros. The menu bar app needs one too:
# it is the responsible process for meetingd, so the calendar and Apple Events
# prompts are attributed to it.
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
bundle MeetingNotes.app MeetingNotes com.bgilpin.meetingnotes
