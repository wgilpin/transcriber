#!/bin/zsh
set -e
cd "$(dirname "$0")"
swiftc -O meetingd.swift -o meetingd \
  -Xlinker -sectcreate -Xlinker __TEXT -Xlinker __info_plist -Xlinker Info.plist
echo "built ./meetingd"
