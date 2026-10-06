# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Small ADB-over-SSH helper for the explicitly authorized Android test device."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="lenovo")
    parser.add_argument("--serial", required=True)
    parser.add_argument("action", choices=("snapshot", "tap-text", "open", "screenshot", "swipe", "key"))
    parser.add_argument("value", nargs="?")
    args = parser.parse_args()

    def adb(*words):
        command = shlex.join(["adb", "-s", args.serial, *map(str, words)])
        return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=yes",
                               "-o", "ConnectTimeout=10", args.host, command],
                              capture_output=True, check=True, timeout=30).stdout

    def tree():
        adb("shell", "uiautomator", "dump", "/data/local/tmp/hearth-ui.xml")
        return ET.fromstring(adb("shell", "cat", "/data/local/tmp/hearth-ui.xml"))

    def visible(node):
        values = list(map(int, re.findall(r"\d+", node.get("bounds", ""))))
        return len(values) == 4 and values[2] > values[0] and values[3] > values[1]

    if args.action == "screenshot":
        target = Path(args.value).resolve()
        allowed = (Path(__file__).resolve().parent / "reports").resolve()
        if not target.is_relative_to(allowed):
            raise ValueError("Save Android screenshots under prototype/reports")
        target.write_bytes(adb("exec-out", "screencap", "-p"))
        print(target)
        return
    if args.action == "open":
        print(adb("shell", "am", "start", "-W", "-a", "android.intent.action.VIEW",
                  "-d", args.value, "com.android.chrome").decode())
    elif args.action == "tap-text":
        matches = [n for n in tree().iter("node") if n.get("text") == args.value
                   and n.get("enabled") == "true" and visible(n)]
        clickable = [n for n in matches if n.get("clickable") == "true"]
        if clickable:
            matches = clickable
        if len(matches) != 1:
            raise ValueError(f"Expected one visible matching text, found {len(matches)}")
        x1, y1, x2, y2 = map(int, re.findall(r"\d+", matches[0].get("bounds")))
        adb("shell", "input", "tap", (x1+x2)//2, (y1+y2)//2)
    elif args.action == "swipe":
        coordinates = args.value.split(",")
        if len(coordinates) != 5 or not all(v.isdigit() for v in coordinates):
            raise ValueError("Supply x1,y1,x2,y2,duration")
        adb("shell", "input", "swipe", *coordinates)
    elif args.action == "key":
        if not args.value.isdigit():
            raise ValueError("Supply a numeric Android key code")
        adb("shell", "input", "keyevent", args.value)
    nodes = []
    for node in tree().iter("node"):
        if (node.get("text") or node.get("content-desc")) and visible(node):
            nodes.append({"text":node.get("text", "")[:160], "description":node.get("content-desc", "")[:100],
                          "bounds":node.get("bounds"), "class":node.get("class"),
                          "resource":node.get("resource-id"), "clickable":node.get("clickable"),
                          "enabled":node.get("enabled"),"checked":node.get("checked")})
    for node in nodes:
        print(json.dumps(node))


if __name__ == "__main__":
    main()
