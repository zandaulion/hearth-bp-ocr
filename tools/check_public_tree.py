# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Check Git's tracked publication surface without printing secret values."""
from pathlib import Path
import hashlib
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
paths = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
paths = [p for p in paths if p]
if not paths:
    raise SystemExit("No tracked files; stage the intended publication files first")

blocked = re.compile(r"(?:^|/)(?:initial-code|dataset|photos|samples|reports|artifacts|runs|node_modules|\.venv|\.ssh|\.aws|\.codex|\.roboflow)(?:/|$)|(?:^|/)\.env(?:\.|$)|ground_truth|\.(?:onnx|pt|pth|joblib|pkl|log|pem|key|p12|pfx|zip|gz|jpg|jpeg|png|webp|heic|mp4|mov)$", re.I)
icons = {"prototype/web/icon-192.png", "prototype/web/icon-512.png"}
# Exact reviewed exports only. A replacement needs a fresh provenance/metadata audit.
models = {
    "prototype/web/models/bp-detector.onnx": "8a7dafaed0aa9056308171d57d024343049cff4ba1a9adfc54d274c38a31fb07",
    "prototype/web/models/bp-digits.onnx": "f41a41f9138c7c80d7e7fa1c3ed5afe93c3f5beacb3efd861ba3222b597cd732",
}
# Assemble markers so this checker does not match its own literal source.
patterns = {
    "private key": re.compile("-----BEGIN " + r"(?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub credential": re.compile(r"\bgh" + r"[pousr]_[A-Za-z0-9]{20,}\b|github" + r"_pat_[A-Za-z0-9_]{20,}"),
    "cloud access key": re.compile(r"\b(?:AK" + r"IA|ASIA)[A-Z0-9]{16}\b"),
    "home directory": re.compile(r"[A-Z]:[/\\]Users[/\\][^/\\\s]+|/ho" + r"me/[^/\s]+", re.I),
    "Tailscale address": re.compile(r"\b100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3}\b"),
    "personal email": re.compile(r"[\w.+-]+@(?:gmail|yahoo|hotmail|outlook)\.com\b", re.I),
    "literal API secret": re.compile(r'''(?:api[_-]?key|access_token|password)\s*[:=]\s*["'][A-Za-z0-9_+/=-]{20,}["']''', re.I),
}
issues = []
for name in paths:
    if name not in icons and name not in models and blocked.search(name):
        issues.append((name, "excluded artifact path"))
    payload = subprocess.check_output(["git", "show", f":{name}"], cwd=ROOT)
    if name in models:
        if hashlib.sha256(payload).hexdigest() != models[name]:
            issues.append((name, "model differs from reviewed export"))
        continue
    if name in icons:
        continue
    if b"\0" in payload:
        issues.append((name, "unexpected binary file"))
        continue
    content = payload.decode("utf-8")
    for label, pattern in patterns.items():
        if pattern.search(content):
            issues.append((name, label))
for name, label in issues:
    print(f"BLOCK: {name}: {label}")
print(f"Checked {len(paths)} tracked files; {len(issues)} issues")
sys.exit(bool(issues))
