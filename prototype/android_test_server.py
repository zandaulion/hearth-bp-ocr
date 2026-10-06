# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Loopback-only Android development harness; receives metrics, never photos."""
import argparse
import json
from pathlib import Path
from http.server import ThreadingHTTPServer
from functools import partial
from urllib.parse import urlparse
from serve import Handler

ROOT = Path(__file__).resolve().parent


class TestHandler(Handler):
    report_path = ROOT / "reports/android_a52.json"

    def do_POST(self):
        if urlparse(self.path).path != "/android-test-result":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0"))
        if not 0 < length <= 100_000:
            self.send_error(413)
            return
        try:
            report = json.loads(self.rfile.read(length))
            assert isinstance(report, dict) and report.get("kind") == "hearth-android-test"
            phase = report.get("phase", "online")
            assert phase in ("online", "offline")
            self.report_path.parent.mkdir(parents=True, exist_ok=True)
            target = self.report_path if phase == "online" else self.report_path.with_name("android_a52_offline.json")
            target.write_text(json.dumps(report, indent=2), encoding="utf-8")
        except (ValueError, AssertionError):
            self.send_error(400)
            return
        self.send_response(204)
        self.end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8875)
    parser.add_argument("--output", type=Path, default=TestHandler.report_path)
    args = parser.parse_args()
    TestHandler.report_path = args.output.resolve()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(TestHandler, directory=str(ROOT / "web")))
    print(f"Android development harness: http://127.0.0.1:{args.port}/android-test.html", flush=True)
    server.serve_forever()
