# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Serve only the browser prototype on loopback; no image-upload API exists."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map,".mjs":"text/javascript",".wasm":"application/wasm",".webmanifest":"application/manifest+json",".onnx":"application/octet-stream"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--port",type=int,default=8765)
    args=parser.parse_args()
    root=Path(__file__).resolve().parent / "web"
    server=ThreadingHTTPServer(("127.0.0.1",args.port),partial(Handler,directory=str(root)))
    print(f"Hearth BP reader: http://127.0.0.1:{args.port}",flush=True)
    server.serve_forever()
