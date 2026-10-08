#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Audit candidate test images for exact and perceptual overlap.

The reference JSON is intentionally image-free. Each entry contains a relative
path, SHA-256 digest, and a 256-bit difference hash (dHash).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def dhash256(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(image.convert("L").resize((17, 16)).getdata())
    value = 0
    for y in range(16):
        for x in range(16):
            value = (value << 1) | (pixels[y * 17 + x] > pixels[y * 17 + x + 1])
    return value


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference_json", type=Path)
    parser.add_argument("candidate_dir", type=Path)
    parser.add_argument("--threshold", type=int, default=18)
    args = parser.parse_args()

    reference = json.loads(args.reference_json.read_text())
    refs = [
        (row["path"], row["sha256"], int(row["dhash256"], 16))
        for row in reference
    ]
    candidates = sorted(
        path
        for path in args.candidate_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )

    exact = []
    near = []
    minimums = []
    for path in candidates:
        sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
        d_hash = dhash256(path)
        best_path, best_sha, best_hash = min(
            refs, key=lambda row: hamming(d_hash, row[2])
        )
        distance = hamming(d_hash, best_hash)
        minimums.append(distance)
        if sha256 == best_sha:
            exact.append((str(path), best_path))
        if distance <= args.threshold:
            near.append((str(path), best_path, distance))

    print(f"candidates={len(candidates)} references={len(refs)}")
    print(f"exact_matches={len(exact)}")
    for candidate, ref in exact:
        print(f"EXACT\t{candidate}\t{ref}")
    print(f"near_matches_at_{args.threshold}={len(near)}")
    for candidate, ref, distance in near:
        print(f"NEAR\t{distance}\t{candidate}\t{ref}")
    if minimums:
        ordered = sorted(minimums)
        print(
            "minimum_distance="
            f"min:{ordered[0]} p05:{ordered[len(ordered) // 20]} "
            f"median:{ordered[len(ordered) // 2]} max:{ordered[-1]}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
