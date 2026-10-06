# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Prepare source-preserving manifests for a detector; never mix in test data."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "prototype/data"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        truth = json.loads((ROOT / f"evaluation/artifacts/roboflow_{split}_truth.json").read_text())
        paths = [(ROOT / r["local_file"]).resolve().as_posix() for r in truth]
        (out / f"{split}.txt").write_text("\n".join(paths) + "\n")
        print(f"{split}: {len(paths)} images")
    config = {"path": ROOT.as_posix(), "train": (out / "train.txt").as_posix(), "val": (out / "valid.txt").as_posix(),
              "names": ["0", "1", "row", "2", "3", "4", "5", "6", "7", "8", "9"]}
    (out / "bp.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
