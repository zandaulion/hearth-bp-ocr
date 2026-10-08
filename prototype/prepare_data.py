# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Prepare source-preserving manifests for a detector; never mix in test data."""
import json
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

# Local external challenge sets are permanently test-only. Keep this guard near
# the final manifest write so a mistakenly edited source manifest cannot leak
# challenge images into detector training or validation.
TEST_ONLY_MARKERS = (
    "/dataset/external_test_",
    "/dataset/roboflow_naphop_external_",
    "/dataset/wikimedia_bp_test_",
    "/dataset/openverse_bp_candidates/",
)


def reject_test_only_paths(paths):
    leaked = [path for path in paths if any(marker in path for marker in TEST_ONLY_MARKERS)]
    if leaked:
        examples = "\n".join(f"  {path}" for path in leaked[:5])
        raise RuntimeError(f"Test-only images found in a training manifest:\n{examples}")


def main():
    out = ROOT / "prototype/data"
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "valid"):
        truth = json.loads((ROOT / f"evaluation/artifacts/roboflow_{split}_truth.json").read_text())
        paths = [(ROOT / r["local_file"]).resolve().as_posix() for r in truth]
        reject_test_only_paths(paths)
        (out / f"{split}.txt").write_text("\n".join(paths) + "\n")
        print(f"{split}: {len(paths)} images")
    config = {"path": ROOT.as_posix(), "train": (out / "train.txt").as_posix(), "val": (out / "valid.txt").as_posix(),
              "names": ["0", "1", "row", "2", "3", "4", "5", "6", "7", "8", "9"]}
    (out / "bp.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


if __name__ == "__main__":
    main()
