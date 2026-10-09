#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Download and freeze a model-blind external BP-display benchmark.

The source annotations are used only to propose SYS/DIA/pulse transcriptions.
Every accepted item must be checked by a person for transcription, scope,
privacy, and overlap before the manifest can be frozen.  This script never
runs OCR inference.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import shutil
import statistics
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps


ROOT = Path(__file__).resolve().parents[1]
SOURCES_DIR = ROOT / "dataset/external_test_v3_sources"
REVIEW_DIR = ROOT / "dataset/external_test_v3_review"
OUTPUT_DIR = ROOT / "dataset/external_test_v3"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
PLAUSIBLE = {"sys": (50, 260), "dia": (30, 180), "pulse": (30, 220)}


@dataclass(frozen=True)
class Source:
    key: str
    workspace: str
    project: str
    version: int
    title: str
    page: str
    license: str = "CC BY 4.0"


SOURCES = (
    Source(
        key="ega_v5",
        workspace="adindra-vickar-ega-odhei",
        project="automated-bp-device-digit-recognition",
        version=5,
        title="Automated BP Device Digit Recognition",
        page="https://universe.roboflow.com/adindra-vickar-ega-odhei/automated-bp-device-digit-recognition/dataset/5",
    ),
    Source(
        key="datacluster_v2",
        workspace="datacluster-labs-agryi",
        project="bp-monitor-reading-medical-device-images",
        version=2,
        title="BP Monitor Reading | Medical Device Images",
        page="https://universe.roboflow.com/datacluster-labs-agryi/bp-monitor-reading-medical-device-images/dataset/2",
    ),
)


def roboflow_api_key() -> str | None:
    if os.environ.get("ROBOFLOW_API_KEY"):
        return os.environ["ROBOFLOW_API_KEY"]
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            key, separator, value = line.partition("=")
            if separator and key.strip() == "ROBOFLOW_API_KEY":
                return value.strip().strip('"').strip("'")
    return None


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash256(path: Path) -> int:
    with Image.open(path) as image:
        pixels = list(ImageOps.exif_transpose(image).convert("L").resize((17, 16)).getdata())
    value = 0
    for y in range(16):
        for x in range(16):
            value = (value << 1) | (pixels[y * 17 + x] > pixels[y * 17 + x + 1])
    return value


def hamming(left: int, right: int) -> int:
    return bin(left ^ right).count("1")


def parse_class_names(data_yaml: Path) -> list[str]:
    """Parse the simple list/dict forms emitted by Roboflow YOLO exports."""
    def scalar(raw: str) -> str:
        try:
            return str(ast.literal_eval(raw))
        except (SyntaxError, ValueError):
            return raw.strip().strip('"').strip("'")

    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines):
        if not line.lstrip().startswith("names:"):
            continue
        indent = len(line) - len(line.lstrip())
        remainder = line.split(":", 1)[1].strip()
        if remainder:
            value = ast.literal_eval(remainder)
            if isinstance(value, dict):
                return [str(value[key]) for key in sorted(value, key=int)]
            return [str(item) for item in value]

        values: list[tuple[int, str]] = []
        next_list_index = 0
        for child in lines[index + 1:]:
            child_indent = len(child) - len(child.lstrip())
            stripped = child.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if child_indent <= indent:
                break
            if stripped.startswith("-"):
                raw = stripped[1:].strip()
                values.append((next_list_index, scalar(raw)))
                next_list_index += 1
            elif ":" in stripped:
                key, raw = stripped.split(":", 1)
                values.append((int(key.strip()), scalar(raw.strip())))
        if values:
            return [value for _, value in sorted(values)]
    raise ValueError(f"Could not parse class names from {data_yaml}")


def safe_extract(archive: Path, destination: Path) -> None:
    resolved = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if target != resolved and resolved not in target.parents:
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bundle.extractall(destination)


def download_source(source: Source, api_key: str) -> None:
    target = SOURCES_DIR / source.key
    if (target / "data.yaml").exists():
        print(f"{source.key}: already downloaded")
        return
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty {target}")
    target.mkdir(parents=True, exist_ok=True)

    endpoint = (
        f"https://api.roboflow.com/{source.workspace}/{source.project}/"
        f"{source.version}/yolov8?"
        + urllib.parse.urlencode({"api_key": api_key})
    )
    try:
        with urllib.request.urlopen(endpoint, timeout=60) as response:
            payload = json.load(response)
        download_url = payload["export"]["link"]
        request = urllib.request.Request(download_url, headers={"User-Agent": "hearth-bp-ocr/1"})
        with tempfile.NamedTemporaryFile(suffix=".zip", delete=False, dir=target.parent) as temporary:
            archive = Path(temporary.name)
            with urllib.request.urlopen(request, timeout=180) as response:
                shutil.copyfileobj(response, temporary)
        safe_extract(archive, target)
        archive.unlink()
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError) as error:
        status = getattr(error, "code", "network/error response")
        raise RuntimeError(f"Roboflow download failed for {source.key}: {status}") from error
    print(f"{source.key}: downloaded")


def label_path_for(image_path: Path) -> Path:
    parts = list(image_path.parts)
    try:
        index = len(parts) - 1 - parts[::-1].index("images")
    except ValueError as error:
        raise ValueError(f"Image is not under an images directory: {image_path}") from error
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def annotation_boxes(image_path: Path, class_names: list[str]) -> list[dict]:
    label_path = label_path_for(image_path)
    if not label_path.exists():
        raise FileNotFoundError(label_path)
    boxes = []
    for line_number, line in enumerate(label_path.read_text().splitlines(), 1):
        fields = line.split()
        if len(fields) < 5:
            raise ValueError(f"{label_path}:{line_number}: incomplete YOLO annotation")
        class_id = int(fields[0])
        if class_id >= len(class_names):
            raise ValueError(f"{label_path}:{line_number}: class id {class_id} is out of range")
        coordinates = list(map(float, fields[1:]))
        if len(coordinates) == 4:
            x, y, width, height = coordinates
        elif len(coordinates) >= 6 and len(coordinates) % 2 == 0:
            xs, ys = coordinates[::2], coordinates[1::2]
            left, right = min(xs), max(xs)
            top, bottom = min(ys), max(ys)
            x, y = (left + right) / 2, (top + bottom) / 2
            width, height = right - left, bottom - top
        else:
            raise ValueError(f"{label_path}:{line_number}: malformed YOLO annotation")
        boxes.append({
            "class": class_names[class_id],
            "x": x,
            "y": y,
            "width": width,
            "height": height,
        })
    return boxes


def reading_from_digit_boxes(boxes: list[dict]) -> tuple[dict | None, str]:
    digits = [box for box in boxes if box["class"] in set("0123456789")]
    if not 6 <= len(digits) <= 9:
        return None, f"digit_count_{len(digits)}"
    ordered = sorted(digits, key=lambda box: (box["y"], box["x"]))
    candidates = []
    for first_end in range(2, 4):
        for second_end in range(first_end + 2, first_end + 4):
            groups = (ordered[:first_end], ordered[first_end:second_end], ordered[second_end:])
            if not all(2 <= len(group) <= 3 for group in groups):
                continue
            centers = [statistics.mean(box["y"] for box in group) for group in groups]
            heights = [statistics.median(box["height"] for box in group) for group in groups]
            if any(
                max(abs(box["y"] - center) for box in group) > 0.7 * height
                for group, center, height in zip(groups, centers, heights)
            ):
                continue
            if any(
                centers[index + 1] - centers[index] < 0.55 * min(heights[index:index + 2])
                for index in range(2)
            ):
                continue
            values = []
            for group in groups:
                left_to_right = sorted(group, key=lambda box: box["x"])
                values.append(int("".join(box["class"] for box in left_to_right)))
            reading = dict(zip(("sys", "dia", "pulse"), values))
            if all(PLAUSIBLE[key][0] <= reading[key] <= PLAUSIBLE[key][1] for key in PLAUSIBLE):
                spread = sum(
                    sum(abs(box["y"] - center) / height for box in group)
                    for group, center, height in zip(groups, centers, heights)
                )
                candidates.append((spread, reading))
    unique = {tuple(candidate[1].values()) for candidate in candidates}
    if not candidates:
        return None, "no_valid_three_row_partition"
    if len(unique) != 1:
        return None, "ambiguous_three_row_partition"
    return min(candidates, key=lambda candidate: candidate[0])[1], "candidate"


def all_source_images(source_dir: Path) -> list[Path]:
    return sorted(
        path for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "images" in path.parts
    )


def make_contact_sheets(rows: list[dict], boxes_by_id: dict[str, list[dict]]) -> None:
    sheets = REVIEW_DIR / "contact_sheets"
    sheets.mkdir(parents=True, exist_ok=True)
    for old_sheet in sheets.glob("candidates_*.jpg"):
        old_sheet.unlink()
    tile_width, tile_height, columns, page_size = 360, 300, 4, 24
    for page_start in range(0, len(rows), page_size):
        page = rows[page_start:page_start + page_size]
        canvas = Image.new("RGB", (columns * tile_width, 6 * tile_height), "#eeeeee")
        draw = ImageDraw.Draw(canvas)
        for offset, row in enumerate(page):
            path = ROOT / row["local_file"]
            image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
            overlay = ImageDraw.Draw(image)
            for box in boxes_by_id[row["id"]]:
                if box["class"] not in set("0123456789"):
                    continue
                left = (box["x"] - box["width"] / 2) * image.width
                top = (box["y"] - box["height"] / 2) * image.height
                right = (box["x"] + box["width"] / 2) * image.width
                bottom = (box["y"] + box["height"] / 2) * image.height
                overlay.rectangle((left, top, right, bottom), outline="#00ff5a", width=max(2, image.width // 500))
                overlay.text((left, top), box["class"], fill="#ff2020")
            image.thumbnail((tile_width - 12, tile_height - 48))
            x = (offset % columns) * tile_width
            y = (offset // columns) * tile_height
            canvas.paste(image, (x + (tile_width - image.width) // 2, y + 4))
            reading = row["annotation_reading"]
            proposed = (
                f"{reading['sys']}/{reading['dia']} p{reading['pulse']}"
                if reading else "manual transcription"
            )
            caption = f"{row['id']}  {proposed}"
            draw.text((x + 5, y + tile_height - 38), caption, fill="black")
        page_number = page_start // page_size + 1
        canvas.save(sheets / f"candidates_{page_number:03d}.jpg", quality=90)


def audit_overlap(rows: list[dict], threshold: int = 18) -> dict:
    candidate_paths = {row["id"]: ROOT / row["local_file"] for row in rows}
    candidate_hashes = {
        row_id: (sha256(path), dhash256(path)) for row_id, path in candidate_paths.items()
    }
    excluded_roots = {SOURCES_DIR.resolve(), REVIEW_DIR.resolve(), OUTPUT_DIR.resolve()}
    references = []
    dataset_root = ROOT / "dataset"
    for path in dataset_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = path.resolve()
        if any(root == resolved or root in resolved.parents for root in excluded_roots):
            continue
        references.append((path, sha256(path), dhash256(path)))

    cross_source = []
    for row_id, (digest, difference_hash) in candidate_hashes.items():
        for path, reference_digest, reference_hash in references:
            distance = hamming(difference_hash, reference_hash)
            if digest == reference_digest or distance <= threshold:
                cross_source.append({
                    "candidate_id": row_id,
                    "reference": path.relative_to(ROOT).as_posix(),
                    "exact": digest == reference_digest,
                    "dhash256_distance": distance,
                })

    within_candidates = []
    ids = sorted(candidate_hashes)
    for index, left_id in enumerate(ids):
        left_digest, left_hash = candidate_hashes[left_id]
        for right_id in ids[index + 1:]:
            right_digest, right_hash = candidate_hashes[right_id]
            distance = hamming(left_hash, right_hash)
            if left_digest == right_digest or distance <= threshold:
                within_candidates.append({
                    "left_id": left_id,
                    "right_id": right_id,
                    "exact": left_digest == right_digest,
                    "dhash256_distance": distance,
                })
    return {
        "method": "SHA-256 exact match plus 256-bit dHash visual-review candidates",
        "dhash256_threshold": threshold,
        "candidate_images": len(rows),
        "reference_images": len(references),
        "cross_reference_matches": cross_source,
        "within_candidate_matches": within_candidates,
    }


def prepare() -> None:
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    candidates = []
    rejections = []
    boxes_by_id = {}
    for source in SOURCES:
        source_dir = SOURCES_DIR / source.key
        data_yaml = next(iter(sorted(source_dir.rglob("data.yaml"))), None)
        if data_yaml is None:
            raise FileNotFoundError(f"Missing {source.key}; run the download command first")
        class_names = parse_class_names(data_yaml)
        if not set("0123456789").issubset(class_names):
            raise ValueError(f"{source.key}: expected digit classes 0..9, got {class_names}")
        for ordinal, image_path in enumerate(all_source_images(source_dir), 1):
            row_id = f"{source.key}_{ordinal:04d}"
            split = next((name for name in ("train", "valid", "test") if name in image_path.parts), "unknown")
            if source.key == "ega_v5" and split == "train":
                rejections.append({
                    "id": row_id,
                    "local_file": image_path.relative_to(ROOT).as_posix(),
                    "reason": "source_train_split_contains_generated_augmentations",
                })
                continue
            try:
                boxes = annotation_boxes(image_path, class_names)
                reading, reason = reading_from_digit_boxes(boxes)
            except (FileNotFoundError, ValueError) as error:
                boxes, reading, reason = [], None, str(error)
            boxes_by_id[row_id] = boxes
            candidates.append({
                "id": row_id,
                "local_file": image_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(image_path),
                "source": source.title,
                "source_url": source.page,
                "license": source.license,
                "annotation_reading": reading,
                "annotation_status": reason,
                "answer_source": "Proposed from source digit annotations; manual verification required",
            })

    review = []
    for row in candidates:
        proposal = row["annotation_reading"] or {key: None for key in PLAUSIBLE}
        review.append({
            "id": row["id"],
            "decision": "pending",
            "privacy_clear": False,
            "overlap_clear": False,
            "sys": proposal["sys"],
            "dia": proposal["dia"],
            "pulse": proposal["pulse"],
            "note": "",
        })
    (REVIEW_DIR / "candidates.json").write_text(json.dumps(candidates, indent=2) + "\n")
    (REVIEW_DIR / "review.json").write_text(json.dumps(review, indent=2) + "\n")
    (REVIEW_DIR / "automatic_rejections.json").write_text(json.dumps(rejections, indent=2) + "\n")
    overlap = audit_overlap(candidates)
    (REVIEW_DIR / "overlap_audit.json").write_text(json.dumps(overlap, indent=2) + "\n")
    make_contact_sheets(candidates, boxes_by_id)
    print(f"candidates={len(candidates)} automatic_rejections={len(rejections)}")
    print(f"cross_reference_matches={len(overlap['cross_reference_matches'])}")
    print(f"within_candidate_matches={len(overlap['within_candidate_matches'])}")
    print(f"review={REVIEW_DIR.relative_to(ROOT) / 'review.json'}")


def freeze() -> None:
    candidates = {row["id"]: row for row in json.loads((REVIEW_DIR / "candidates.json").read_text())}
    reviews = json.loads((REVIEW_DIR / "review.json").read_text())
    unknown = sorted({row["id"] for row in reviews} - set(candidates))
    if unknown:
        raise ValueError(f"Review contains unknown ids: {unknown[:5]}")
    pending = [row["id"] for row in reviews if row["decision"] == "pending"]
    if pending:
        raise ValueError(f"Review is incomplete; pending items include {pending[:5]}")

    frozen = []
    for review in reviews:
        if review["decision"] == "exclude":
            continue
        if review["decision"] != "accept":
            raise ValueError(f"{review['id']}: decision must be accept, exclude, or pending")
        if not review["privacy_clear"] or not review["overlap_clear"]:
            raise ValueError(f"{review['id']}: privacy_clear and overlap_clear are required")
        reading = {key: int(review[key]) for key in PLAUSIBLE}
        if not all(PLAUSIBLE[key][0] <= reading[key] <= PLAUSIBLE[key][1] for key in PLAUSIBLE):
            raise ValueError(f"{review['id']}: implausible manually reviewed reading {reading}")
        source = candidates[review["id"]]
        frozen.append({
            "id": f"external_v3_readable_{len(frozen) + 1:04d}",
            "source_id": source["id"],
            "local_file": source["local_file"],
            "sha256": source["sha256"],
            **reading,
            "source": source["source"],
            "source_url": source["source_url"],
            "license": source["license"],
            "answer_source": "Source digit annotations, manually verified before OCR inference",
            "review_note": review["note"],
        })
    if not frozen:
        raise ValueError("Review accepted no images")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "readable_truth.json").write_text(json.dumps(frozen, indent=2) + "\n")
    (OUTPUT_DIR / "refusal_truth.json").write_text("[]\n")
    source_counts = {
        source.title: sum(row["source"] == source.title for row in frozen)
        for source in SOURCES
    }
    metadata = {
        "schema_version": 1,
        "role": "test-only, model-blind external benchmark",
        "permitted_uses": ["evaluation", "regression testing", "error analysis"],
        "prohibited_uses": ["training", "validation", "threshold calibration", "augmentation"],
        "readable_cases": len(frozen),
        "refusal_cases": 0,
        "source_counts": source_counts,
        "selection": "Digit-label parsing followed by manual transcription, privacy, scope, and overlap review before inference",
        "truth_sha256": sha256(OUTPUT_DIR / "readable_truth.json"),
        "limitations": [
            "The source images were public rather than prospectively captured for this study.",
            "Source annotations proposed the transcription; a human verified it before inference.",
            "The benchmark is device-image evidence, not clinical safety validation.",
        ],
    }
    (OUTPUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"frozen_readable={len(frozen)}")
    print(f"truth_sha256={metadata['truth_sha256']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("download", help="download both pinned Roboflow exports")
    subparsers.add_parser("prepare", help="parse labels and create a blind-review pack")
    subparsers.add_parser("freeze", help="freeze a fully completed manual review")
    args = parser.parse_args()

    if args.command == "download":
        api_key = roboflow_api_key()
        if not api_key:
            raise SystemExit(
                "ROBOFLOW_API_KEY is not set; provide it via the environment or the ignored repo .env file"
            )
        for source in SOURCES:
            download_source(source, api_key)
    elif args.command == "prepare":
        prepare()
    else:
        freeze()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
