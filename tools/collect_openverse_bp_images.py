#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Collect openly licensed blood-pressure-monitor image candidates.

The output is an ignored, local review queue. Images are not annotations and
must not be used for training until a person verifies relevance, licensing,
privacy, and labels. Openverse indexes third-party license claims, so the
original landing page remains part of every record for later verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dataset" / "openverse_bp_candidates"
API_URL = "https://api.openverse.org/v1/images/"
USER_AGENT = "hearth-bp-ocr-openverse-collector/1.0 (dataset provenance research)"
ALLOWED_LICENSES = "cc0,pdm,by,by-sa"
ALLOWED_CONTENT_TYPES = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
}
DEFAULT_QUERIES = (
    "blood pressure monitor",
    "digital blood pressure",
    "blood pressure machine",
    "digital sphygmomanometer",
)


def request_json(url: str) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def search(query: str, page: int, page_size: int) -> dict:
    params = urllib.parse.urlencode(
        {
            "q": query,
            "license": ALLOWED_LICENSES,
            "extension": "jpg,jpeg,png",
            "mature": "false",
            "filter_dead": "true",
            "page": page,
            "page_size": page_size,
        }
    )
    return request_json(f"{API_URL}?{params}")


def extension_for(content_type: str, url: str) -> str | None:
    normalized = content_type.split(";", 1)[0].strip().lower()
    if normalized in ALLOWED_CONTENT_TYPES:
        return ALLOWED_CONTENT_TYPES[normalized]
    suffix = Path(urllib.parse.urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return ".jpg"
    if suffix == ".png":
        return suffix
    guessed, _ = mimetypes.guess_type(url)
    return ALLOWED_CONTENT_TYPES.get(guessed or "")


def download(url: str, max_bytes: int) -> tuple[bytes, str]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError(f"declared size {declared} exceeds {max_bytes}")
        data = response.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError(f"download exceeds {max_bytes} bytes")
        content_type = response.headers.get("Content-Type", "")
    suffix = extension_for(content_type, url)
    if suffix is None:
        raise ValueError(f"unsupported content type {content_type!r}")
    return data, suffix


def manifest_record(item: dict, query: str, local_file: str, sha256: str) -> dict:
    return {
        "local_file": local_file,
        "sha256": sha256,
        "query": query,
        "openverse_id": item.get("id"),
        "title": item.get("title"),
        "creator": item.get("creator"),
        "creator_url": item.get("creator_url"),
        "license": item.get("license"),
        "license_version": item.get("license_version"),
        "license_url": item.get("license_url"),
        "attribution": item.get("attribution"),
        "source": item.get("source"),
        "provider": item.get("provider"),
        "foreign_landing_url": item.get("foreign_landing_url"),
        "original_url": item.get("url"),
        "width": item.get("width"),
        "height": item.get("height"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pages", type=int, default=2)
    parser.add_argument("--page-size", type=int, default=20)
    parser.add_argument("--max-images", type=int, default=60)
    parser.add_argument("--max-mib", type=int, default=20)
    parser.add_argument("--query", action="append", dest="queries")
    args = parser.parse_args()

    if args.pages < 1 or args.page_size < 1 or args.max_images < 1:
        parser.error("pages, page-size, and max-images must be positive")

    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous_manifest: dict = {}
    existing = []
    if manifest_path.exists():
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = previous_manifest.get("images", [])

    records = list(existing)
    seen_ids = {record.get("openverse_id") for record in records}
    seen_hashes = {record.get("sha256") for record in records}
    failures: list[dict] = list(previous_manifest.get("failures", []))
    queries = tuple(args.queries or DEFAULT_QUERIES)
    recorded_queries = [record.get("query") for record in records if record.get("query")]
    all_queries = list(
        dict.fromkeys([*recorded_queries, *previous_manifest.get("queries", []), *queries])
    )
    max_bytes = args.max_mib * 1024 * 1024

    for query in queries:
        for page in range(1, args.pages + 1):
            if len(records) >= args.max_images:
                break
            try:
                payload = search(query, page, args.page_size)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                failures.append({"query": query, "page": page, "error": str(error)})
                continue

            for item in payload.get("results", []):
                if len(records) >= args.max_images:
                    break
                identifier = item.get("id")
                url = item.get("url")
                if not identifier or not url or identifier in seen_ids:
                    continue
                try:
                    data, suffix = download(url, max_bytes)
                except (OSError, ValueError, urllib.error.HTTPError) as error:
                    failures.append(
                        {"query": query, "openverse_id": identifier, "url": url, "error": str(error)}
                    )
                    continue
                digest = hashlib.sha256(data).hexdigest()
                seen_ids.add(identifier)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                filename = f"{len(records) + 1:04d}_{identifier}{suffix}"
                path = images_dir / filename
                path.write_bytes(data)
                records.append(
                    manifest_record(
                        item,
                        query,
                        str(path.relative_to(output)),
                        digest,
                    )
                )
                print(f"{len(records):04d} {item.get('license')} {item.get('title')}")
                time.sleep(0.15)

    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source": "Openverse API",
        "source_url": API_URL,
        "queries": all_queries,
        "allowed_licenses": ALLOWED_LICENSES.split(","),
        "review_required": True,
        "images": records,
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Openverse BP-monitor candidate review queue\n\n"
        f"Generated {generated_at}. Contains {len(records)} downloaded candidates and "
        f"{len(failures)} recorded download/search failures.\n\n"
        "Do not train on this directory as-is. Review each original landing page, verify "
        "the recorded license and attribution, reject images with people or identifying "
        "information, remove irrelevant/duplicate images, and annotate approved images "
        "into a separate dataset split. Never commit the images or displayed readings.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(records)} unique candidates to {output}")
    print(f"Recorded {len(failures)} failures in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
