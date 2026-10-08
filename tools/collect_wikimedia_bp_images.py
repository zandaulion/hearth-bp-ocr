#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Collect Wikimedia Commons BP-monitor candidates with provenance.

The output is a local, ignored review queue. Search results and Commons license
metadata are only the first filter: every retained image still requires manual
relevance, privacy, license, and ground-truth review before evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "dataset" / "wikimedia_bp_test_v2_candidates"
API_URL = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "hearth-bp-ocr-dataset-review/1.0 (local research dataset collector)"
ALLOWED_MIME = {"image/jpeg": ".jpg", "image/png": ".png"}
ALLOWED_LICENSE_MARKERS = ("cc0", "public domain", "cc by", "cc-by")
DEFAULT_QUERIES = (
    "blood pressure monitor",
    "digital blood pressure monitor",
    "automatic blood pressure monitor",
    "wrist blood pressure monitor",
    "electronic sphygmomanometer",
    "digital sphygmomanometer",
    "blood pressure monitor display",
    "Omron blood pressure monitor",
    "Beurer blood pressure monitor",
    "Microlife blood pressure monitor",
    "Braun blood pressure monitor",
    "A&D blood pressure monitor",
    "digital tensiometer",
    "Blutdruckmessgerät digital",
    "tensiomètre électronique",
    "tensiómetro digital",
    "misuratore pressione digitale",
    "ciśnieniomierz elektroniczny",
)


def request_json(params: dict[str, str | int]) -> dict:
    url = f"{API_URL}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        return json.load(response)


def clean(value: str | None) -> str | None:
    if not value:
        return None
    text = re.sub(r"<[^>]+>", " ", html.unescape(value))
    return " ".join(text.split())


def metadata_value(metadata: dict, key: str) -> str | None:
    item = metadata.get(key, {})
    return clean(item.get("value")) if isinstance(item, dict) else None


def license_allowed(metadata: dict) -> bool:
    label = " ".join(
        filter(
            None,
            (
                metadata_value(metadata, "LicenseShortName"),
                metadata_value(metadata, "UsageTerms"),
            ),
        )
    ).lower()
    return any(marker in label for marker in ALLOWED_LICENSE_MARKERS)


def search(query: str, continuation: str | None, limit: int) -> dict:
    params: dict[str, str | int] = {
        "action": "query",
        "format": "json",
        "formatversion": 2,
        "generator": "search",
        "gsrsearch": query,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "gsrwhat": "text",
        "prop": "imageinfo|info",
        "inprop": "url",
        "iiprop": "url|mime|size|extmetadata|sha1",
        "iiurlwidth": 1600,
    }
    if continuation:
        params["gsrcontinue"] = continuation
    return request_json(params)


def download(url: str, max_bytes: int) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:
        declared = response.headers.get("Content-Length")
        if declared and int(declared) > max_bytes:
            raise ValueError(f"declared size {declared} exceeds {max_bytes}")
        data = response.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"download exceeds {max_bytes} bytes")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--query", action="append", dest="queries")
    parser.add_argument("--pages", type=int, default=3)
    parser.add_argument("--page-size", type=int, default=50)
    parser.add_argument("--max-images", type=int, default=500)
    parser.add_argument("--max-mib", type=int, default=15)
    args = parser.parse_args()
    if min(args.pages, args.page_size, args.max_images, args.max_mib) < 1:
        parser.error("numeric limits must be positive")
    if args.page_size > 50:
        parser.error("page-size cannot exceed the Commons anonymous limit of 50")

    output = args.output.resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    records = list(previous.get("images", []))
    failures = list(previous.get("failures", []))
    seen_pages = {item["pageid"] for item in records}
    seen_hashes = {item["sha256"] for item in records}
    queries = tuple(args.queries or DEFAULT_QUERIES)
    max_bytes = args.max_mib * 1024 * 1024

    for query in queries:
        continuation = None
        for page_number in range(1, args.pages + 1):
            if len(records) >= args.max_images:
                break
            try:
                payload = search(query, continuation, args.page_size)
            except (OSError, ValueError, json.JSONDecodeError) as error:
                failures.append({"query": query, "page": page_number, "error": str(error)})
                break
            pages = payload.get("query", {}).get("pages", [])
            for page in pages:
                if len(records) >= args.max_images:
                    break
                pageid = page.get("pageid")
                info = (page.get("imageinfo") or [{}])[0]
                metadata = info.get("extmetadata") or {}
                mime = info.get("mime")
                if not pageid or pageid in seen_pages or mime not in ALLOWED_MIME:
                    continue
                if not license_allowed(metadata):
                    continue
                url = info.get("thumburl") or info.get("url")
                if not url:
                    continue
                try:
                    data = download(url, max_bytes)
                except (OSError, ValueError) as error:
                    failures.append({"query": query, "pageid": pageid, "url": url, "error": str(error)})
                    continue
                digest = hashlib.sha256(data).hexdigest()
                seen_pages.add(pageid)
                if digest in seen_hashes:
                    continue
                seen_hashes.add(digest)
                filename = f"{len(records) + 1:04d}_{pageid}{ALLOWED_MIME[mime]}"
                path = images_dir / filename
                path.write_bytes(data)
                records.append(
                    {
                        "local_file": str(path.relative_to(output)),
                        "sha256": digest,
                        "query": query,
                        "pageid": pageid,
                        "title": page.get("title"),
                        "description_url": info.get("descriptionurl") or page.get("canonicalurl"),
                        "original_url": info.get("url"),
                        "download_url": url,
                        "mime": mime,
                        "width": info.get("width"),
                        "height": info.get("height"),
                        "creator": metadata_value(metadata, "Artist"),
                        "credit": metadata_value(metadata, "Credit"),
                        "license": metadata_value(metadata, "LicenseShortName"),
                        "license_url": metadata_value(metadata, "LicenseUrl"),
                        "usage_terms": metadata_value(metadata, "UsageTerms"),
                    }
                )
                print(f"{len(records):04d} {records[-1]['license']} {page.get('title')}", flush=True)
                time.sleep(0.2)
            continuation = payload.get("continue", {}).get("gsrcontinue")
            if not continuation:
                break
            time.sleep(0.5)

    recorded_queries = [item.get("query") for item in records if item.get("query")]
    all_queries = list(dict.fromkeys([*recorded_queries, *previous.get("queries", []), *queries]))
    generated_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "source": "Wikimedia Commons MediaWiki API",
        "source_url": API_URL,
        "queries": all_queries,
        "review_required": True,
        "images": records,
        "failures": failures,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    (output / "README.md").write_text(
        "# Wikimedia BP-monitor candidate review queue\n\n"
        f"Generated {generated_at}; {len(records)} candidates and {len(failures)} failures.\n\n"
        "Do not evaluate or train on this directory as-is. Verify every Commons landing "
        "page and license, remove irrelevant/private/duplicate images, independently "
        "transcribe readable displays, and freeze checksums before the first inference. "
        "Keep all images and readings local and Git-ignored.\n"
    )
    print(f"Saved {len(records)} unique candidates to {output}")
    print(f"Recorded {len(failures)} failures in {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
