#!/usr/bin/env python3
"""Evaluate TeleOCR as a generic OCR baseline on frozen BP-monitor suites.

The model is asked only to transcribe visible text. A deterministic parser then
uses the known three-row format and physiological-format bounds. It never edits
or substitutes an observed digit. Detailed reports are development artifacts
and must remain ignored; ``--aggregate-output`` writes a shareable summary.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import statistics
import time
from pathlib import Path


FIELDS = ("sys", "dia", "pulse")
PROMPTS = {
    "generic": "Please output the text content from the image.",
    "digits": "Please output all visible digits from the image, preserving their reading order.",
    "structured": (
        "Read the blood pressure monitor display. Return exactly one line as "
        "SYS=<integer> DIA=<integer> PULSE=<integer>. Use UNKNOWN for a field "
        "that is not clearly visible. Do not infer or correct unreadable digits."
    ),
}


def preprocess_image(image, variant: str):
    from PIL import ImageOps

    image = image.convert("RGB")
    if variant == "original":
        return image
    if variant == "upscale":
        return image.resize((image.width * 2, image.height * 2))
    gray = ImageOps.grayscale(image)
    if variant == "grayscale":
        return gray.convert("RGB")
    if variant == "autocontrast-upscale":
        gray = ImageOps.autocontrast(gray)
        return gray.resize((gray.width * 2, gray.height * 2)).convert("RGB")
    raise ValueError(f"unknown preprocessing variant: {variant}")


def plausible(reading: dict[str, int] | None) -> bool:
    return bool(
        reading
        and 50 <= reading["sys"] <= 280
        and 25 <= reading["dia"] <= 180
        and 20 <= reading["pulse"] <= 250
        and reading["sys"] > reading["dia"]
    )


def _label_value(text: str, labels: tuple[str, ...]) -> int | None:
    joined = "|".join(re.escape(label) for label in labels)
    after = re.search(rf"(?:{joined})\s*[:=\-]?\s*(\d{{2,3}})\b", text, re.I)
    if after:
        return int(after.group(1))
    before = re.search(rf"\b(\d{{2,3}})\s*(?:{joined})\b", text, re.I)
    return int(before.group(1)) if before else None


def parse_reading(text: str) -> tuple[dict[str, int] | None, str]:
    """Parse one unambiguous bounded triplet without correcting OCR output."""
    labeled = {
        "sys": _label_value(text, ("sys", "systolic")),
        "dia": _label_value(text, ("dia", "diastolic")),
        "pulse": _label_value(text, ("pulse", "pul", "bpm")),
    }
    if all(value is not None for value in labeled.values()):
        reading = {key: int(value) for key, value in labeled.items()}
        return (reading, "labels") if plausible(reading) else (None, "implausible-labels")

    tokens = [int(token) for token in re.findall(r"(?<!\d)\d{2,3}(?!\d)", text)]
    candidates: set[tuple[int, int, int]] = set()
    for triple in itertools.combinations(tokens, 3):
        reading = dict(zip(FIELDS, triple))
        if plausible(reading):
            candidates.add(triple)
    if len(candidates) == 1:
        return dict(zip(FIELDS, next(iter(candidates)))), "unique-bounded-sequence"
    return None, "ambiguous" if candidates else "no-plausible-triplet"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.5)))
    return ordered[index]


def load_cases(paths: list[Path], expected_refusal: bool) -> list[dict]:
    cases: list[dict] = []
    for path in paths:
        for item in json.loads(path.read_text()):
            image = Path(item["local_file"])
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest != item["sha256"]:
                raise ValueError(f"checksum mismatch for {item['id']}: {image}")
            cases.append({**item, "expected_refusal": expected_refusal})
    return cases


def infer(processor, model, image, prompt: str, max_new_tokens: int) -> str:
    import torch

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
    ]
    chat_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[chat_prompt], images=[image.convert("RGB")], padding=True, return_tensors="pt"
    ).to(device=model.device, dtype=model.dtype)
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs, use_cache=True, max_new_tokens=max_new_tokens, do_sample=False
        )
    generated = output_ids.cpu().tolist()[0][len(inputs.input_ids[0]) :]
    return processor.batch_decode(
        [generated], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()


def summarize(records: list[dict], model_load_ms: float) -> dict:
    readable = [record for record in records if not record["expected_refusal"]]
    negatives = [record for record in records if record["expected_refusal"]]
    returned = [record for record in readable if record["reading"] is not None]
    exact = [record for record in returned if record["exact_triplet"]]
    times = [record["elapsed_ms"] for record in records]
    return {
        "readable_images": len(readable),
        "negative_images": len(negatives),
        "exact_triplets": len(exact),
        "exact_triplet_accuracy": len(exact) / len(readable) if readable else None,
        "returned_readings": len(returned),
        "returned_precision": len(exact) / len(returned) if returned else None,
        "coverage": len(returned) / len(readable) if readable else None,
        "correct_fields": {
            field: sum(record["field_correct"][field] for record in readable)
            for field in FIELDS
        },
        "safe_negatives": sum(record["reading"] is None for record in negatives),
        "median_ms": statistics.median(times) if times else None,
        "p95_ms": percentile(times, 0.95),
        "model_load_ms": model_load_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--truth", type=Path, nargs="+", required=True)
    parser.add_argument("--refusals", type=Path, nargs="*", default=[])
    parser.add_argument("--model", default="XingChen-AGI/TeleOCR")
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="generic")
    parser.add_argument(
        "--preprocess",
        choices=("original", "upscale", "grayscale", "autocontrast-upscale"),
        default="original",
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor

    cases = load_cases(args.truth, False) + load_cases(args.refusals, True)
    if args.limit is not None:
        cases = cases[: args.limit]

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(
        args.model, trust_remote_code=True, use_fast=True, cache_dir=args.cache_dir
    )
    model = AutoModel.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
    ).cuda().eval()
    model_load_ms = (time.perf_counter() - load_started) * 1000

    records = []
    for index, case in enumerate(cases, 1):
        started = time.perf_counter()
        image = preprocess_image(Image.open(case["local_file"]), args.preprocess)
        raw = infer(
            processor,
            model,
            image,
            PROMPTS[args.prompt],
            args.max_new_tokens,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        reading, parse_method = parse_reading(raw)
        expected = None if case["expected_refusal"] else {field: case[field] for field in FIELDS}
        records.append(
            {
                "id": case["id"],
                "expected_refusal": case["expected_refusal"],
                "expected": expected,
                "reading": reading,
                "raw_text": raw,
                "parse_method": parse_method,
                "exact_triplet": reading == expected if expected else False,
                "field_correct": {
                    field: bool(reading and expected and reading[field] == expected[field])
                    for field in FIELDS
                },
                "elapsed_ms": elapsed_ms,
            }
        )
        print(
            json.dumps(
                {
                    "progress": f"{index}/{len(cases)}",
                    "id": case["id"],
                    "reading": reading,
                    "parse_method": parse_method,
                    "elapsed_ms": round(elapsed_ms, 1),
                }
            ),
            flush=True,
        )

    summary = summarize(records, model_load_ms)
    report = {
        "model": args.model,
        "prompt": args.prompt,
        "prompt_text": PROMPTS[args.prompt],
        "preprocess": args.preprocess,
        "parser": "labels, else exactly one bounded three-token subsequence; no correction",
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.aggregate_output:
        args.aggregate_output.parent.mkdir(parents=True, exist_ok=True)
        args.aggregate_output.write_text(
            json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2)
            + "\n"
        )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
