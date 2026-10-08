# External test v2: construction, audit and results

This document records the external blood-pressure-monitor image testing work
completed on 8 October 2026. The resulting local suite contains 100 readable
three-row displays and 25 images that should produce no reading. It is now a
**test-only, consumed regression benchmark**. It must not be used for training,
validation, threshold calibration or augmentation.

No dataset images, per-image answers, credentials, infrastructure identifiers
or detailed inference records are included in Git. Images, frozen manifests,
contact sheets and detailed reports remain under ignored local paths.

## Objective and starting point

The earlier external web pilot contained only eight manually transcribed images.
The single-pass reader produced three exact triplets and the shipped staged
reader produced six; staged candidate precision was 100% at 75% coverage. Eight
correlated examples were too few for a useful generalization estimate. The v2
work therefore aimed to create:

- 100 readable images from public sources, frozen before inference;
- at least 25 relevant negative or unreadable cases for refusal testing;
- source provenance and image checksums;
- duplicate and training-overlap audits;
- a repeatable staged ONNX evaluation that reports accuracy, precision,
  coverage, refusals, confidence intervals and per-source results.

## Collection and provenance

Three acquisition paths were explored.

1. `tools/collect_openverse_bp_images.py` created the first pilot and its local
   provenance manifest. A later expansion attempt was stopped after the public
   API returned repeated rate limits; it contributed no v2 readable cases.
2. `tools/collect_wikimedia_bp_images.py` queried Wikimedia Commons in multiple
   languages, retained only Public Domain, CC BY and CC BY-SA results, recorded
   each source page and license, and downloaded 68 unique review candidates.
3. The public [Blood Pressure Monitor Digit Reader v9 dataset](https://universe.roboflow.com/naphop/blood-pressure-monitor-digit-reader/dataset/9)
   supplied a YOLOv8 test split of 215 images. The dataset page declares CC BY
   4.0. Only the test split and root metadata were transferred locally.

The final readable set contains:

| Source | Cases | Answer construction |
| --- | ---: | --- |
| Roboflow Universe v9 test split | 90 | Source digit/row annotations, visually cross-checked before inference |
| Wikimedia Commons | 10 | Manual transcription of visible SYS, DIA and pulse rows before inference |

The refusal set contains 25 Wikimedia images of analog, powered-off,
disassembled or cuff-only devices with no readable three-row digital result.
Candidates containing identifiable people, unrelated search results and known
overlap with the earlier web pilot were excluded. Wikimedia license and source
page details remain attached to every local manifest entry.

The Roboflow download used the pre-existing secret mechanism on temporary
Runpod CPU Pods. Secret values were never printed or placed in files. The
downloaded archive was checksum-verified during transfer. Both temporary CPU
Pods were stopped and then terminated after the data and audit hashes were
secured locally. The existing training Pod and its shared volume were left
unchanged.

## Review, annotation and selection

Contact sheets were generated for all 215 Roboflow test images and all 68
Wikimedia candidates. The review removed:

- missing or ambiguous target rows;
- source annotations polluted by dates or unrelated display digits;
- repeated crops and same-session clusters;
- hospital displays outside the intended three-row home-monitor scope;
- Commons images lacking all three target values;
- identifiable people and irrelevant search results.

The readable selection was deterministic after the exclusion list was frozen.
The local builder is `tools/build_external_test_v2.py`. It parses the source
YOLO labels, applies reviewed exclusions, validates physiological-format ranges,
adds the manually transcribed Commons cases from an ignored local truth file,
computes SHA-256 for every image, and writes the manifests. The per-image
readings remain outside version control; only aggregate results and
reproducibility tooling are committed.

It writes:

```text
dataset/external_test_v2/metadata.json
dataset/external_test_v2/readable_truth.json
dataset/external_test_v2/refusal_truth.json
```

The two answer manifests contain 125 unique local cases. Their file digests are:

```text
readable_truth.json  c37efaf6471190d199ef2ef8d916855662654619ad36242dcf3fdfbf0a65010d
refusal_truth.json   ed089c0643e09d30264e95cc5bf21b570ad75237f2b34aeb5adf7f75413fa7c3
```

The manifests were written before model inference. The evaluator recomputes
every image SHA-256 and aborts on a mismatch, preventing silent benchmark drift.

## Duplicate and leakage audit

The original public detector corpus on the persistent training volume contained
309 train/validation/test images. A temporary CPU Pod computed image-free audit
records containing relative paths, SHA-256 digests and 256-bit difference hashes.
Only this hash inventory was transferred for comparison.

`tools/audit_external_test.py` compared all v2 candidates with that inventory:

| Audit | Roboflow candidates | Wikimedia candidates |
| --- | ---: | ---: |
| Candidates checked | 215 | 68 |
| Exact SHA-256 matches | 0 | 0 |
| dHash distance ≤18 | 13 | 0 |

The 13 Roboflow near-matches were visually consistent with re-encoded or
preprocessed copies from the original corpus. They formed one later source block
and were all outside the selected 90 cases. The closest Wikimedia candidate had
dHash distance 71, leaving a clear separation from the threshold. A separate
within-set perceptual review removed repeated crops and capped same-device or
same-session clusters.

This audit reduces obvious leakage; it cannot prove that no related scene,
device, upstream dataset or unrecognized transformation appeared in training.

## Test-only enforcement

The local metadata explicitly permits evaluation, regression testing and error
analysis, and prohibits training, validation, calibration and augmentation.
`prototype/prepare_data.py` rejects known external-test paths if one is
accidentally introduced into a generated detector training or validation
manifest.

All raw images and detailed records remain covered by repository ignore rules:

```text
dataset/
prototype/reports/
```

No raw test data, detailed records, credentials or infrastructure identifiers
are included in the commit that documents this work.

## Evaluation implementation

`prototype/evaluate_external_v2.py` runs the frozen suite with the bundled ONNX
models and released `v2-adaptive3` configuration:

| Component | Frozen value |
| --- | --- |
| Detector SHA-256 | `8a7dafaed0aa9056308171d57d024343049cff4ba1a9adfc54d274c38a31fb07` |
| Digit classifier SHA-256 | `f41a41f9138c7c80d7e7fa1c3ed5afe93c3f5beacb3efd861ba3222b597cd732` |
| Detector minimum score | 0.20 |
| Candidate acceptance score | 0.25 |
| Digit recognizer minimum score | 0.50 |

The Python reference applies the production policy in this order:

1. one full-frame detector and digit-recognizer pass;
2. after no plausible result on a portrait image, two fixed central crops, used
   only when both produce the same plausible candidate;
3. after continued refusal, a row-derived crop with two lighting-normalization
   radii, used only when both views produce the same plausible reading.

The reference uses the shipped model, assembly and adaptive-crop Python code. It
is not a pixel-exact browser-runtime test: browser canvas resizing, RGB/BGR
handling and rounding can produce small differences. Target-browser testing is
still required before a release claim.

## Results

### Readable images

| Metric | Full-frame baseline | Staged policy |
| --- | ---: | ---: |
| Images | 100 | 100 |
| Returned readings, including review results | 30 | 38 |
| Exact SYS/DIA/pulse triplets | 20 | 27 |
| Exact-triplet accuracy | 20.0% | 27.0% |
| Exact accuracy, 95% Wilson interval | 13.3–28.9% | 19.3–36.4% |
| Candidate outputs | 19 | 24 |
| Correct candidates | 17 | 22 |
| Candidate coverage | 19.0% | 24.0% |
| Candidate precision | 89.5% | 91.7% |
| Candidate precision, 95% Wilson interval | 68.6–97.1% | 74.2–97.7% |
| Correct SYS fields | 25 | 33 |
| Correct DIA fields | 27 | 34 |
| Correct pulse fields | 26 | 34 |

The adaptive stage was selected on eight of the 125 total cases. It recovered
seven additional exact readable triplets; the remaining adaptive result was an
incorrect review result rather than an accepted candidate. No case selected the
portrait fallback. The two incorrect accepted outputs were isolated single-digit
classification errors, one in pulse and one in DIA.

The latest local rerun reported a 250 ms median and 337 ms p95 end-to-end Python
latency. These are workstation observations, not phone-browser latency claims.

### By readable source

| Source | Exact triplets | Candidates | Candidate precision |
| --- | ---: | ---: | ---: |
| Roboflow v9 test | 24/90 (26.7%) | 21/90 (23.3% coverage) | 19/21 (90.5%) |
| Wikimedia Commons | 3/10 (30.0%) | 3/10 (30.0% coverage) | 3/3 (100%) |

The Commons group is far too small for a stable source-specific estimate; its
candidate-precision Wilson interval is 43.9–100%.

### Refusal images

| Metric | Result |
| --- | ---: |
| Images with no readable triplet | 25 |
| Returned readings | 0 |
| Accepted candidate outputs | 0 |
| Safe refusal rate | 25/25 (100%) |
| Safe refusal rate, 95% Wilson interval | 86.7–100% |

The refusal set is deliberately relevant but narrow. It does not cover every
possible confusing display, screen, sign, adversarial input or medical device.

## Comparison with BP Digitizer's Gemini model

The same frozen cases were subsequently processed once by the closed-weights
`gemini-2.5-flash` model selected by BP Digitizer's production configuration.
Gemini produced 92/100 exact readable triplets, compared with Hearth's 27/100,
and both systems were safe on 25/25 negative cases under BP Digitizer's rule
requiring both SYS and DIA. Gemini's median response time was 2.96 seconds,
including network and cloud service time, compared with 250 ms for Hearth's
local Python run. These are different execution environments rather than a
controlled model-speed test.

See [Hearth versus Gemini 2.5 Flash](GEMINI_COMPARISON.md) for definitions,
paired results, complete-output behavior, cost and limitations. The comparison
does not make this consumed regression set independent again.

## Interpretation

The larger set reverses the overly optimistic impression created by the
eight-image pilot. The reader is conservative and rejects all tested negative
cases, but it also refuses many valid displays. The staged policy improves exact
accuracy and coverage without reducing measured candidate precision on this set.
Even so, the lower confidence bound on candidate precision is only 74.2%, and
two accepted readings are wrong. The desired greater-than-90% precision on new
captures therefore remains unproven.

These results are external challenge evidence, not clinical validation:

- 90% of readable cases come from one public source dataset;
- those 90 answers originate from source annotations rather than independent
  double transcription;
- related images and repeated devices weaken independence assumptions behind
  binomial intervals;
- the set was inspected and evaluated in the development workflow and is now
  consumed for future model comparison;
- the run used the Python ONNX reference rather than the exact target browser;
- neither OCR correctness nor refusal behavior establishes medical safety.

Future tuning may use aggregate error categories, but not these images as
training examples or threshold-selection data. A later model still needs a new,
sealed, independently transcribed device/session holdout for unbiased evidence.
The preceding GPU-training changes and validation results are recorded in the
[Runpod training report](RUNPOD_TRAINING.md).

## Reproduction

With the ignored datasets and manifests present locally:

```sh
.venv/bin/python prototype/evaluate_external_v2.py
```

The detailed output is written to:

```text
prototype/reports/external_test_v2.json
```

Supporting commands:

```sh
python tools/build_external_test_v2.py          # dry-run selection check
python tools/build_external_test_v2.py --write  # intentionally refreeze manifests
python tools/audit_external_test.py REFERENCE_HASHES.json CANDIDATE_IMAGE_DIR
```

Do not use `--write` during an ordinary regression run: it changes the frozen
manifest files and metadata. A deliberate refreeze creates a new benchmark
revision and must be documented separately.

## Verification performed

- all 125 selected image checksums matched their frozen manifest values;
- the complete evaluator ran twice with identical accuracy/precision/refusal
  counts;
- Python compilation passed for the collector, builder, audit and evaluator;
- eight Python reading/geometry tests and six metric tests passed;
- three JavaScript reading/crop/adaptive suites passed;
- the training-path rejection guard passed a direct safe/leak test;
- `git diff --check` passed;
- no credentials or secret values appeared in the added scripts or reports;
- temporary Runpod CPU resources were terminated after transfer and audit.

The full detector data preparation command was not rerun in this local checkout
because its generated original-corpus reference manifests are not present here.
The guard is positioned immediately before training/validation path manifests
are written and will execute when that workflow runs where those inputs exist.
