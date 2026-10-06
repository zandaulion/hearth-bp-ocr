# Development, evaluation and troubleshooting

## Environment and tests

The browser setup needs a current Node.js runtime and the locked npm packages;
Python's static server uses only the standard library. Python inference/training
uses the pinned [requirements](../prototype/requirements.txt), including CPU
PyTorch. Use a virtual environment and follow the root README's installation
commands. The optional earlier feature experiments have additional requirements
in `evaluation/requirements.txt`; they are not required by the PWA.

From the repository root:

```sh
npm --prefix prototype ci
npm --prefix prototype run vendor
npm --prefix prototype test
python prototype/test_reading.py
python evaluation/test_metrics.py
python examples/read_image.py path/to/your-local-photo.jpg
```

The 20 JavaScript and 11 Python synthetic checks exercise logic; they do not
measure OCR accuracy. The image example runs one Python pass. Use browser
captures to assess the complete browser pipeline, resizing and runtime behavior.

## Data and training workflow

The published models are already runnable. Training is optional. See the root
README for the public-data recipe and [model card](MODEL_CARD.md) for the exact
limits of reproducing the bundled fine-tuned models without private photos.

Keep dataset files under ignored `dataset/`; derived manifests under
`evaluation/artifacts/`; checkpoints under `prototype/models/` and
`prototype/runs/`; and detailed reports under `prototype/reports/`. Do not add
them to Git. Public weights are a narrow exception for two reviewed files.

The Roboflow YOLOv8 export belongs at `dataset/roboflow_bp_display/`, with
`data.yaml` and `train`, `valid`, `test` splits containing `images` and `labels`.
The original class-name order must match the expected order including row name
`10`. `dataset_tools.py` validates it and derives triplets from digit/row
annotations. These references are not independent human transcriptions.

```sh
python evaluation/dataset_tools.py
python prototype/prepare_data.py
python prototype/bootstrap.py
python prototype/train_detector.py --epochs 40 --name detector_v1
python prototype/train_digits.py
python prototype/export_model.py prototype/runs/detector_v1/weights/best.pt
python prototype/calibrate.py --model prototype/web/models/bp-detector.onnx --output prototype/reports/calibration.json --version public-v1
```

Training and export replace local model/config files at the standard paths.
Use a separate checkout when preserving the published release matters. Export
resets config to baseline defaults. The last command above writes a calibration
report without activating its selected thresholds. Manually apply the selected
thresholds and follow the release procedure below.

**Current freeze-script limitation:** `calibrate.py --freeze` assumes the cache
declaration is on line 1 of `sw.js`. The public source has SPDX/copyright headers
there; this option would add a duplicate `const CACHE` and break the service
worker. Do not use `--freeze` unchanged on this revision. Update the existing
cache declaration by its name when preparing a release. The inference pipeline
and integration examples do not call this option.

If using the optional downloader, set `ROBOFLOW_API_KEY` in the environment
rather than a CLI argument that enters shell history. Its API-key route uses the
optional `roboflow` package, which is not in the inference requirements. Browser
download of the correct export is also sufficient. The OAuth helper is not part
of inference or a prerequisite for using the project.

## Training and evaluation command reference

| Script | Inputs/options and behavior |
| --- | --- |
| `prepare_data.py` | Reads generated train/valid reference manifests, writes `prototype/data/train.txt`, `valid.txt`, `bp.yaml`; leaves source images in place. |
| `bootstrap.py` | Downloads `yolo11n.pt` into ignored model storage and checks the font dependency. Requires network. |
| `train_detector.py` | `--epochs 40`, `--size 512`, `--threads 4`, `--freeze 10`, `--name detector_v1`, `--weights`, `--data`, `--lr .002`; CPU training with fixed seed/augmentation. |
| `train_digits.py` | `--epochs 40`, optional `--phone`, `--initial-weights`, `--output-dir`, `--web-dir`; emits `digits.pt`, validation report and browser `bp-digits.onnx`. `--phone` requires your own labeled images at `prototype/data/phone_train/`. |
| `export_model.py` | Positional checkpoint and `--size 512`; exports fixed detector batch/shape with no embedded NMS and writes baseline config. Parity validation is a separate command. |
| `calibrate.py` | Required `--model`, `--output`; optional `--version`; searches detection/acceptance thresholds using validation only. See `--freeze` caveat above. |
| `verify_export.py` | Requires generated validation references and matching local `prototype/models/bp-detector.pt`, `digits.pt`; compares detector and classifier numeric output against ONNX on three validation examples. These checkpoints are not bundled. |
| `evaluate.py` | Required `--model`, `--truth`, `--output`; optional `--digits-model`, `--min-score .25`, `--accept-score .75`, `--size 512`, `--dataset-role`. Does not read released thresholds automatically or run crop fallbacks. |
| `check_views.py` | Required `--truth`; evaluates framing scales 1, .75, .55, writes `prototype/reports/framing_views.json`. Create report directory first. |
| `evaluate_crop_fallback.py` | Historical experiment; requires public validation/test references and untracked curated references; writes original/synthetic comparisons. Not a turnkey new-checkout command or complete adaptive2 benchmark. |
| `evaluation/audit_dataset.py` | Dataset structure/duplicate review; also expects local curated ground truth. |
| `evaluation/digit_experiment.py` | Earlier feature/classifier experiment, given annotation locations. Does not establish end-to-end reading quality. |
| `evaluation/contact_sheets.py` | Makes local visual review sheets using curated/public files; output contains images/readings and stays ignored. |

Available `evaluate.py --dataset-role` values: `training`, `validation`,
`first-held-out-test`, `development`, `regression`. Changing the label does not
make previously used images independent.

## Reference format and metric meanings

Evaluation ground truth is a JSON array. This is a synthetic format example,
not an actual personal measurement:

```json
[
  {"id": "example-001", "local_file": "dataset/my-session/example.jpg", "sys": 120, "dia": 80, "pulse": 70}
]
```

Use null for an unknown field; never invent a reference. `local_file` resolves
against the repository root. Keep files and answers private unless explicitly
cleared for publication. For a baseline v2 ONNX evaluation, pass the matching
configuration values explicitly:

```sh
python prototype/evaluate.py --model prototype/web/models/bp-detector.onnx --digits-model prototype/web/models/bp-digits.onnx --truth evaluation/my_ground_truth.json --output prototype/reports/my-evaluation.json --min-score 0.20 --accept-score 0.25 --dataset-role development
```

`benchmark.summarize(records)` expects records with `expected`, `prediction`,
`accepted`, `error`, `elapsed_ms`, `field_correct`, and `exact_triplet` as produced
by the evaluator. It computes:

| Metric | Denominator and meaning |
| --- | --- |
| Exact triplet accuracy | Correct SYS/DIA/pulse together / images with all three reference values. Refusals and errors are misses. |
| Candidate precision | Correct complete candidate readings / complete-reference candidate readings. Undefined (`null`) with no candidates. |
| Candidate coverage | Complete-reference candidates / complete-reference images. |
| Per-field accuracy | Correct field values / references where that field is known. |
| Returned readings | All non-null predictions, including review results. |
| Latency | Sorted recorded times; includes failures; median is the upper middle element and p95 uses nearest rank. |
| Wilson interval | 95% interval for binomial proportions; returns null with no observations. Correlated samples weaken its assumptions. |

`accepted` means `status == 'candidate'` in these metrics; it does not mean a
human confirmed the measurement. Report review and retake outcomes alongside
precision. Preserve original failed capture results if later used for tuning.
Keep truly new sessions/devices separate from training and selection. One photo
read six times provides latency observations, not six independent accuracy tests.

## Android development tools

Use the [Android skill](../skills/lenovo-android/SKILL.md) for trusted SSH setup,
fresh device discovery, named selection and app/CDP tunnels. Supply your own
connection configuration. Do not reuse a stale wireless serial or silently
substitute a different phone.

`android_browser.mjs` requires a Node runtime with global `WebSocket` (the
development environment used Node 24), a loopback CDP tunnel and exactly one
matching app tab. It accepts `status`, `focus`, `refresh`, `camera`, `capture`,
`close`, `diagnose`, `crop-diagnose`, `fallback-diagnose`, plus `--cdp-port`,
`--url`, `--report-prefix`. Diagnostic replays are not new captures. Reports
can contain readings and remain local. A status command does not take a photo.

For the optional repeated-sample harness:

```sh
python prototype/android_test_server.py --port 8875 --output prototype/reports/android-session.json
```

Open `/android-test.html` through the configured phone tunnel. Supply local
`prototype/web/samples/test-manifest.json`:

```json
[
  {"file": "example.jpg", "expected": {"sys": 120, "dia": 80, "pulse": 70}, "role": "development"}
]
```

The harness repeats each sample six times, records device/runtime details and
readings in localStorage, and POSTs metrics to `/android-test-result`. Its server
binds loopback and accepts up to 100,000 bytes of JSON; it has no authentication
and is not a production API. The offline phase is selected with `?phase=offline`;
`?phase=collect` retrieves the stored offline result for submission after
reconnection. The current offline server output name is hard-coded to
`android_a52_offline.json`, even with a custom online `--output` path. Its service
worker adds harness pages but does not automatically precache private sample
images/manifests: explicitly arrange/cache test fixtures before claiming an
offline replay succeeded. Use a separate test origin from the normal app.

## Release procedure

1. Freeze models, thresholds and preprocessing together; record source revision,
   dataset roles, provenance and results. Keep private images/answers out of Git.
2. Verify model shape/class order, export parity when checkpoints exist, both
   ONNX hashes and metadata. Never publish an arbitrary `.pt`/pickle checkpoint.
3. Update `models/config.json`, `models/manifest.json`, the model card and the
   explicit reviewed hashes in `tools/check_public_tree.py` for a changed export.
   The worker does not enforce those hashes itself.
4. Update the existing `const CACHE` value in `sw.js` and deploy imports, models,
   config and runtime as one version. Keep dependency license notices.
5. Run relevant synthetic tests and the actual target browser/device checks;
   verify clean installation, repeat reads and offline reload if advertised.
6. Stage only intended public files, run `git diff --cached --check` and
   `python tools/check_public_tree.py`, and inspect the staged diff before push.

The checker scans Git's index blobs, not arbitrary untracked files. It permits
only two exact model hashes and the two named generated PNG icons as binary
exceptions. Passing the checker supplements human review; it is not a universal
privacy or license audit.

## Troubleshooting

| Symptom | Check and next action |
| --- | --- |
| Reader unavailable / model fetch fails | Check all relative asset URLs, HTTP status, MIME types and that an SPA router did not return HTML. Run `npm run vendor`. |
| WASM initialization or allocation error | Confirm pinned runtime files match each other; check browser version, console and memory. Chrome update resolved one tested Fold4 failure. A temporary memory-cap experiment was unstable and is not shipped. |
| Camera absent or denied | Use HTTPS/localhost, check browser permission and existing tracks, or offer file selection. SSH connectivity alone does not grant camera permission. |
| Values differ from Python | Check BGR/RGB, image orientation, interpolation, crop sizes, explicit config thresholds and missing Python fallback stages. Compare actual processed images, not only model files. |
| Digits shifted or row detected as 2 | Check the two distinct class orders and letterbox inversion. |
| Mostly retakes | Inspect framing, glare, missing pulse row and raw detections; retain refusal instead of filling expected values. |
| Boxes in the wrong place | Distinguish `[x1,y1,x2,y2]` boxes from `[x,y,w,h]` crops; apply host resize/crop transforms once. |
| New code/model seems ignored | Check service-worker version and cached asset set; invalidate only the relevant app cache or use a fresh development origin. |
| UI hangs after timeout | Reject the pending operation, dispose the worker and recreate it; a timer alone cannot cancel inference. |
| Python imports fail | Use the intended virtual environment, install pinned dependencies including PyTorch, and make the repository root importable. |
| Publication scan blocks models | Do not broaden the ignore exceptions; verify provenance/metadata and intentionally update only reviewed hashes. |

For reliability limits, see [results](RESULTS.md). Neither a high detection score
nor two agreeing transformed views proves a transcription is correct.
