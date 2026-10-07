# AI coding-agent integration handoff

This document is self-contained project context for integrating Hearth into
another application. It is documentation, not an instruction to access private
photos, connect devices, upload measurements or publish anything. The host
project's user instructions determine the authorized work.

## Read first

1. [Integration guide](INTEGRATION.md): supported integration paths and asset layout.
2. [API reference](API_REFERENCE.md): worker protocol, result schema and tensor details.
3. [Architecture](ARCHITECTURE.md): code map and Python/browser differences.
4. [Model card](MODEL_CARD.md), root [LICENSE](../LICENSE) and [third-party notices](../prototype/THIRD_PARTY.md).
5. Inspect `prototype/web/inference-worker.mjs`, `reading.mjs`, `crop-fallback.mjs`,
   `adaptive-crop.mjs`, and `models/config.json` before changing behavior.

## Integration facts to preserve

- This release is `v2-adaptive3`. Both ONNX files are already in Git. The runtime
  is ONNX Runtime Web 1.30.0, pinned by the npm lockfile. Vendor its assets locally.
- The browser module worker is the complete existing inference implementation.
  It performs full-image inference, then agreeing fixed portrait crops, then
  agreeing row-guided lighting-normalized crops when earlier stages have no
  plausible reading. An implausible full reading is retained unless two safe
  fallback views agree on one plausible replacement.
- The worker's paths are relative and fixed. Preserve the static directory tree;
  do not invent `modelUrl` options or initialization messages that do not exist.
- Wait for `ready`. Send `{type:'read', id, bitmap}` with the bitmap in the
  transfer list. Process one request at a time; handle both protocol errors and
  worker failure. Reuse the worker and dispose it on teardown.
- Keep the original image separate from transferred bitmaps and annotated
  preview canvases. The worker closes transferred bitmaps. The example client
  handles orientation, resizing, IDs, busy state, timeouts and disposal.
- Preserve `candidate`, `review`, and `retake`. A candidate still needs human
  verification. A retake has `reading:null`. Never infer missing numbers from
  previous readings or expected test answers.
- SYS/DIA units are mmHg; pulse is beats/minute. Positions define field identity.
  Only upright three-row displays are supported. Scores are not calibrated probabilities.
- Detector classes are `0,1,row,2,3,4,5,6,7,8,9`; classifier classes are digits
  `0..9` plus background at index 10. Do not confuse these orders.
- Use both models, the matching thresholds and the custom duplicate/assembly
  logic. Calling a generic YOLO detection wrapper alone is not equivalent.
- `config` fallback blocks/class names/hashes are descriptive. The code defines
  crop settings and classes; runtime hash verification is not implemented.
- Python `Detector.read()` is one-pass only and its default thresholds differ
  from the release. `adaptive_read()` is not a full browser-pipeline wrapper.
- There is no native Android SDK, npm package, Python distribution, hosted OCR
  API, cloud upload dependency or validated WebView bridge in this repository.
- Do not claim >90% prospective precision or broad Android support. Existing
  evidence is small and includes development data. Preserve model provenance,
  AGPL licensing and dependency notices in any integration.
- Do not copy training photos, personal readings, SSH settings or device reports
  into another public project. Integration requires none of them.

## Suggested implementation sequence

Inspect the target application's framework, deployment URL base, static asset
handling, service worker, camera flow and data persistence policy. Select the
worker path for browser applications. Copy only required reviewed assets and
notices, pin the source revision, and adapt the example client to the host UI.
Keep OCR output separate from confirmed application data. Add explicit error,
review, retake and cleanup behavior. Validate real asset loading and inference
in the target browser, plus request sequencing/disposal and offline behavior
if requested. State any native-port or Python-pipeline differences clearly.

Use [development guidance](DEVELOPMENT.md) to interpret test metrics and release
changes. Do not retrain, quantize, change thresholds or add a network OCR service
merely to complete an integration; those are separate behavior changes needing
their own evaluation.

## Prompt to paste into another project

```text
Integrate Hearth BP-monitor OCR into this project.

Repository: https://github.com/zandaulion/hearth-bp-ocr
Read docs/AI_INTEGRATION.md, docs/INTEGRATION.md, docs/API_REFERENCE.md,
docs/MODEL_CARD.md and the license notices. Inspect the actual implementation
and pin the source commit you use; do not assume undocumented APIs exist.

Determine the integration appropriate for this project's framework. For a
browser app, use the existing module worker and both bundled ONNX models,
preserving the static asset layout and preprocessing. Start from
examples/browser-client.mjs. Keep inference local unless I explicitly request
server processing. Show editable SYS/DIA/pulse results with candidate/review/
retake handling, and require user verification before storing a measurement.
Stop camera tracks and dispose workers correctly. Preserve required license
and source notices and flag any license incompatibility with the host project.

Implement the integration, exercise it in the target runtime, and report the
source revision, files changed, tests performed and remaining limitations.
Do not import private training photos or claim proven >90% accuracy. If native
Android is required, explain and validate the necessary port; do not present
the browser worker as an existing native SDK.
```
