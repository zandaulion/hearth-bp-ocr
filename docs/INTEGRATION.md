# Integrating Hearth into another project

## Choose the integration boundary

| Host project | Path | What you reuse |
| --- | --- | --- |
| Website, PWA, React/Vue/Svelte or plain JS | Static assets plus module worker | Full released pipeline; recommended starting point. |
| Existing Android web app | Same browser/PWA integration | Full pipeline, subject to that browser's support. |
| Native Android/Kotlin/Java | New ONNX Runtime integration and preprocessing/decoder port | Weights can be reused; there is no native Android SDK in this repo. |
| Python application/backend | `Detector` plus explicit preprocessing/fallback choices | Existing one-pass CPU implementation; not automatic PWA parity. |
| A link to a standalone tool | Deploy `prototype/web/` | Existing UI, camera, verification and clipboard flow. |

For inference, no Roboflow account, API key, dataset download, Python training,
friend's starter code or external inference server is needed. Browser runtime
files must be installed once using npm and shipped with your assets.

The project and bundled weights are AGPL-3.0-only. Preserve license and upstream
notices, and assess the AGPL obligations of the combined application before
incorporating this into a proprietary product. A separate worker, iframe or
service is not by itself a license exemption. See [LICENSE](../LICENSE),
[model provenance](MODEL_CARD.md) and [third-party notices](../prototype/THIRD_PARTY.md).

## Browser assets

Start with a pinned repository commit. Install the exact npm lockfile:

```sh
cd prototype
npm ci
npm run vendor
cd ..
```

For a custom host UI, copy the following tree into a same-origin static directory
such as `public/hearth/`. Preserve these relative paths:

```text
hearth/
  inference-worker.mjs
  reading.mjs
  crop-fallback.mjs
  adaptive-crop.mjs
  models/
    bp-detector.onnx
    bp-digits.onnx
    config.json
    manifest.json
  vendor/
    ort.wasm.min.mjs
    ort-wasm-simd-threaded.mjs
    ort-wasm-simd-threaded.wasm
    ONNX-RUNTIME-LICENSE.txt
    ONNX-RUNTIME-NOTICE.txt
```

Also distribute the root `LICENSE`, `NOTICE` and applicable dependency/model notices.
`manifest.json` is for release verification, not runtime loading. You do not need
the stock HTML, CSS, camera UI, PWA icons or service worker for worker-only use.
If using the whole existing app, deploy all of `prototype/web/` after vendoring,
omitting development harness/sample files from a production deployment.

Use the [example browser client](../examples/browser-client.mjs):

```js
import {createHearthReader} from './browser-client.mjs';

const reader = await createHearthReader('/hearth/inference-worker.mjs');
const output = await reader.read(selectedFile);
const {result, elapsedMs, imageSize} = output;

if (result.status === 'retake') {
  // Show crop/retake guidance; do not save a guessed measurement.
} else {
  // Show result.reading.sys, .dia and .pulse in editable fields.
  // Retain the candidate/review distinction and require user confirmation.
}

// Keep the reader for later photos. On component/page teardown:
reader.dispose();
```

`selectedFile` is obtained through your own file input. The example also accepts
a canvas or ImageBitmap and creates a transfer copy, so the host can keep its
own original. It orients images, limits the longest edge to 1920 and returns
the processed `imageSize` for overlays. Raw worker usage and result fields are
documented in [the API contract](API_REFERENCE.md).

In a component framework, initialize after mounting on the browser client, never
during server rendering. Disable capture until creation resolves. Keep the
instance in component state/ref, serialize reads, and dispose on unmount. Handle
unmount while initialization is pending by disposing the instance as soon as it
resolves. Use a current-request token to prevent obsolete results from replacing
a newer form. A timeout closes the example client; construct a new one to retry.

The example is a static module. It needs no bundler-specific loader. If your
bundler rewrites the worker into a generated chunk directory, its relative
model paths may break. Serving the worker and its dependencies unchanged from
the static directory avoids that issue. With an app deployed below a URL prefix,
adjust `/hearth/` to include that prefix. Do not assume it is at the origin root.

## Camera, hosting and offline use

Request camera access from a user action, use the rear camera when available,
copy a frame to a canvas, and pass the canvas to the reader. Stop every media
track after capture or teardown and when hiding the camera. This matches
`prototype/web/app.mjs`. Feed unannotated pixels to OCR; do not re-read the canvas
after painting boxes or text on it.

Serve over HTTPS in deployment; localhost works for development. Opening HTML
with `file://` is not supported. The host needs module workers, WebAssembly,
OffscreenCanvas and createImageBitmap. MIME types must be correct: JavaScript
for `.mjs`, `application/wasm` for `.wasm`, and JSON for `.json`. ONNX can use
`application/octet-stream`. Ensure SPA routing returns actual files rather
than `index.html` for missing models. Keep the assets same-origin.

The worker sets WASM `numThreads=1`, `proxy=false` and execution provider `wasm`.
It does not select WebGPU. The tested configuration works without custom
COOP/COEP headers on the updated test browser. If your host has a Content Security
Policy, allow the same-origin worker/assets and WebAssembly compilation according
to your policy; inspect violations rather than disabling the policy globally.
Browser behavior and runtime requirements should be checked against the
[ONNX Runtime Web deployment docs](https://onnxruntime.ai/docs/tutorials/web/deploy.html)
when changing runtimes.

For offline use in an existing PWA, add all worker imports, both models, config
and vendor runtime assets to your own cache strategy. Publish them as one matched
version and invalidate old assets together. Do not copy Hearth's `sw.js` into an
unrelated app's root without adapting scope, asset list and cache names. Its
activation deletes older caches whose names begin `hearth-bp-`, and its
cache-first behavior can otherwise keep stale models/config.

The stock app's “cached” message indicates successful service-worker readiness;
a deployment should still test a real offline reload and inference. Normal
images should not enter asset caches or analytics. Host apps control their own
storage and upload behavior; using Hearth does not automatically prevent a
host analytics SDK from collecting sensitive form fields.

## Python integration

Install the pinned Python environment from the root README. The existing
`Detector` import also imports PyTorch through `digit_model.py`, even for ONNX
inference. Do not assume an ONNX Runtime-only environment is sufficient for the
current Python code. Keep the `prototype` package importable from your project.

```python
import json
from pathlib import Path
import cv2
from prototype.detector import Detector

models = Path('prototype/web/models')
config = json.loads((models / 'config.json').read_text())
detector = Detector(models / 'bp-detector.onnx',
                    digit_model=models / 'bp-digits.onnx')
detector.recognizer.min_score = config['digitMinScore']

image = cv2.imread('path/to/monitor.jpg')  # BGR, not RGB
if image is None:
    raise ValueError('Image cannot be decoded')
result = detector.read(image, config['minScore'], config['acceptScore'])
```

Load sessions once, not once per request. The runnable equivalent is
`python examples/read_image.py path/to/monitor.jpg`. This is a single-pass result.
`prototype.adaptive.adaptive_read` can add row/lighting fallback after refusal,
but does not include the fixed portrait crops. Use the browser worker if you
need the existing complete pipeline unchanged. A Python server that receives
images needs its own authentication, request/image size limits, retention and
concurrency design; no such service is included.

## Native Android port

The phone tests used Android Chrome/PWA, not a native Kotlin application.
No AAR, JNI bridge, CameraX analyzer or Android Studio project is included.
The models use ordinary ONNX tensors; a native port should use the official
[ONNX Runtime Android guidance](https://onnxruntime.ai/docs/tutorials/mobile/)
and implement the [full preprocessing and decoding contract](API_REFERENCE.md).

Start with a still image and CPU execution. Match orientation, RGB/BGR conversion,
letterboxing, digit normalization, class order, duplicate suppression, assembly
and both fallback stages. Map coordinates back to the oriented source image.
Move inference off the UI thread, reuse sessions, close tensors and bound the
number/size of in-flight images. CameraX YUV conversion and rotation are host
responsibilities. Validate any NNAPI/GPU or quantized variant separately; current
browser evidence cannot establish the native port's accuracy or speed.

An Android WebView wrapper can reuse web code only if the chosen WebView and
asset origin support the required APIs and camera permission flow. It has not
been validated here; do not describe it as a tested native integration.

## Integration acceptance checks

Before using an integration, verify that a clean checkout loads both exact model
hashes, starts one worker, reads a consented image, handles refusal and review,
and survives a failed asset load without leaving controls stuck. Confirm overlay
coordinates after resize/crop, camera cleanup, repeated reads and offline behavior
if claimed. Do not count repeated replays of one photo as independent accuracy
samples. Evaluate new monitors, angles, lighting and devices with held-out human
transcriptions; report precision together with coverage and refusals.

See [troubleshooting and evaluation procedures](DEVELOPMENT.md).
