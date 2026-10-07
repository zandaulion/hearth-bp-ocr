# API and model contract

The source is the authority for this version. See the matching
[configuration](../prototype/web/models/config.json) and [model manifest](../prototype/web/models/manifest.json).
The integration surface is a browser module worker, not an HTTP OCR endpoint.

## Worker protocol

Create a same-origin module worker at `inference-worker.mjs`. It starts loading
both models immediately; there is no `init` request. Keep one worker alive and
submit one image at a time. The worker does not implement cancellation, a queue,
runtime configuration updates or progress events.

| Direction | Payload | Meaning |
| --- | --- | --- |
| Worker → client | `{type: 'ready', size: 512}` | Both model sessions initialized. |
| Client → worker | `{type: 'read', id, bitmap}` | Transfer the `ImageBitmap` in the `postMessage` transfer list. Use a unique request ID. |
| Worker → client | `{type: 'result', id, result, elapsedMs}` | Result for that request; elapsed time is rounded milliseconds, excluding initialization and client preprocessing. |
| Worker → client | `{type: 'error', id?, message}` | Initialization errors have no ID; request errors carry their request ID. |

Messages of other types are ignored. A concurrent read is rejected with an error
and its bitmap is closed; it is not queued. The worker also closes the active
bitmap in `finally`. After transfer the caller must not use that bitmap. On an
initialization failure, recreate the worker after resolving the failed asset or
runtime problem. A client timeout does not stop computation: terminate the
worker if you need to abandon it, reject pending requests, then create a new one.

The [browser example](../examples/browser-client.mjs) implements this lifecycle,
including timeouts, disposal, orientation and a 1920-pixel longest-edge limit.

## Result schema

```ts
type Reading = { sys: number; dia: number; pulse: number };
type Box = [number, number, number, number]; // x1, y1, x2, y2
type Rect = [number, number, number, number]; // x, y, width, height
type Detection = {
  class: string;       // "row" or a digit "0"..."9"
  score: number;       // uncalibrated score, not probability of correctness
  box: Box;
  detector_class?: string;
  recognition_score?: number;
};
type Row = {
  value: number;
  box: Box;
  digits: Detection[]; // left to right
  score: number;
  cx: number;
  cy: number;
  height: number;
};
type Result = {
  status: "candidate" | "review" | "retake";
  reading: Reading | null;
  score: number | null;
  reasons: string[];
  rows: Row[];        // SYS, DIA, pulse; empty on retake
  detections: Detection[];
  cropFallback?: {
    method: "two-agreeing-center-crops" | "two-agreeing-row-crops-light-normalized";
    views: {name: string; rect: Rect}[];
  };
};
```

Values are integers; units are mmHg for SYS/DIA and beats/minute for pulse.
Coordinates are floating-point pixels in the bitmap sent to the worker, with
origin at its top left. Fallback boxes are translated back to that bitmap.
If the host app cropped/downscaled/rotated before transfer, it must maintain that
outer transform itself to draw boxes on the original full-resolution photo.

`candidate` means one complete stack passed the current score and consistency
rules; still request human verification. `review` has a complete reading but
one or more concerns. `retake` has no reading and null score. Do not coerce null
to zero or fill missing digits from prior measurements. Reasons are displayable
English text, not stable error codes. Keep machine decisions based on status.

The result score is the weakest row/digit score; a refined digit score is the
minimum of its detector score and recognizer softmax score. Agreement fallback
takes the weaker result score from its two views. It does not turn scores into
calibrated probabilities. Store user-confirmed corrections separately from raw OCR
if your application needs an audit trail; Hearth itself has no persistence API.

## Model tensors and preprocessing

| Export | Input | Output |
| --- | --- | --- |
| `bp-detector.onnx` | `images`, float32 `[1,3,512,512]` | `output0`, float32 `[1,15,5376]` |
| `bp-digits.onnx` | `crops`, float32 `[N,1,48,32]` | `logits`, float32 `[N,11]` |

Both current exports use ONNX opset 17. The detector batch and spatial dimensions
are fixed. The recognizer batch is dynamic; skip it when no digit boxes exist.
Query tensor names/shapes when porting, and fail on an incompatible replacement.

Detector preprocessing:

1. Compute `scale = min(512 / width, 512 / height)`.
2. Round resized width/height; center with `floor((512 - resizedSize) / 2)`.
3. Fill letterbox padding with RGB `(114,114,114)` and draw the resized photo.
4. Convert RGB to planar channel-first float32, dividing each byte by 255.
5. Decode channel-first predictions. Channels 0–3 are `cx,cy,w,h`; remaining
   channels are class scores. There is no separate objectness channel and no
   additional sigmoid/softmax in this decoder. Undo letterboxing and clip boxes.

The detector class order is **`["0","1","row","2","3","4","5","6","7","8","9"]`**.
Class index 2 means a numeric row, not digit 2 or the whole LCD. The decoder uses
the source constant, not `config.classNames`. Duplicate suppression compares
digits across classes, separately from rows: IoU > 0.45, or for digits only,
intersection / smaller-area > 0.75. This is not ordinary per-class YOLO NMS.

Recognizer preprocessing in the browser:

1. Crop each non-row box from the current source view using floor/ceil bounds.
2. Resize to width 32, height 48 using canvas.
3. Convert to rounded grayscale `0.299 R + 0.587 G + 0.114 B`.
4. Subtract the crop mean and divide by `max(populationStdDev, 12)`.
5. Pack `[N,1,48,32]`, run inference, and apply stable softmax to each 11-logit row.

Recognizer indices 0–9 are digits, **index 10 is background/noise**. This order
differs from the detector! Drop background and predictions below `digitMinScore`.
Refined detections retain the original box and original class in `detector_class`.

## Assembly and fallbacks

Assembly considers the eight highest-score row boxes. Digit centers get 8%
vertical slack; horizontally, a digit box may touch the row or extend up to 20%
of row height beyond an edge. Basing this allowance on row height recovers a
narrow edge digit from a prematurely ended row proposal without sweeping a
distant digit into a wide row. A row must have exactly two or three digits and
no leading zero. Assembly searches all three-row combinations for one
unambiguous vertical stack and rejects overlapping rows, reused digits,
side-by-side columns and strongly mismatched SYS/DIA heights.

Consistency checks require SYS 50–280, DIA 25–180, pulse 20–250 and SYS > DIA.
These are transcription filters, not medical interpretations. Failing them
changes a complete full-pass result to `review` and permits fallback attempts;
it does not invent or alter a digit. The original result remains unless both
fallback views agree on the same plausible replacement.

The fixed portrait views apply at height ≥ 1.2 × width. Their normalized
`[x,y,width,height]` are `[.15,.25,.7,.6]` and `[.25,.32,.6,.5]`, rounded to pixels.
Both must be candidates with exactly the same triplet.

Adaptive proposals use two or three original row boxes with score ≥ 0.25 and
compatible geometry, padded to leave space for a pulse row. Two normalized
images divide grayscale by a background estimated with three replicated-border
box blurs, using radius fractions 0.04 and 0.05. Both readings must be plausible
and agree. If either view is `review`, the combined result stays `review`.
See [the implementation notes](ADAPTIVE_CROP.md) for evidence and limitations.

## Configuration: executable versus descriptive

| Property | Current behavior |
| --- | --- |
| `size` = 512 | Worker tensor/letterbox size; must match detector export. |
| `minScore` = 0.20 | Detection filtering and assembly threshold. |
| `acceptScore` = 0.25 | Weakest row/digit threshold for candidate status. |
| `digitMinScore` = 0.50 | Minimum recognizer softmax score. |
| `version`, `scope`, `scoreNotice` | Descriptive metadata. |
| `classNames` | Descriptive; class constants in code perform decoding. |
| `detectorSha256`, `digitsSha256` | Release metadata; the worker does not verify hashes at runtime. |
| `portraitCropFallback`, `adaptiveCropFallback` | Descriptive mirrors of code; changing `enabled`, crop geometry or fractions here alone has no effect. |

The worker has no `configUrl`, `modelUrl` or constructor-options message.
Asset names are fixed relative to the worker location. Python assembly/read
defaults are 0.25 / 0.75, which differ from released thresholds: pass config
values explicitly. `DigitRecognizer.min_score` defaults to 0.5; assign it if
using a different config.

## Python functions

| Symbol | Contract |
| --- | --- |
| `Detector(model, threads=4, digit_model=None)` | Load CPU detector and optional recognizer once; size read from export. |
| `Detector.read(image, min_score=.25, accept_score=.75, view_scale=1)` | BGR uint8 H×W×3 input; returns one-pass result dictionary. |
| `DigitRecognizer(model, threads=2, min_score=.5)` | CPU digit session; `refine(image, detections)` returns refined detections. |
| `letterbox(image, size=512, view_scale=1)` | Returns tensor and inverse-transform metadata. `view_scale` is a development framing knob, not a general crop API. |
| `decode_output(output, transform, min_score=.25)` | Detector tensor → source-coordinate detections with NMS. |
| `digit_tensor(crop)` | BGR or grayscale crop → normalized `[1,48,32]` tensor. |
| `assemble(detections, min_score=.25, accept_score=.75)` | Detections → result; does not perform neural inference. |
| `row_region(detections, width, height)` | Adaptive `[x,y,w,h]` proposal or None. |
| `normalize_light(image, fraction)` | BGR image → normalized BGR image. |
| `adaptive_read(detector, image, full, min_score=.2, accept_score=.25)` | Returns `(selectedResult, views)`; only adaptive fallback, no fixed central crops. |

Python adaptive results use `adaptive: {rect, fractions}` rather than browser
`cropFallback`. Do not claim wire-format parity for that metadata. The existing
classes are mutable research objects without a documented concurrent-call
guarantee; serialize calls per instance or create isolated instances.
