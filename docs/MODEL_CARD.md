# Hearth v2 models with adaptive3 postprocessing

Released 7 October 2026. Both ONNX exports are included for local browser
inference under **AGPL-3.0-only**; see [LICENSE](../LICENSE). No training is
required to use them. Their bytes match the models used in the reported v2
evaluations and Fold4 tests. The adaptive2 preprocessing and adaptive3 row
association/fallback changes do not alter learned weights.

## Files and interfaces

| File in `prototype/web/models/` | Bytes | Purpose |
| --- | ---: | --- |
| `bp-detector.onnx` | 10,512,570 | YOLO11n detector: float RGB input, 1 x 3 x 512 x 512; detects numeric rows and digits. |
| `bp-digits.onnx` | 424,258 | DigitNet crop recognizer: float grayscale input, N x 1 x 48 x 32; 11 logits for digits 0–9 and background. |

Exact SHA-256 checksums are in [manifest.json](../prototype/web/models/manifest.json)
and the matching inference configuration. `tools/check_public_tree.py` pins the
approved binary hashes. Browser preprocessing is implemented in
`inference-worker.mjs`; use it with `reading.mjs`, `crop-fallback.mjs`, and
`adaptive-crop.mjs`. Input sizes alone do not describe the required normalization.

## Training provenance

The detector starts from official Ultralytics YOLO11n pretrained weights and
uses Ultralytics 8.4.173. Its initial local training used 239 structurally usable
public training images. A second phase fine-tuned that checkpoint with eight
additional owner-provided monitor photos, repeated six times in the training
list alongside the public images. Fine-tuning used 12 epochs, image size 512,
batch size 8, AdamW at initial learning rate 0.001, the first ten layers frozen,
CPU execution and seed 20261006. See `prototype/train_detector.py` for the
augmentation and optimizer settings, and `prototype/export_model.py` for export.

The public training dataset is **Blood Pressure Monitor Display, Final Project,
version 1**, hosted on [Roboflow Universe](https://universe.roboflow.com/final-project-cwtfb/blood-pressure-monitor-display/dataset/1)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Training selected
usable annotations, renamed row class "10" to "row", and applied resizing and
augmentation. DigitNet used digit crops from these annotations, the eight
owner-provided photos, and generated background crops. Its architecture,
preprocessing and training/export implementation are in `prototype/digit_model.py`
and `prototype/train_digits.py`. Kaggle images were not used for these weights.

The owner confirmed ownership of the eight additional photos and authorized
public release of the trained weights. The photos, per-photo labels and private
paths remain excluded. Their annotations used manually reviewed values and
geometry proposals from the excluded starter implementation. The shipped
inference and training source does not import that implementation.

The two recent Fold4 development photos were used to tune crop and illumination
handling, not to update learned weights. They also remain excluded. The public
training recipe can produce replacement models, but cannot reproduce the exact
bundled weights without the private training examples. This is a usable model
release with a documented reproducibility limitation.

Ultralytics provides its trained models under AGPL-3.0 by default; see its
[licensing statement](https://www.ultralytics.com/license). Upstream notices and
detector metadata are retained. Dataset licenses remain separate from the
project/model license; see [data attribution](DATA_LICENSES.md) and
[dependency notices](../prototype/THIRD_PARTY.md).

## Validation and limitations

The intended scope is upright home-monitor displays with three SYS/DIA/pulse
rows. Check every value against the monitor. The reader does not interpret
measurements, and detector scores are not correctness probabilities.

See [aggregate results](RESULTS.md) and [adaptive results](ADAPTIVE_CROP.md).
Public validation was used for selection; the earlier reserved test was consumed
by v1 and later became regression data. A fresh Fold4 capture on adaptive2 was
correct in 1.16 seconds, confirmed by the user, after four refusals across earlier
browser/pipeline revisions. One success on a known monitor is not a representative
benchmark. Adaptive3 adds later regression checks described in the adaptive
notes; they were used for selection and are not a new benchmark.
**More than 90% prospective precision remains unproven.**

Before publication, both ONNX files passed structural validation. Protobuf string
fields were inspected for personal paths and identifiers; neither file references
external tensors. No images, annotation tables, optimizer state or pickled Python
checkpoints are bundled. This inspection is not a formal test of training-data
memorization or a privacy guarantee.
