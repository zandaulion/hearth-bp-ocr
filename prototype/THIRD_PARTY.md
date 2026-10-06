# Dependencies and source provenance

Original source, documentation, generated app icons, and bundled model weights are
licensed under AGPL-3.0-only. See the root [LICENSE](../LICENSE). This grant
does not replace licenses on third-party dependencies or their notice files.

The detector training and decoding workflow uses Ultralytics YOLO11n and
official starting weights from [Ultralytics assets](https://github.com/ultralytics/assets).
Ultralytics is distributed under AGPL-3.0; the installed package's license is
retained at `licenses/ULTRALYTICS-LICENSE.txt`. Redistribution and integration must
respect the applicable Ultralytics terms. The fine-tuned detector is bundled as
an ONNX export under AGPL-3.0-only, with its upstream metadata retained.

Browser inference uses Microsoft ONNX Runtime Web 1.30.0, pinned in npm metadata.
`npm run vendor` copies the runtime for local offline use and includes its
[MIT license](https://github.com/microsoft/onnxruntime/blob/main/LICENSE) and notice.
Generated runtime files are ignored by Git. Other dependencies retain their own
licenses; see their package distributions.

Public training labels come from Final Project's Blood Pressure Monitor Display,
version 1, on Roboflow Universe, under CC BY 4.0. Class name "10" denotes numeric
row regions; model metadata renames it "row" without changing source files.
See [dataset notices](../docs/DATA_LICENSES.md) for source and license links.

The user-supplied seven-segment starter implementation is excluded from this
snapshot. Remaining code does not import it. Its original files stay in the
separate local research workspace.

The bundled DigitNet crop recognizer uses the architecture and training source
in this repository and is released under AGPL-3.0-only. Eight owner-provided
training photos were cleared for public release of the trained models, while
the photos and annotations remain private. See the [model card](../docs/MODEL_CARD.md)
and [Ultralytics licensing](https://www.ultralytics.com/license).
