# Hearth BP monitor OCR

An experimental reader for systolic pressure (SYS), diastolic pressure (DIA),
and pulse on common upright, three-row blood-pressure monitor displays.
The PWA performs ONNX inference locally in a browser worker and supports camera
capture, image selection, cropping, manual corrections, and copying a verified
reading. It does not interpret measurements medically.

After a refusal or an assembled reading that fails consistency checks, the
current pipeline can try agreeing central crops followed by a crop inferred
from numeric rows with uneven-lighting correction. See the
[adaptive fallback design and development results](docs/ADAPTIVE_CROP.md).

Original source code, documentation, generated app icons, and bundled model weights are licensed under
**AGPL-3.0-only**; see [LICENSE](LICENSE). Third-party dependencies retain their
own licenses and notices in [THIRD_PARTY.md](prototype/THIRD_PARTY.md).

**More than 90% precision on new real-world captures has not been established.**
Check every value against the monitor. Detection scores are not calibrated
probabilities. See the [plain-language report](docs/PLAIN_LANGUAGE_REPORT.md) or
[aggregate development results](docs/RESULTS.md).

The tested detector and digit-recognizer ONNX weights are included (10.9 MB total).
See the [model card](docs/MODEL_CARD.md) for provenance, checksums and limitations.
The user-supplied starter code, private photos, individual readings, screenshots,
logs, machine connection settings and downloaded datasets remain excluded.

## Browser setup

For reuse in another project, start with the [integration guide](docs/INTEGRATION.md)
and [runnable examples](examples/README.md). Give another coding agent the
[AI integration handoff](docs/AI_INTEGRATION.md). The [documentation index](docs/README.md)
links the architecture, API/model contract, training and troubleshooting guides.

Use Python 3.12 and a current Node.js version. Install the browser runtime:

```sh
cd prototype
npm ci
npm run vendor
npm test
cd ..
```

The checkout includes both models and their matching `config.json`; no training
or dataset download is needed to run OCR. Only these two reviewed ONNX exports
are allowed in Git. Other checkpoints and user photos remain ignored.

```sh
python prototype/serve.py
```

Open <http://127.0.0.1:8765/> after installing and vendoring the browser runtime.
Camera capture needs a secure context. For Android, use HTTPS or the localhost
tunnel described by the [phone skill](skills/lenovo-android/SKILL.md).
After a successful complete load, the service worker caches the static app and
models. Normal user photos stay in memory and are not uploaded or persisted.
The separate development harness records metrics when explicitly run.

## Reproduce public-data training

Obtain the [Roboflow version 1 dataset](https://universe.roboflow.com/final-project-cwtfb/blood-pressure-monitor-display/dataset/1)
under its CC BY 4.0 terms. Download YOLOv8 format to
`dataset/roboflow_bp_display/`, preserving its license and class metadata.
`download_roboflow_dataset.py` accepts `ROBOFLOW_API_KEY` through the environment;
never commit API keys or downloaded authentication files.

Create a virtual environment. In PowerShell use `.venv/Scripts/python.exe` after
creation; on Linux/macOS activate `.venv/bin/activate`.

```sh
python -m venv .venv
python -m pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r prototype/requirements.txt
python evaluation/dataset_tools.py
python prototype/prepare_data.py
python prototype/bootstrap.py
python prototype/train_detector.py --epochs 40 --name detector_v1
python prototype/train_digits.py
python prototype/export_model.py prototype/runs/detector_v1/weights/best.pt
python prototype/calibrate.py --model prototype/web/models/bp-detector.onnx --output prototype/reports/calibration.json --version public-v1
```

Use the virtual environment's Python for all commands. The training workflow
overwrites local exported models and generates a calibration report. Follow the
[release procedure and freeze-script caveat](docs/DEVELOPMENT.md) before activating
new thresholds or publishing another model version. The recipe
above uses public images only and will not recreate the bundled fine-tuned
models' exact weights or metrics because their private training photos are omitted.
Keep genuinely new test images untouched until
models and thresholds are frozen. Optional scripts that consume local personal
ground truth need those untracked inputs; they are not required by this workflow.

## Tests and Android helpers

```sh
python prototype/test_reading.py
python prototype/test_prepare_data.py
python evaluation/test_metrics.py
python tools/check_public_tree.py
```

The Python checks need the listed Python dependencies; JavaScript reading and
crop-fallback tests need Node.js only. Tests use synthetic readings.
The [Lenovo Android skill](skills/lenovo-android/SKILL.md) provides fresh ADB
discovery, named phone selection, screenshots, UI snapshots, and SSH tunnels.
Configure your own trusted SSH connection; no host address or credentials are
included. Its helper needs Python 3 and OpenSSH locally, and ADB on the SSH host.

For the optional Android timing harness, provide consented local images and
`prototype/web/samples/test-manifest.json`, an array of objects with `file`,
`expected` (`sys`, `dia`, `pulse`), and `role`. Those inputs are ignored by Git.
Reports are generated locally and remain ignored too.

The ignored external-v2 regression suite, when present locally, runs with
`python prototype/evaluate_external_v2.py`. See its complete
[construction, leakage audit and results](docs/EXTERNAL_TEST_V2.md). These cases
are permanently excluded from training and calibration.

See [data licensing](docs/DATA_LICENSES.md), [publication policy](docs/PUBLICATION.md),
and [third-party notices](prototype/THIRD_PARTY.md).
