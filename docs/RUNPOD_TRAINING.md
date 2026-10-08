# Runpod training workflow and 8 October 2026 results

This document records the GPU-training work completed before the external-v2
test expansion. It contains aggregate results only. Credentials, Pod and volume
identifiers, private photos, per-image readings and detailed infrastructure
metadata are intentionally omitted.

## Connection and resource handling

The Runpod MCP connection was verified by reading the account's Pod inventory.
The existing project training Pod and persistent workspace were identified
without changing unrelated resources. The Roboflow credential was attached
through an existing Runpod secret reference; the secret value was never printed,
written to the repository or passed as a command-line argument.

Temporary CPU Pods were later used only to transfer the public external test
split and compute image hashes for the leakage audit. They were stopped promptly
and terminated after their local outputs were verified. The project training
Pod and shared training volume remained intact. At the end of the documented
work, the training Pod was stopped (`EXITED`), so no GPU compute job was left
running.

## GPU support added to the trainers

`prototype/train_detector.py` was extended with:

- `--device` selection (`auto`, `cpu` or a CUDA index);
- configurable `--batch`;
- optional `--amp` mixed precision;
- a fail-fast check when CUDA is requested but unavailable;
- a startup record of the selected device, GPU, batch size and AMP state.

The detector's existing deterministic seed, augmentation policy, optimizer,
freeze depth, early stopping and artifact locations were preserved.

`prototype/train_digits.py` was extended with:

- the same automatic/explicit device selection and CUDA fail-fast behavior;
- device placement for the model, training batches and validation tensors;
- checkpoint loading with an explicit `map_location`;
- CPU conversion before ONNX export;
- a startup record of the selected compute device.

These changes allow the same scripts to run on CPU locally and CUDA on Runpod
without maintaining separate training implementations.

## Reproducible Runpod helper

`runpod/train_public.sh` and `runpod/README.md` were added as a local workflow.
The helper:

1. creates or reuses a virtual environment;
2. installs the pinned CUDA PyTorch build and project requirements;
3. verifies CUDA and reports the runtime and GPU name;
4. obtains the public Roboflow training dataset only when absent;
5. validates annotations and prepares explicit train/validation manifests;
6. trains the detector and digit classifier on CUDA;
7. copies the best detector checkpoint to the expected model location;
8. exports both models to ONNX and runs export-parity checks;
9. evaluates and calibrates on validation data;
10. runs synthetic Python checks;
11. packages ignored checkpoints, reports and exported artifacts for retrieval.

The helper reads `RUNPOD_SECRET_ROBOFLOW_API_KEY` or a shell-local
`ROBOFLOW_API_KEY`. It does not embed a credential value. Generated datasets,
checkpoints and reports remain ignored.

## Training and validation results

| Evaluation | Result |
| --- | ---: |
| Detector best-checkpoint validation mAP50 | 0.9161 |
| Detector best-checkpoint validation mAP50–95 | 0.6889 |
| Digit-classifier validation | 179/179 correct |
| PyTorch/ONNX export parity | Passed |
| Synthetic tests from that run | 14/14 passed |
| Baseline end-to-end validation exact triplets | 52% |
| Calibrated validation exact triplets | 72% |
| Calibrated accepted precision | 100% |
| Calibrated accepted coverage | 48% |

The detector mAP values measure bounding-box detection across overlap
thresholds, not correct complete blood-pressure readings. The digit result uses
annotation-supplied crops, so it does not measure row localization or complete
end-to-end OCR. Export parity verifies implementation agreement, not accuracy.

The calibrated end-to-end figures were selected and measured on validation
data. They are therefore development results, not independent held-out evidence.
The apparent 100% accepted precision has a small denominator and must always be
reported with its 48% coverage. It did not establish greater-than-90% precision
on new captures.

## Relationship to external testing

After training, a web pilot produced:

| Pilot metric | Result |
| --- | ---: |
| Images with readable three-row ground truth | 8 |
| Single-pass exact triplets | 3/8 (37.5%) |
| Staged exact triplets | 6/8 (75.0%) |
| Staged accepted precision | 100% |
| Staged accepted coverage | 6/8 (75.0%) |

The pilot was too small and correlated to support a reliability claim. It led
to the larger [external test v2 construction and evaluation](EXTERNAL_TEST_V2.md),
which measured only 27/100 exact triplets and 22/24 correct candidates. That
larger shift is the stronger indicator of present generalization limits.

The external-v2 images are permanently test-only. `prototype/prepare_data.py`
now rejects their known paths if they are mistakenly added to detector training
or validation manifests, and `prototype/test_prepare_data.py` checks that guard.

## Commands

The public-data GPU workflow is:

```sh
bash runpod/train_public.sh
```

The external regression, when its ignored local manifests are available, is:

```sh
.venv/bin/python prototype/evaluate_external_v2.py
```

Neither command publishes artifacts. Review all generated outputs and the exact
Git diff before any future commit or push, and never publish secrets, private
images, personal readings or infrastructure identifiers.
