# Runpod public-data training

This local-only helper trains the public-data detector and digit recognizer on
one CUDA GPU. It does not use or upload private photos.

## Pod requirements

- Runpod PyTorch Pod with one NVIDIA GPU (16 GB VRAM is ample for this recipe)
- Python 3.10 or newer
- At least 20 GB container disk and 10 GB volume disk mounted at `/workspace`
- SSH enabled; a public IP is convenient for copying the final archive with SCP
- A Runpod secret named `RUNPOD_SECRET_ROBOFLOW_API_KEY`, or a shell-local
  `ROBOFLOW_API_KEY`, for downloading the public Roboflow dataset

Clone the public repository into `/workspace`, apply the local GPU patch, and run:

```sh
cd /workspace/hearth-bp-ocr
bash runpod/train_public.sh
```

The script verifies CUDA before training. It uses the official PyTorch 2.14.1
CUDA 12.6 wheels, trains both models, exports ONNX, checks export parity,
evaluates the public validation split, calibrates thresholds without activating
them, runs synthetic tests, and packages ignored outputs at:

```text
prototype/reports/runpod-public-output.tar.gz
```

Retrieve the archive before terminating a Pod that has no persistent network
volume. Do not commit generated checkpoints, datasets, reports, API keys, or
private images.

See the aggregate [Runpod training record](../docs/RUNPOD_TRAINING.md) for the
implemented GPU changes, results and interpretation. External-v2 regression
images are test-only and are not inputs to this training helper.
