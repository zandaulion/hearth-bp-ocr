# Evaluation tools

`dataset_tools.py` validates Roboflow's exported class-name order and reconstructs
triplet references from annotations. Class name `10` identifies numeric rows,
not the full LCD. References derived from these labels are not independent human
transcriptions. Download the correct dataset version before parsing real labels.

`benchmark.py` provides shared functions for complete-reading accuracy, candidate precision, coverage, individual
fields, latency, and Wilson intervals. Refusals count as misses in overall
accuracy; references missing pulse do not enter the full-triplet denominator.
`test_metrics.py` exercises these rules with synthetic inputs and needs no photos.

Other scripts provide annotation audits, digit experiments, and contact sheets.
Scripts involving local curated/personal references require untracked ground
truth files supplied separately. Generated manifests, images, model artifacts,
and reports are excluded from Git. Keep new prospective test sessions separate
from training, model selection, and threshold tuning.

The local `dataset/external_test_v2/` manifests are a frozen regression suite:
100 readable displays plus 25 images that should yield no reading. Run them with
`python prototype/evaluate_external_v2.py`. `prototype/prepare_data.py` rejects
known external-test paths if they are accidentally introduced into a training
or validation manifest.

See [aggregate development results](../docs/RESULTS.md) and
[dataset licensing](../docs/DATA_LICENSES.md).

## Prospective external test v3

`tools/build_external_test_v3.py` prepares a new model-blind benchmark from two
pinned Roboflow Universe exports. Keep the API key in `ROBOFLOW_API_KEY` or in
the ignored repository `.env` file; never pass it on the command line.

```bash
python tools/build_external_test_v3.py download
python tools/build_external_test_v3.py prepare
```

`prepare` reconstructs candidate readings only from the source digit boxes,
creates annotated contact sheets, and audits exact and perceptual overlap with
the existing local corpus. It does not run Hearth. A person must complete the
ignored `dataset/external_test_v3_review/review.json`, checking the visible
reading, scope, privacy, and every reported near duplicate. Only then run:

```bash
python tools/build_external_test_v3.py freeze
```

The frozen manifest is permanently test-only. Run inference only after its
checksum has been recorded; do not use its outcomes to tune preprocessing,
thresholds, weights, or selection rules.
