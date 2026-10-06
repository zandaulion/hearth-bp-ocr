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

See [aggregate development results](../docs/RESULTS.md) and
[dataset licensing](../docs/DATA_LICENSES.md).
