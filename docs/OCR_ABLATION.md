# OCR preprocessing and retry ablation

## Status and scope

This experiment measured which parts of the local OCR pipeline change end-to-end
performance. It used the already-consumed `external-v2` regression suite: 100
readable BP-monitor photos and 25 images that should be refused. The suite is
test-only. These results are useful for regression analysis, but are not fresh
held-out evidence and must not be used to tune thresholds or train models.

All reference values and image checksums were frozen before this experiment.
Every proposed crop was selected from image geometry and model detections, never
from the expected SYS, DIA or pulse values.

## Hearth ablation

Each row adds or substitutes one specific behavior. An exact result requires all
three values to match. Candidate precision is the fraction of best-quality
answers whose complete triplet is exact. Latency is measured by the Python ONNX
reference on the local ARM test machine and includes all passes in that row.

| Pipeline | Exact triplets | Candidates | Candidate precision | Safe refusals | Median | p95 | Mean passes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| One full-frame pass | 20/100 | 19 | 89.5% | 25/25 | 86 ms | 96 ms | 1.00 |
| Current staged retries | 27/100 | 24 | 91.7% | 25/25 | 253 ms | 268 ms | 2.26 |
| Current + expanded normalization | 28/100 | 24 | 91.7% | 25/25 | 763 ms | 806 ms | 5.19 |
| Current + display rectification | 28/100 | 25 | 92.0% | 25/25 | 260 ms | 458 ms | 2.94 |
| Current + normalization + rectification | 29/100 | 25 | 92.0% | 25/25 | 770 ms | 1,041 ms | 5.86 |
| Ensemble digit classifier throughout | 26/100 | 20 | 100% | 25/25 | 262 ms | 297 ms | 2.30 |
| Current + ensemble only as rescue | 28/100 | 24 | 91.7% | 25/25 | 518 ms | 551 ms | 4.14 |

### Marginal impact

The largest measured improvement is already in the staged retry pipeline. It
adds seven exact triplets over a single full-frame pass, while keeping all 25
negative images safe.

Relative to that current pipeline:

- display rectification adds one exact triplet and one correct candidate for
  about 7 ms additional median latency;
- expanded normalization adds one exact triplet but adds about 510 ms median
  latency;
- combining normalization and rectification adds two exact triplets but adds
  about 517 ms median latency;
- using the digit ensemble only after a refusal adds one exact triplet but adds
  about 266 ms median latency;
- replacing the normal digit classifier with the ensemble reduces total exact
  results by one and candidate coverage by four. Its 100% candidate precision
  comes from returning fewer candidates, not from improving total accuracy.

Display rectification therefore has the best measured accuracy/latency tradeoff.
It remains an experimental candidate because this consumed suite cannot provide
independent evidence for promoting the change.

## Google ML Kit ablation on Galaxy A52

A separate Android benchmark used bundled Google ML Kit Text Recognition v2
`16.0.1` on a Samsung Galaxy A52 (`SM-A525F`, Android API 34). The APK does not
use Firebase, does not call a server and does not alter the production app. Its
debug APK was 46,547,632 bytes (about 44.4 MiB).

“App-accepted” below follows the earlier BP Digitizer rule that SYS and DIA are
both present; accepted precision still requires the complete SYS/DIA/pulse
triplet to be exact.

The detector produced a row crop for 90/100 readable cases and 0/25 negative
cases. Geometric display rectification was available for 43/100 readable and
7/25 negative cases. Missing derived views count as no reading in the accuracy
denominator. ML Kit latency below is measured on the A52. Detector preprocessing
time was measured separately on the local ARM machine, so the two timings must
not be added and presented as a single-device end-to-end result.

| ML Kit input/pass | Available | Exact | Correct SYS/DIA/pulse | App-accepted | Accepted precision | Safe refusals | Median ML Kit time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original image | 100/100 | 0/100 | 7 / 4 / 0 | 15 | 0% | 13/25 | 57 ms |
| Full-image grayscale | 100/100 | 0/100 | 6 / 3 / 0 | 13 | 0% | 19/25 | 60 ms |
| Detector row crop | 90/100 | 0/100 | 3 / 0 / 0 | 5 | 0% | 25/25 | 63 ms |
| Row crop + grayscale | 90/100 | 0/100 | 3 / 0 / 0 | 5 | 0% | 25/25 | 68 ms |
| Row crop + Otsu | 90/100 | 0/100 | 2 / 0 / 0 | 1 | 0% | 25/25 | 64 ms |
| Row crop + Otsu + dilation | 90/100 | 0/100 | 0 / 0 / 0 | 0 | — | 25/25 | 58 ms |
| Rectified display | 43/100 | 0/100 | 1 / 0 / 0 | 4 | 0% | 24/25 | 64 ms |
| Rectified + grayscale | 43/100 | 0/100 | 0 / 0 / 0 | 4 | 0% | 24/25 | 70 ms |
| Strict agreement across passes | 100/100 | 0/100 | 2 / 1 / 0 | 3 | 0% | 22/25 | 426 ms |

The parser was checked separately. In the original-image pass, ML Kit's raw
text contained the correct SYS string in 7 images, DIA in 6 and pulse in 16,
but never contained all three correct values in the same image. The absence of
exact triplets is therefore a recognition failure on these LCD digits, not just
a row-grouping or parsing bug.

For this dataset, ML Kit preprocessing improves refusal behavior but does not
recover correct complete readings. It should not replace Hearth's specialized
detector and digit classifier.

## What remains unmeasured

The public suite contains one still image per case, so it cannot measure a true
three-frame camera burst. Reprocessing one still image through several visual
variants was measured above; temporal voting between genuinely different frames
was not. A valid burst experiment needs consented sequences of the same stable
monitor reading, captured before rules are chosen, plus a separate frozen test
set. Latency, battery use, agreement rate and exact-triplet accuracy should all
be reported.

## Reproduction and artifacts

The Hearth ablation runner verifies every frozen image checksum and writes its
detailed report only to the ignored reports directory:

```sh
.venv/bin/python prototype/evaluate_ablation.py
```

The public Android bundle, including detector-selected derived views, is created
outside the repository:

```sh
.venv/bin/python tools/export_android_benchmark.py \
  /tmp/hearth-mlkit-benchmark --derived-views
```

Detailed per-image Hearth and ML Kit reports remain ignored because they are
development artifacts. The tracked code contains the experiment definitions;
no benchmark images, individual readings, device connection details or secrets
are added to Git.
