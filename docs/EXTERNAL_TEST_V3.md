# External test v3: fresh public-source benchmark

This document records the external evaluation completed on 9 October 2026.
The benchmark was assembled and frozen before Hearth inference, then consumed
by one evaluation run. It now contains 39 readable, test-only cases and must
not be used for training, validation, threshold selection or augmentation.

No images, per-image readings, secrets, contact sheets or detailed inference
records are tracked by Git. They remain in ignored local directories.

## Sources and acquisition

Two pinned Roboflow Universe versions were used:

| Source | Exported images | Declared license |
| --- | ---: | --- |
| [Automated BP Device Digit Recognition v5](https://universe.roboflow.com/adindra-vickar-ega-odhei/automated-bp-device-digit-recognition/dataset/5) | 534 | CC BY 4.0 |
| [BP Monitor Reading / Medical Device Images v2](https://universe.roboflow.com/datacluster-labs-agryi/bp-monitor-reading-medical-device-images/dataset/2) | 121 | CC BY 4.0 |

The local `.env` file was confirmed ignored, untracked and absent from Git
history. The Roboflow credential was supplied through a write-only secret and
was not printed, copied into the repository or included in the generated
reports. A temporary isolated worker downloaded the public archives over HTTPS
and verified their integrity, then was permanently terminated.

## Model-blind construction

`tools/build_external_test_v3.py` implements three separate phases:

1. download the two pinned exports;
2. prepare source-annotation proposals, contact sheets and overlap reports
   without running Hearth;
3. freeze only fully reviewed cases, with image SHA-256 values, before
   inference.

The Ega export contained 494 generated training variants and only 40 original
validation/test images. All 494 training variants were excluded to avoid
counting augmented versions of the same source photographs as independent
tests. All 121 DataCluster images and the 40 Ega validation/test images entered
the review pool.

The source digit boxes produced complete three-row proposals for 65 of the 161
review candidates. The other 96 lacked a complete source-annotated triplet and
were not manually transcribed for this benchmark. Contact sheets were reviewed
without Hearth predictions. Visible readings, annotation order, intended scope
and privacy were checked before the answer manifest was frozen.

## Leakage and repetition audit

Every review image was compared with the existing local corpus using exact
SHA-256 and 256-bit difference hashes at Hamming distance 18 or less.

| Audit result | Count |
| --- | ---: |
| Review pool | 161 |
| Exact matches with the existing corpus | 0 |
| Perceptual matches requiring review | 22 |
| Within-pool perceptual pairs requiring review | 7 |

All 22 cross-corpus matches were Ega images related to the already consumed
external-v2 suite and were excluded. The within-pool report was reviewed along
with repeated readings and source sequences. Three repeated-session cases were
removed, as was one `888/888/888` all-segments display test that was not a blood
pressure measurement.

The final exclusions were:

| Reason | Cases |
| --- | ---: |
| No complete source-annotated triplet | 96 |
| Perceptual overlap with external v2 | 22 |
| Repeated source session/reading | 3 |
| All-segments device test | 1 |
| **Total excluded** | **122** |

The frozen set contains 39 cases: 11 from Ega and 28 from DataCluster. The
answer manifest was written before inference with this SHA-256:

```text
b6af166a44d6db1f3b5bbc8677ac5171d5e2d1ca683884a5e00d58a282f56f8b
```

## Evaluation

The existing `v2-adaptive3` ONNX release and its frozen thresholds were used.
The evaluator verified each image checksum before processing it.

| Component | SHA-256 |
| --- | --- |
| Detector | `8a7dafaed0aa9056308171d57d024343049cff4ba1a9adfc54d274c38a31fb07` |
| Digit classifier | `f41a41f9138c7c80d7e7fa1c3ed5afe93c3f5beacb3efd861ba3222b597cd732` |

The full-frame baseline makes one pass. The staged policy adds the existing
two-agreeing portrait crops and then the two-agreeing adaptive normalized crop
when earlier stages do not return a plausible reading. No setting was changed
after inspecting this benchmark.

## Results

| Metric | Full-frame baseline | Staged policy |
| --- | ---: | ---: |
| Readable images | 39 | 39 |
| Returned readings, including review results | 8 | 12 |
| Exact SYS/DIA/pulse triplets | 8 | 11 |
| Exact-triplet accuracy | 20.5% | 28.2% |
| Exact accuracy, 95% Wilson interval | 10.8–35.5% | 16.5–43.8% |
| Accepted candidates | 6 | 10 |
| Accepted candidate precision | 100% | 90.0% |
| Candidate precision, 95% Wilson interval | 61.0–100% | 59.6–98.2% |
| Candidate coverage | 15.4% | 25.6% |
| Median / p95 staged wall time | Not separately timed | 232 / 456 ms |

The staged policy added three exact readings and one incorrect accepted result.
That error preserved SYS and pulse but misread a DIA digit. This is important:
the acceptance policy reduced output frequency, but did not make accepted
readings reliably error-free.

Per-source staged results were:

| Source | Exact triplets | Candidate precision | Candidate coverage |
| --- | ---: | ---: | ---: |
| Ega | 5/11 (45.5%) | 5/5 (100%) | 45.5% |
| DataCluster | 6/28 (21.4%) | 4/5 (80.0%) | 17.9% |

There were no negative/refusal cases in v3, so this run does not add refusal
safety evidence. The small source slices and wide confidence intervals should
not be used to rank the datasets or device families.

## Comparison with Gemini 2.5 Flash

After the benchmark and Hearth results were frozen, the same 39 images were
sent once to the `gemini-2.5-flash` model selected by BP Digitizer's production
configuration. The request prompt and light image preprocessing matched the
earlier external-v2 comparison. All requests completed without retries. This
was a direct hosted-model comparison, not a complete Android app test.

| Metric | Hearth staged policy | Gemini 2.5 Flash |
| --- | ---: | ---: |
| Exact SYS/DIA/pulse triplets | 11/39 | 33/39 |
| Exact-triplet accuracy | 28.2% | 84.6% |
| Exact accuracy, 95% Wilson interval | 16.5–43.8% | 70.3–92.8% |
| Correct SYS fields | 12/39 | 38/39 |
| Correct DIA fields | 11/39 | 35/39 |
| Correct pulse fields | 12/39 | 35/39 |

The paired outcomes were:

| Outcome on the same image | Images |
| --- | ---: |
| Both systems exact | 11 |
| Hearth exact, Gemini not exact | 0 |
| Gemini exact, Hearth not exact | 22 |
| Neither system exact | 6 |

Hearth accepted 10/39 images and 9 of those triplets were exact: 90.0%
precision at 25.6% coverage. Gemini returned all three fields for all 39 images
and 33 triplets were exact: 84.6% exact output at 100% coverage. These categories
are not equivalent. Hearth uses an explicit confidence threshold; Gemini does
not expose a comparable calibrated confidence and was instructed to omit
uncertain fields. Under BP Digitizer's SYS-and-DIA acceptance rule, all 39
Gemini outputs would be accepted.

| Source | Hearth exact | Gemini exact |
| --- | ---: | ---: |
| Ega | 5/11 | 11/11 |
| DataCluster | 6/28 | 22/28 |

Observed median/p95 wall time was 232/456 ms for local Hearth inference and
2.53/5.39 seconds for Gemini including its network request. Gemini was about
10.9 times slower at the median and 11.8 times slower at p95, but this is not a
controlled hardware comparison. The 39 Gemini requests used 67,229 tokens in
total. V3 contains no negative cases, so it cannot compare refusal behavior.

## Post-hoc variant comparison

After the frozen Hearth and Gemini comparison above had consumed v3, every
pre-existing Hearth ablation variant was run on the same 39 readable images.
These results can be combined with v2 for regression analysis, but are not a
second prospective test.

| Hearth pipeline | Exact triplets | Exact candidates | Candidate coverage | Median / p95 rerun |
| --- | ---: | ---: | ---: | ---: |
| One full-frame pass | 8/39 | 6/6 | 15.4% | 155 / 212 ms |
| Current staged retries | 11/39 | 9/10 | 25.6% | 361 / 864 ms |
| Expanded normalization | 14/39 | 11/12 | 30.8% | 355 / 2,951 ms |
| Display rectification | 13/39 | 10/11 | 28.2% | 664 / 2,036 ms |
| Normalization + rectification | 16/39 | 12/13 | 33.3% | 703 / 2,797 ms |
| Digit ensemble throughout | 9/39 | 8/9 | 23.1% | 461 / 986 ms |
| Digit ensemble only as rescue | 11/39 | 9/10 | 25.6% | 709 / 1,714 ms |

The combined normalization-and-rectification variant recovered five exact
triplets beyond the current policy on v3. Its settings were fixed before this
run, but the set had already been inspected; selecting it here still requires a
new untouched test. Rerun latency differs from the original frozen run because
these measurements were collected separately.

The nine ML Kit variants were also rerun on the same Galaxy A52. The original-
image pass was exact on 1/39; every grayscale, crop, threshold, rectification and
strict-consensus variant was exact on 0/39. V3 still contains no negatives.

## Interpretation

This is stronger evidence than rerunning the already consumed v2 set: source
selection, transcription review, privacy review and overlap exclusions were
completed before Hearth inference. It is still a small public-source challenge
set, not a clinical study. The answers began as source annotations and were
checked by one reviewer rather than independently transcribed by two people.
Device and session correlations remain possible even after the hash and visual
audits.

The result confirms the earlier conclusion. Retry and crop logic improves
end-to-end recovery, but most valid images still receive no complete reading,
and a confident-looking accepted result can contain a clinically meaningful
digit error. Human confirmation of all three values remains mandatory.

After this run the suite is a consumed regression benchmark. Future changes
may be compared against it, but gains on it are not fresh held-out evidence.
