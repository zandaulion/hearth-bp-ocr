# Hearth versus Gemini 2.5 Flash

Initial evaluation: 8 October 2026. Fresh external-v3 follow-up: 9 October
2026.

This report compares the bundled Hearth ONNX pipeline with the closed-weights
Gemini model currently selected for production by BP Digitizer's live Firebase
Remote Config. It is a benchmark of the two recognition approaches on the same
consumed public test set, not a clinical validation or a new independent test.

## What was compared

Both systems processed the frozen [external test v2](EXTERNAL_TEST_V2.md):

- 100 readable photographs with reference SYS, DIA and pulse values;
- 25 analog, powered-off, disassembled or cuff-only images that should not be
  accepted as readings.

Hearth used its released `v2-adaptive3` staged Python ONNX reference. Gemini was
called through the Vertex AI global endpoint with the production model resolved
from live Remote Config: `gemini-2.5-flash`. Every successful response reported
that model version. The run completed all 125 cases with no final request errors
and one automatically retried transient request.

The Gemini run used BP Digitizer's current extraction prompt. Its preprocessing
approximated the Android app's light preprocessing: EXIF orientation, a maximum
dimension of 1,024 pixels, grayscale conversion and JPEG encoding at quality
100. Three requests ran concurrently. Each image received one model response
using the service's default generation behavior.

This called Vertex AI directly. It did not exercise the complete Android,
Firebase AI Logic, App Check or user-interface path, so it should be understood
as a model comparison rather than an end-to-end BP Digitizer app benchmark.

## Fresh external-v3 follow-up

A second benchmark was assembled from two other pinned public datasets. Its 39
readable cases, answers, exclusions and checksums were frozen before either
model result was inspected. Generated variants, repeated sessions and
perceptual overlaps with external v2 were excluded. The complete construction
record is in [External test v3](EXTERNAL_TEST_V3.md).

The same released Hearth policy and the same Gemini model, prompt and light
preprocessing were used. All 39 hosted-model requests completed without an
error or retry.

| Metric | Hearth | Gemini 2.5 Flash |
| --- | ---: | ---: |
| Exact SYS/DIA/pulse triplets | 11/39 | 33/39 |
| Exact-triplet accuracy | 28.2% | 84.6% |
| Exact accuracy, 95% Wilson interval | 16.5–43.8% | 70.3–92.8% |
| Correct SYS fields | 12/39 | 38/39 |
| Correct DIA fields | 11/39 | 35/39 |
| Correct pulse fields | 12/39 | 35/39 |

| Outcome on the same image | Images |
| --- | ---: |
| Both systems exact | 11 |
| Hearth exact, Gemini not exact | 0 |
| Gemini exact, Hearth not exact | 22 |
| Neither system exact | 6 |

Hearth produced 10 accepted candidates, nine exact: 90.0% precision at 25.6%
coverage. Gemini returned complete outputs for all 39 images, of which 33 were
exact: 84.6% exact output at 100% coverage. Those figures are not directly
equivalent because only Hearth applies a calibrated candidate threshold.

| Source | Hearth exact | Gemini exact |
| --- | ---: | ---: |
| Ega | 5/11 | 11/11 |
| DataCluster | 6/28 | 22/28 |

Observed median/p95 wall time was 232/456 ms for local Hearth and 2.53/5.39
seconds for Gemini including the network request. Gemini was about 10.9 times
slower at the median and 11.8 times slower at p95, but the execution environments
were different. The hosted run used 67,229 tokens in total. External v3 has no
negative cases, so it adds no comparative refusal evidence.

This fresh follow-up supports the original finding: Gemini recovered many more
exact triplets, while Hearth's advantages remain local execution and an explicit
selective-output threshold. The v3 set became a consumed regression benchmark
as soon as these results were inspected.

## Readable-image results

An exact result requires all three fields to match. One wrong or omitted field
makes the entire triplet incorrect.

### All measured variants on the combined consumed benchmark

This post-hoc table combines the 100 external-v2 and 39 external-v3 readable
images. The 25 negative cases all come from v2 because v3 contains no negatives.
Both suites were already consumed before this aggregation, so the larger
denominator improves regression measurement but does not create fresh held-out
evidence.

For Hearth, a selective output is a calibrated candidate. For ML Kit, it is an
app-accepted result containing SYS and DIA. For Gemini, it is a complete
SYS/DIA/pulse output. Precision is exact-triplet precision within that row's
selective outputs; coverage is the fraction of the 139 readable images that
produced one. These different output definitions make the table useful for
comparison, but not perfectly like-for-like.

| System / pipeline | Exact triplets | Selective outputs | Exact-output precision | Coverage | Safe negatives | Median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Hearth: one full-frame pass | 28/139 | 23/25 exact | 92.0% | 18.0% | 25/25 | 87 / 178 ms |
| Hearth: current staged retries | 38/139 | 31/34 exact | 91.2% | 24.5% | 25/25 | 253 / 796 ms |
| Hearth: expanded normalization | 42/139 | 33/36 exact | 91.7% | 25.9% | 25/25 | 757 / 1,886 ms |
| Hearth: display rectification | 41/139 | 33/36 exact | 91.7% | 25.9% | 25/25 | 262 / 1,446 ms |
| Hearth: normalization + rectification | 45/139 | 35/38 exact | 92.1% | 27.3% | 25/25 | 767 / 2,162 ms |
| Hearth: digit ensemble throughout | 35/139 | 28/29 exact | 96.6% | 20.9% | 25/25 | 262 / 898 ms |
| Hearth: digit ensemble only as rescue | 39/139 | 31/34 exact | 91.2% | 24.5% | 25/25 | 520 / 1,617 ms |
| ML Kit: original image | 1/139 | 1/24 exact | 4.2% | 17.3% | 13/25 | 63 / 334 ms |
| ML Kit: full-image grayscale | 0/139 | 0/22 exact | 0% | 15.8% | 19/25 | 64 / 192 ms |
| ML Kit: detector row crop | 0/139 | 0/9 exact | 0% | 6.5% | 25/25 | 65 / 121 ms |
| ML Kit: row crop + grayscale | 0/139 | 0/7 exact | 0% | 5.0% | 25/25 | 72 / 133 ms |
| ML Kit: row crop + Otsu | 0/139 | 0/1 exact | 0% | 0.7% | 25/25 | 68 / 158 ms |
| ML Kit: row crop + Otsu + dilation | 0/139 | 0/0 | — | 0% | 25/25 | 64 / 178 ms |
| ML Kit: rectified display | 0/139 | 0/9 exact | 0% | 6.5% | 24/25 | 71 / 287 ms |
| ML Kit: rectified + grayscale | 0/139 | 0/9 exact | 0% | 6.5% | 24/25 | 74 / 222 ms |
| ML Kit: strict agreement across passes | 0/139 | 0/5 exact | 0% | 3.6% | 22/25 | 460 / 1,205 ms |
| Gemini 2.5 Flash | 125/139 | 125/138 exact | 90.6% | 99.3% | 25/25 | 2.76 / 15.24 s |

The Hearth rows combine the v2 and v3 local Python ablation runs. The ML Kit
rows combine two runs on the same Galaxy A52; detector preprocessing for derived
crops was measured separately and is not included in their latency column.
Gemini latency includes the network and hosted service. Runtime numbers therefore
describe the observed deployments, not intrinsic model speed. The complete
ablation setup is in
[OCR preprocessing and retry ablation](OCR_ABLATION.md).

“Safe negatives” means that BP Digitizer's SYS-and-DIA acceptance rule rejected
the result. The detector produced row crops for 113/139 readable images and
rectified views for 66/139; unavailable derived views count as no reading in the
139-image denominator.

The normalization-plus-rectification variant recovered seven exact triplets
beyond the current staged policy and lost none of the current policy's exact
successes. It reached 45/139 exact (32.4%, 95% Wilson interval 25.2–40.5%) and
35/38 exact candidates (92.1%, interval 79.2–97.3%) at roughly three times the
median latency. Because this variant is being selected after inspecting the
consumed benchmark, those gains are a regression result and require confirmation
on a new untouched set.

Every one of those 45 exact Hearth results was also exact for Gemini. Gemini was
exact alone on another 80 images, and both missed 14. ML Kit's original-image
pass recovered one exact triplet in v3, making 1/139 combined; every processed
ML Kit variant remained at 0/139.

### Original external-v2 released Hearth and Gemini by field

On external v2 alone, the released Hearth row is the current staged-retry
pipeline. Its exact accuracy has a 95% Wilson interval of 19.3–36.4%; Gemini's
interval is 85.0–95.9%.

| Metric | Hearth | Gemini 2.5 Flash |
| --- | ---: | ---: |
| Correct SYS fields | 33/100 | 98/100 |
| Correct DIA fields | 34/100 | 97/100 |
| Correct pulse fields | 34/100 | 96/100 |

The paired result makes the difference especially clear:

| Outcome on the same image | Images |
| --- | ---: |
| Both systems exact | 27 |
| Hearth exact, Gemini not exact | 0 |
| Gemini exact, Hearth not exact | 65 |
| Neither system exact | 8 |

All 27 images read exactly by Hearth were also read exactly by Gemini. Gemini
read another 65 exactly. This is a strong result on this consumed benchmark,
but it does not turn the set into independent held-out evidence.

### Outputs and confidence are not directly equivalent

Hearth has a calibrated candidate threshold. Gemini does not expose an
equivalent calibrated confidence value, and the prompt instead asks it to omit
uncertain fields. The closest observable comparison is therefore useful but not
perfectly like-for-like:

| Selective-output metric | Hearth candidate | Gemini complete output |
| --- | ---: | ---: |
| Outputs | 24/100 | 99/100 |
| Exact outputs | 22 | 92 |
| Coverage | 24.0% | 99.0% |
| Exact precision among these outputs | 91.7% | 92.9% |
| Precision, 95% Wilson interval | 74.2–97.7% | 86.1–96.5% |

BP Digitizer's current acceptance rule requires SYS and DIA but not pulse. By
that rule Gemini returned an accepted result for 100/100 readable images, of
which 92 contained the exact full triplet. That app rule is different from both
Hearth's candidate definition and the complete-output comparison above.

### Results by source

| Source | Hearth exact | Gemini exact |
| --- | ---: | ---: |
| Roboflow v9 test | 24/90 | 85/90 |
| Wikimedia Commons | 3/10 | 7/10 |

The ten-image Wikimedia group is too small for a stable source-specific claim.

## Refusal-image results

| Metric | Hearth | Gemini 2.5 Flash |
| --- | ---: | ---: |
| Safe under BP Digitizer's SYS-and-DIA acceptance rule | 25/25 | 25/25 |
| Incorrectly accepted | 0/25 | 0/25 |
| Returned no field values at all | 25/25 | 24/25 |

Gemini returned only a SYS value for one negative case. BP Digitizer would
reject that response because DIA was absent. The 25/25 app-safe refusal result
has a 95% Wilson interval of 86.7–100% for either system. These 25 cases are a
narrow test and do not cover every kind of misleading display or photograph.

## Runtime and cost

| Observation | Hearth | Gemini 2.5 Flash |
| --- | ---: | ---: |
| Median end-to-end time | 250 ms | 2.96 s |
| p95 end-to-end time | 337 ms | 17.79 s |
| Where inference ran | Local Python CPU | Network request plus Google cloud |

Gemini was about 11.9 times slower at the median and 52.8 times slower at p95 in
this run. This is not a controlled hardware benchmark: Hearth ran locally while
Gemini included network and managed-service time, and neither measurement is
phone-browser latency.

The 125 successful Gemini requests used 185,945 prompt tokens, 3,396 candidate
tokens and 41,497 reasoning tokens. At the published standard Gemini 2.5 Flash
[Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)
used for this calculation, the recorded requests cost an estimated **$0.168**:
$0.0558 for input and $0.1122 for output/reasoning. This excludes negligible
smoke/restart requests, discounts, taxes and future pricing changes.

Hearth has no per-request model fee and can keep normal user photos on-device.
Gemini requires a cloud request, with the resulting connectivity, service,
privacy and billing considerations.

## Interpretation

On this particular benchmark, Gemini produced far more exact full readings
while preserving the same measured app-safe refusal behavior. Its complete
outputs had roughly the same measured precision as Hearth's much smaller set of
calibrated candidates, but covered nearly the whole readable set.

Hearth's advantages are different: its weights and decision logic are bundled,
inference can run locally without a per-image cloud charge, and its candidate
threshold is explicit. Its present accuracy and coverage are not competitive
with Gemini on these images, so it still requires substantial model improvement.

## Limitations

- This public set has already been inspected and evaluated. It is now a
  regression benchmark, not fresh independent evidence.
- Ninety readable images come from one annotated dataset, with related devices
  and source correlations.
- Neither system was evaluated clinically or on a representative population of
  devices, lighting conditions and users.
- Gemini is a changing hosted service. The model alias, serving implementation,
  default generation behavior, latency and pricing can change.
- The Gemini run used one response per image. Generative output is not
  guaranteed to be identical on a repeat run.
- The direct Vertex preprocessing closely approximated, but did not execute,
  the complete Android production path.
- Runtime measurements compare different execution environments and should not
  be treated as a pure model-speed comparison.

Only public benchmark images were sent to the hosted model. The detailed
per-image predictions remain in an ignored local report; no images, readings,
credentials, tokens, project identifiers or request records are included in
Git.
