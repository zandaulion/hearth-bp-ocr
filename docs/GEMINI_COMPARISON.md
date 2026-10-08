# Hearth versus Gemini 2.5 Flash

Evaluation date: 8 October 2026.

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

## Readable-image results

An exact result requires all three fields to match. One wrong or omitted field
makes the entire triplet incorrect.

| Metric | Hearth | Gemini 2.5 Flash |
| --- | ---: | ---: |
| Exact SYS/DIA/pulse triplets | 27/100 | 92/100 |
| Exact-triplet accuracy | 27.0% | 92.0% |
| Exact accuracy, 95% Wilson interval | 19.3–36.4% | 85.0–95.9% |
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
