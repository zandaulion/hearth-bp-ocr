# Aggregate development results

Evaluation date: 8 October 2026. These summarize the local research workflow;
the v2 ONNX models are bundled, while detailed per-photo records remain private.

| Model and dataset role | Correct complete readings | Accuracy with refusals | Candidates / eligible | Candidate precision |
| --- | ---: | ---: | ---: | ---: |
| v1 first frozen reserved test | 4 / 13 | 30.8% | 4 / 13 | 100% |
| v2 public validation, used for selection | 21 / 25 | 84.0% | 19 / 25 | 100% |
| v2 previously used test, regression only | 7 / 13 | 53.8% | 7 / 13 | 100% |
| v2 external challenge, staged Python reference, now regression-only | 27 / 100 | 27.0% | 24 / 100 | 91.7% |

The v2 model included additional local training examples whose photos and
individual readings are omitted. These aggregate figures do not predict the
results of training the public-only recipe in this repository.

Complete-reading accuracy requires SYS, DIA and pulse all to match the reference;
refusals count as misses. Candidate precision uses only accepted complete
readings. Report coverage alongside precision. The validation candidate precision
has a 95% Wilson interval of 83.2–100%, and validation was also used for selection.
The first frozen test had only four candidates, with a 51.0–100% interval.
Session and source correlations further weaken independent-observation assumptions.
**The >90% precision target on new captures remains unproven.**

The external challenge set combines 90 annotation-backed Roboflow test images
with 10 manually transcribed Wikimedia images. A hash audit excluded 13
near-matches to the original training corpus. Its single-pass result was 20/100
exact triplets with 17/19 correct candidates; the staged policy recovered seven
additional exact readings and produced 22/24 correct candidates. The staged
exact-accuracy Wilson interval is 19.3–36.4%, while candidate precision is
74.2–97.7%. Separately, all 25 analog/off/disassembled refusal cases returned no
reading (Wilson interval 86.7–100%). This public-source set is correlated,
partially annotation-derived, and was consumed by this evaluation; it is useful
for regression and error analysis, not an independent clinical claim. The full
collection, leakage-audit and evaluation record is in
[External test v2](EXTERNAL_TEST_V2.md).

## Comparison with BP Digitizer's closed model

The same consumed external v2 set was also run once through BP Digitizer's
production-configured `gemini-2.5-flash` model. Gemini read 92/100 complete
triplets exactly, compared with Hearth's 27/100. All 27 Hearth successes were
also Gemini successes; Gemini alone was exact on another 65 images, and both
systems missed eight.

Gemini produced 99 complete outputs, 92 of them exact (92.9% precision at 99%
coverage). Hearth produced 24 calibrated candidates, 22 exact (91.7% precision
at 24% coverage). These selective-output figures are not strictly equivalent:
Hearth applies an explicit confidence threshold, while Gemini has no comparable
calibrated score and was instructed to omit uncertain fields.

Both systems safely rejected all 25 negative cases under BP Digitizer's rule
that SYS and DIA must both be present. Gemini emitted a lone SYS value for one
case, so strict no-value refusal was 24/25 for Gemini and 25/25 for Hearth.
Observed median/p95 times were 250/337 ms for local Hearth Python inference and
2.96/17.79 seconds for Gemini including its network request. Those different
execution environments do not form a controlled speed comparison.

The complete setup, source breakdown, confidence intervals, cost calculation
and limitations are in [Hearth versus Gemini 2.5 Flash](GEMINI_COMPARISON.md).
This comparison does not validate either system clinically or restore the
consumed set's independence.

A subsequent [adaptive crop and lighting-correction revision](ADAPTIVE_CROP.md)
preserved these original validation/regression results and recovered additional
development examples. Its results are reported separately because it was tuned
after these model evaluations.

The later adaptive3 row-association change was selected through local regression
testing. It has not rerun the omitted aggregate corpora in this fresh public
checkout, so the table above must not be presented as adaptive3 validation.

Android browser inference and offline replay worked on a Galaxy A52. Warm OCR
was approximately 0.84 seconds on repeated development samples. Live Android
tests and a conservative portrait crop fallback demonstrated feasibility but
did not provide enough independent captures for a reliability claim. The fallback
requires two predefined crops to produce identical complete candidate readings
after a full-frame refusal; it can still refuse or misread new images.
