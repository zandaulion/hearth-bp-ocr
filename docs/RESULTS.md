# Aggregate development results

Evaluation date: 6 October 2026. These summarize the local research workflow;
the v2 ONNX models are bundled, while detailed per-photo records remain private.

| Model and dataset role | Correct complete readings | Accuracy with refusals | Candidates / eligible | Candidate precision |
| --- | ---: | ---: | ---: | ---: |
| v1 first frozen reserved test | 4 / 13 | 30.8% | 4 / 13 | 100% |
| v2 public validation, used for selection | 21 / 25 | 84.0% | 19 / 25 | 100% |
| v2 previously used test, regression only | 7 / 13 | 53.8% | 7 / 13 | 100% |

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
