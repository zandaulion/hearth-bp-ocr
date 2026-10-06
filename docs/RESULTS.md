# Aggregate development results

Evaluation date: 6 October 2026. These summarize the local research workflow;
trained models and detailed per-photo records are not distributed in this repo.

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

Android browser inference and offline replay worked on a Galaxy A52. Warm OCR
was approximately 0.84 seconds on repeated development samples. Live Android
tests and a conservative portrait crop fallback demonstrated feasibility but
did not provide enough independent captures for a reliability claim. The fallback
requires two predefined crops to produce identical complete candidate readings
after a full-frame refusal; it can still refuse or misread new images.
