# Adaptive crop and lighting correction

The reader can refuse or assemble an implausible reading when a row proposal
ends inside a narrow edge digit, sits outside the fixed central crops, or uneven
illumination hides digits. After the existing pipeline returns no plausible
reading, the worker proposes a crop from two or three detected numeric rows,
leaving space below the upper rows for the pulse reading.

Two versions of the crop divide grayscale intensities by a locally smoothed
background. Each background uses three replicated-border box blurs, with radii
of 4% and 5% of the shorter crop dimension. Running sums keep preprocessing
linear in pixel count. Both versions must return plausible complete readings
and agree on all three fields. Otherwise the original refusal remains.
Candidate output requires both views to meet the existing confidence threshold.
If either view is low-confidence, the output remains **Please review** and is
excluded from accepted-reading precision. No confidence threshold is lowered.
Overlay coordinates are translated back to the source photo. Existing plausible
readings and model weights are unchanged. An implausible complete reading can be
replaced only under the same two-view agreement rule used for refusals.

The geometry and normalization parameters were tuned using two consented local
development photos. Those photos and their readings are excluded from this repository.
They cannot subsequently serve as independent test evidence. Agreement between
these correlated views does not establish correctness or calibrated confidence.

## Development checks

Compared with the previous fixed-crop pipeline, exact complete readings changed
as follows. Refusals count as misses. Synthetic portraits are transformations of
existing examples, not new observations.

| Dataset role | Original images, before → after | Synthetic portraits, before → after |
| --- | ---: | ---: |
| Public validation, 25 images | 21 → 21 | 3 → 5 |
| Previously consumed test, 13 images | 7 → 7 | 2 → 2 |
| Curated development, 9 eligible images | 3 → 5 | 1 → 1 |
| Prior private training, 8 images | 8 → 8 | 3 → 8 |

All nine added complete readings in these groups matched their references;
existing returned readings were preserved. These are small development and
regression sets, including data used for selection. The public-only training
recipe does not reproduce the bundled models used for these measurements,
which also used private training photos.

The first adaptive revision recovered the initial tuning photo, then refused
the next fresh capture. That failure remains in the live results and was
subsequently added to development. The final revision recovered both photos on
a Galaxy Z Fold4 running Chrome 154 in 1.17 and 1.10 seconds respectively,
including earlier unsuccessful inference passes. The second result is explicitly
low-confidence and requires review. These are replays, not new captures or a
latency benchmark. A subsequent fresh Fold4 capture on this final revision
returned all three values correctly in 1.16 seconds, confirmed by the user.
That is one successful fresh attempt on the final revision, on the same known
monitor. The earlier four refusals occurred across previous pipeline/browser
revisions and remain recorded locally. These observations do not establish the
greater-than-90% prospective precision target.

Twenty-five JavaScript and fourteen Python checks cover assembly, fallback and
metrics behavior. The JavaScript suite and a dependency-free Python assembly
replay pass in a fresh checkout; the complete Python suite requires the pinned
inference dependencies. A synthetic grayscale fixture previously produced
identical normalization bytes in the Python/OpenCV reference and JavaScript
implementation. Browser resizing can still change detections at pixel
boundaries; final validation must exercise the browser pipeline.

## Adaptive3 edge-row revision

Additional local regression testing exposed implausible SYS assemblies caused
by a row proposal ending before a narrow edge digit. The
adaptive3 postprocessor associates a digit whose box edge is within 20% of row
height, and permits agreeing fallbacks after an implausible full-frame result.
Local browser replays exercised both recovery and control behavior. These checks
were used to select the change and are not independent evidence; the earlier
public and regression aggregate tables predate adaptive3.

## Android compatibility observation

Chrome 134 on the tested Fold4 could not grow the runtime's shared WebAssembly
memory. A temporary cap allowed inference but was unstable. Updating that phone
to Chrome 154 allowed the unmodified runtime to initialize and complete replays.
The memory-cap experiment is not part of this implementation.
