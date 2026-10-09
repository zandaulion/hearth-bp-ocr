# Hearth blood-pressure screen reader: a plain-language report

## The short version

Hearth is an experimental tool that tries to copy three numbers from a photo of
a home blood-pressure monitor:

- **SYS** — the top number;
- **DIA** — the middle number;
- **pulse** — the bottom number.

It does not measure blood pressure itself, diagnose anything, or decide whether
a result is healthy. It only attempts to transcribe digits already visible on a
screen.

The current system is cautious, but it is not yet dependable enough to trust
without checking the monitor. In the largest recent test, it read all three
numbers correctly in **27 of 100** readable photos. It labelled 24 results as its
best-quality “candidate” answers; **22 were correct and 2 were wrong**. It also
correctly refused all 25 images that did not contain a readable three-row
digital result.

The practical rule is simple: **always compare SYS, DIA and pulse with the
original monitor before using or saving the result.**

## What happens when a photo is read

The reader works in several steps:

1. It looks for the three horizontal groups of numbers on the monitor.
2. It finds the individual digit shapes inside those groups.
3. A second, smaller model checks which digit each shape represents.
4. The software arranges the groups from top to bottom as SYS, DIA and pulse.
5. It rejects layouts that are missing numbers, ambiguous or internally
   inconsistent.
6. If the first attempt fails, it may try carefully chosen crops or improve
   uneven lighting. A fallback result is used only when two altered views agree.

The app can return one of three practical outcomes:

- **Check values**: the best-quality automatic result, still requiring a human
  check;
- **Please review**: a reading was found, but something looks uncertain;
- **Retake photo**: no sufficiently clear complete reading was found.

“Check values” does not mean “guaranteed correct.” Two such results were wrong
in the 100-photo test.

## How the models were trained

The system uses two learned models.

### 1. The detector

The detector learns where the number rows and individual digits are. Training
photos include boxes drawn around these parts of the display. It began with a
general-purpose image model and was then trained specifically on blood-pressure
monitor screens.

The public training source contained 239 usable monitor images. The bundled
model was later fine-tuned with eight additional owner-approved photos. Those
private photos and their individual readings are not published.

During training, images were slightly rotated, shifted, resized and adjusted in
brightness. This helps the model learn that a digit is still the same digit when
the camera angle or lighting changes a little. Horizontal and vertical flips
were disabled because an upside-down or mirrored medical display is not a
realistic training example.

### 2. The digit checker

The digit checker receives small crops of detected digits and decides whether
each crop is 0 through 9 or background. It was trained with real annotated digit
crops, generated background examples and similar mild visual changes.

### GPU training and export

Training was run on a Runpod GPU. The training programs were updated so they can
choose CPU or GPU explicitly and stop with a clear error if a requested GPU is
not available. A repeatable helper now installs the required software, checks
the GPU, trains both models, exports them to browser-friendly ONNX files, tests
the export and packages the local reports.

The Roboflow download credential was supplied through a secret reference. Its
value was never printed or stored in the project. Temporary data-transfer and
audit computers were deleted after use, while the stopped training workspace
was left intact.

## What the training results mean

The training run produced several encouraging development results:

| Check | Result | Plain-language meaning |
| --- | ---: | --- |
| Detector mAP50 | 91.61% | The detector often placed boxes near the labelled rows and digits under a relatively forgiving overlap rule. |
| Detector mAP50–95 | 68.89% | Accuracy fell when box placement was judged more strictly. |
| Digit validation | 179/179 correct | Every already-cropped validation digit was classified correctly. This does not test finding the digits in a whole photo. |
| ONNX export parity | Passed | The exported browser model closely matched the training model's calculations. It does not prove that either answer is correct. |
| Synthetic checks | 14/14 passed | Hand-built logic examples behaved as expected. They are not real-photo accuracy evidence. |

On the validation images used during development:

- the basic complete reader got 52% of full three-number readings correct;
- the adjusted version got 72% correct;
- it reported a best-quality candidate on 48% of images;
- every candidate in that validation run was correct.

Those numbers are useful for building the model, but they are not an independent
test. The same validation data helped choose the settings, so it is normal for
the model to look better there than on truly new images.

## Why a larger outside test was needed

An early web test used only eight readable photos. The staged reader got six of
the eight exactly right and every accepted answer was correct. That looked
promising, but eight similar images are much too few to establish reliability.

A larger test set was therefore built from public sources:

- 90 readable images from a separate Roboflow test split;
- 10 readable images from Wikimedia Commons;
- 25 analog, powered-off, disassembled or cuff-only images that should produce
  no reading.

The images were reviewed before running the model. Incorrect labels, missing
rows, obvious repeated crops, out-of-scope hospital displays, unrelated search
results and images containing identifiable people were removed.

To reduce accidental “testing on the homework,” the candidate images were
compared with 309 images from the original model corpus. Thirteen near-duplicate
images were found and excluded. The final answers and image checksums were then
frozen before inference. The training pipeline now refuses to include paths from
this test collection.

## Results on 100 readable photos

An answer counted as correct only when **all three values—SYS, DIA and pulse—were
exactly right**. One wrong digit made the whole photo incorrect.

| Outcome | Number of photos |
| --- | ---: |
| No reading returned | 62 |
| Best-quality candidate, correct | 22 |
| Best-quality candidate, wrong | 2 |
| Lower-confidence review result, correct | 5 |
| Lower-confidence review result, wrong | 9 |
| **All three numbers correct in total** | **27/100** |

This can also be described with two common terms:

- **Candidate precision: 91.7%** — of the 24 answers labelled as the best
  candidates, 22 were correct.
- **Candidate coverage: 24%** — the system offered such a candidate for only 24
  of the 100 readable photos.

Precision and coverage must be reported together. Saying “91.7% precision” by
itself hides the fact that the system declined to offer a candidate on most
readable photos.

The statistical uncertainty is also substantial. A conventional 95% interval
for candidate precision spans roughly **74% to 98%**. The measured value is
above 90%, but the evidence does not prove that future candidate answers will
stay above 90%.

### Did the fallback steps help?

Yes, but not enough.

Without the staged fallbacks, the reader got 20/100 complete readings correct
and produced 17 correct candidates out of 19. With the fallbacks, it reached
27/100 and 22 correct candidates out of 24. The lighting-based adaptive step was
used eight times: it recovered seven exact readings, while the remaining result
was wrong but marked for review rather than accepted as a candidate.

No image in this test was recovered by the portrait-crop step.

### Which additional ideas helped most?

A follow-up experiment separated several common OCR suggestions so their impact
could be measured instead of guessed. The existing staged retries remained the
largest improvement: they raised complete accuracy from 20/100 for one pass to
27/100. Automatically straightening a likely display recovered one additional
photo with almost no change to typical runtime. More aggressive lighting
normalization also recovered one, but made a typical run roughly three times as
slow. Combining both reached 29/100, again at roughly three times the current
runtime. A multi-crop digit ensemble did not work well as a replacement; used
only after a refusal, it recovered one additional photo at substantial cost.

Google ML Kit Text Recognition v2 was also tested locally in a separate harness
on a Galaxy A52. Raw images, grayscale, detector-selected crops, thresholding,
straightening and strict agreement between retries all produced **0/100 exact
triplets**. Looking at ML Kit's raw text showed that no photo contained all three
correct values at once, so changing the parser could not recover a complete
answer. On this set, cropping made ML Kit more likely to refuse safely, but did
not make it a useful replacement for the specialized models. See the
[full ablation](OCR_ABLATION.md) for the per-step accuracy and latency table.

## Results on 25 images with no readable result

All 25 negative cases were refused:

- no reading was returned;
- no best-quality candidate was produced;
- measured refusal rate: **25/25, or 100%**.

This is good behavior, but 25 examples cannot cover every misleading screen,
sign, calculator, medical device or unusual photograph. A conventional 95%
interval for the refusal rate is approximately 87% to 100%.

## How did it compare with BP Digitizer's cloud model?

For comparison, the same 125 public test images were sent once to the
closed-weights Gemini 2.5 Flash model selected by BP Digitizer's production
configuration. This was a direct cloud-model test, not a complete Android app
test.

On the 100 readable photos:

- Hearth got all three numbers right in 27 photos;
- Gemini got all three numbers right in 92 photos;
- every photo Hearth read exactly was also read exactly by Gemini;
- Gemini alone read another 65 photos exactly;
- both systems missed eight photos.

Hearth offered 24 best-quality candidates and 22 were exact. Gemini returned
all three fields for 99 photos and 92 were exact. Those are not identical
confidence categories: Hearth applies a numerical acceptance threshold, while
Gemini does not provide a directly comparable confidence score and was asked to
leave out uncertain fields.

Both systems safely rejected all 25 negative cases under BP Digitizer's rule
that SYS and DIA must both be present. Gemini returned a SYS value alone for one
negative image, but the app would reject it because DIA was missing.

Hearth's local Python run took about 0.25 seconds for a typical image. The
Gemini cloud request took about 2.96 seconds at the median and 17.79 seconds at
the slowest five-percent boundary in this run. This is not a pure speed test:
one ran locally and the other included the internet and a cloud service.

The 125 recorded Gemini requests cost about $0.17 at the standard published
price used for the calculation. Hearth has no per-photo cloud-model fee and can
keep normal photos on the device. Gemini was much more accurate on this test,
but requires a network service and brings different cost and privacy tradeoffs.
Neither result proves medical safety, and these already-examined images are no
longer a fresh independent test.

## What these results tell us

A second model-blind outside test was later built from two other public image
collections. Generated variants, repeated sessions and images resembling the
older test were removed before the model ran. This left 39 new readable photos.
The simple full-photo pass read 8 correctly. The complete retry pipeline read
11 correctly, and 9 of its 10 accepted answers were right. In the remaining
accepted answer, one digit in the middle number was wrong.

This newer result tells the same basic story: retries help, but the system still
refuses most usable photographs and confidence filtering does not eliminate
digit errors.

The same 39 photos were then tested once with Gemini 2.5 Flash. Hearth got all
three numbers right in 11 photos, while Gemini did so in 33. Every Hearth
success was also a Gemini success. Hearth offered 10 best-quality candidates;
nine were exact. Gemini returned all three fields for every photo, and 33 of
those 39 complete outputs were exact. Those are different kinds of output:
Hearth deliberately withholds most answers through a confidence threshold,
while Gemini does not provide an equivalent calibrated score.

Typical measured processing time was about 0.23 seconds for Hearth locally and
2.53 seconds for the Gemini request including the network. This is not a fair
hardware speed contest, but it illustrates the practical local-versus-cloud
tradeoff. The 39 cases are now a consumed regression set, and they contain no
negative images for testing refusal safety. Full construction and results are
in [external test v3](EXTERNAL_TEST_V3.md).

The larger test gives a more realistic and less flattering picture than the
small pilot or validation results:

- the system is appropriately cautious on the tested non-readable images;
- the fallback logic improves the number of correct complete readings;
- most valid photos still end in refusal;
- some returned review results contain wrong digits;
- two high-quality candidate results were also wrong;
- therefore the reader is useful only as an assisted transcription tool with
  mandatory human verification.

The strong detector and cropped-digit scores did not translate into strong
end-to-end reading accuracy. Finding a digit crop, identifying it, grouping the
right digits into rows and rejecting unrelated screen numbers must all work at
the same time. A mistake at any step can spoil the full three-number result.

## What these results do not prove

They do not prove that the reader is medically safe or clinically validated.
Important limitations remain:

- 90 of the 100 readable images came from one public dataset;
- many answers came from that dataset's annotations rather than two independent
  people transcribing each display;
- repeated device types and related sources weaken the statistical assumptions;
- the test has now been examined and must be treated as a regression set, not a
  fresh holdout for future claims;
- the detailed run used the Python ONNX reference, which can differ slightly
  from browser image resizing and rounding;
- the intended scope is upright home monitors with three clearly ordered rows;
- the tool cannot judge whether a blood-pressure value is medically good, bad
  or urgent.

## How the test data will be used

The 125 cases are now permanently marked **test only**. They may be used to:

- detect whether a future code or model change makes performance better or
  worse;
- study broad error categories;
- verify that refusals remain conservative.

They may not be used to train either model, choose confidence thresholds, create
new augmented training images or claim fresh independent evidence.

For the next trustworthy improvement cycle, the project needs a new sealed set
from genuinely different devices and sessions, with answers independently
checked by two people. That set should remain unseen until the next model and
all acceptance settings are frozen.

## Privacy and safety

Normal photos selected in the browser app are processed locally and are not sent
to a project server. The public test images, answer manifests, contact sheets and
detailed reports remain in ignored local directories. No dataset images,
individual readings, secrets or infrastructure identifiers were committed or
pushed during this work.

Even with local processing, a monitor photo can contain personal surroundings,
device identifiers or health information. Do not publish such images or detailed
results without appropriate permission and review.

## Bottom line

The project has a sound experimental process: separate training and test data,
duplicate checks, frozen answers, export checks, conservative refusals and
repeatable reports. The model itself still needs substantial improvement.

At present, Hearth can assist someone who is already looking at the monitor and
will verify every digit. It should not be used as an unattended reader, as a
source of medical advice, or as the sole record of a measurement.

For the full technical record, see [Runpod training](RUNPOD_TRAINING.md),
[external test v3](EXTERNAL_TEST_V3.md),
[external test v2](EXTERNAL_TEST_V2.md),
[Hearth versus Gemini](GEMINI_COMPARISON.md), [aggregate results](RESULTS.md),
and the [model card](MODEL_CARD.md).
