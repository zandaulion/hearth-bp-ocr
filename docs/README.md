# Documentation

Hearth reads SYS, DIA and pulse from photographs of upright, three-row home BP
monitor displays. It includes two ONNX models and the preprocessing, decoding
and reading-selection code needed to use them. It is not a general OCR SDK or
a medically validated measurement system.

| Your task | Start here |
| --- | --- |
| Run the existing PWA | [Repository quick start](../README.md#browser-setup) |
| Understand the project without technical background | [Plain-language report](PLAIN_LANGUAGE_REPORT.md) |
| Add OCR to a website, PWA or Android project | [Integration guide](INTEGRATION.md) |
| Give an AI coding agent the integration context | [AI integration handoff](AI_INTEGRATION.md) |
| Implement a client or port the pipeline | [API and model contract](API_REFERENCE.md) |
| Understand how the code fits together | [Architecture and source map](ARCHITECTURE.md) |
| Train, evaluate, debug or publish changes | [Development guide](DEVELOPMENT.md), [Runpod training record](RUNPOD_TRAINING.md) |
| Check model origins, rights and limitations | [Model card](MODEL_CARD.md), [data licenses](DATA_LICENSES.md), [dependency notices](../prototype/THIRD_PARTY.md) |
| Interpret the existing evidence | [Results](RESULTS.md), [external test v2](EXTERNAL_TEST_V2.md), [adaptive crop experiments](ADAPTIVE_CROP.md) |
| Understand the public/private boundary | [Publication policy](PUBLICATION.md) |
| Test phones through an SSH host | [Android skill](../skills/lenovo-android/SKILL.md) |

Runnable integration examples live in [examples](../examples/README.md).
Paths in these documents are relative to the repository root unless stated
otherwise. Commands assume the repository root as the working directory.
The interface described here is the current `v2-adaptive3` implementation;
there is no published npm package, Python package or stable versioned SDK.

The fastest route for another AI agent is to read `AI_INTEGRATION.md`, then
`INTEGRATION.md` and `API_REFERENCE.md`, and inspect the referenced source.
