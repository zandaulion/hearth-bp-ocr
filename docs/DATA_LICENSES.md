# Dataset provenance and redistribution

Checked on 8 October 2026. No dataset images or per-image ground-truth manifests
are intended for Git publication; local ignored test material may exist in a
working checkout. Public availability alone does not grant redistribution rights.
Copyright permission and privacy clearance are separate questions.

| Source | Declared license | Copyright redistribution conditions |
| --- | --- | --- |
| [Blood Pressure Monitor Display, Final Project, version 1](https://universe.roboflow.com/final-project-cwtfb/blood-pressure-monitor-display/dataset/1) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | Credit the creator/project, link the original source and license, retain supplied notices, and indicate modifications. |
| [Blood Pressure Monitor Digit Reader, naphop, version 9](https://universe.roboflow.com/naphop/blood-pressure-monitor-digit-reader/dataset/9) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | External-test source only. Credit the creator/project, source and license; preserve supplied notices and describe preprocessing or modifications. |
| [Wikimedia Commons](https://commons.wikimedia.org/) external-test candidates | Per-file Public Domain, CC BY or CC BY-SA metadata recorded in the ignored local manifest | Follow each file's source-page license and attribution terms; do not infer one license for the whole collection. |
| [BP Monitor Reading / Medical Device Images, DataCluster Labs](https://www.kaggle.com/datasets/dataclusterlabs/bp-monitor-reading-medical-device-images) | [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) in Kaggle's public dataset metadata | Copyright redistribution is permitted under CC0. Credit and provenance are still useful. |
| User-provided, camera, and catalog/reference photos | No public redistribution clearance established | Keep local; do not redistribute without appropriate permission. |

Kaggle's license was verified via its public dataset metadata endpoint, matching
the DataCluster Labs listing; the local Kaggle download has no bundled license
file. Confirm that any files selected for redistribution belong to that exact
dataset/version. Do not apply its license to an unrelated download.

Licenses do not override privacy, publicity, or other third-party rights. Before
adding any images, review the pixels and annotations for faces, names, personal
health records, identifiable surroundings, and device/account identifiers;
check EXIF for location and personal metadata. Record any redaction or crop.
Retain required attribution without copying unnecessary private metadata.

For a Roboflow data addition, an attribution can identify:
"Blood Pressure Monitor Display, Final Project, version 1, Roboflow Universe,
CC BY 4.0", followed by the source and license links above, the supplied notices,
and a description of modifications if any. Dataset licensing remains separate
from source-code and model licensing.

Large datasets are usually easier to obtain from their original hosts using
documented download instructions than to embed in normal Git history.

The eight owner-provided training photos have been cleared by their owner for
public distribution of the trained weights, not distribution of the photos.
The bundled weights use the Roboflow source above plus those photos; the Kaggle
download is not a training source for this release. See [model provenance](MODEL_CARD.md).
