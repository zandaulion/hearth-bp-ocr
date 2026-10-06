# Public repository boundary

This repository is a reviewed source export of a separate local research
workspace. Independently authored source, synthetic tests, generated app icons, dataset links,
required dependency notices, reviewed ONNX exports, and aggregate findings are included.
Original source and documentation are licensed under AGPL-3.0-only; third-party
notices remain intact. The initial public commit was replaced with a cleaned
snapshot to exclude the starter implementation from the branch's reachable history.

Excluded items include the user-supplied starter code, personal photos, screenshots, videos, per-photo answers,
camera/device reports, training manifests with local paths, logs, other model
checkpoints, downloaded datasets, credentials, private SSH configuration, and
machine/account addresses. Exclusions preserve the local files; they do not delete
them. Git author metadata uses the public GitHub account's noreply address.

The browser sample picker and private sample-copying step were removed in this
public snapshot. The timing harness accepts a local ignored manifest instead of
embedding personal readings. Metric tests can import annotation logic without a
downloaded dataset; class-order validation still runs when real labels are parsed.

Run `python tools/check_public_tree.py` before pushing. It checks tracked paths
and common sensitive text patterns, and pins the two allowed ONNX files to their
reviewed SHA-256 hashes. Their protobuf metadata and graph strings were reviewed;
they contain no external tensor references. The [model card](MODEL_CARD.md)
records provenance and the limits of this review. The checker cannot prove arbitrary content is
free of personal information. Review new files, especially images, before staging.
Ignored files already tracked by Git remain tracked, so inspect the index too.

Future work intended for publication should take place in this repository, with
local test data kept in its ignored directories. Import changes from another
workspace selectively and review their diff; do not bulk-copy local reports or
add everything with `git add -f`.
