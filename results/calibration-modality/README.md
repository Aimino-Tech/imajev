# Per-modality temperatures for imajev-4b (phase 3, adapter `c9e5f132`)

Prompted by a reader's question on the launch thread: the card said photo-only verification was already calibrated raw and the single
shipped temperature over-softened it, so why not fit one temperature for text-only requests and one for requests with an image?
We did, on 2026-09-29, with data already on disk. Script: `scripts/calibration/fit_modality_temperatures.py`; numbers: `fit-report.json`.

## What the data says

Temperature fitted by held-out negative log-likelihood on each set, single-pass logits of the phase-3 4B. ECE is 10-bin, max-probability.

| Held-out set | n | Fitted T | ECE raw | ECE with shipped 1.305 | ECE with own T |
|---|---:|---:|---:|---:|---:|
| Constructed image groups + state/pairs probes (image + record) | 1,860 | 1.296 | 0.033 | 0.015 | 0.015 |
| **Photo-only verification, ABO + VizWiz (image, empty state)** | 823 | **1.028** | **0.015** | **0.029** | **0.012** |
| MMLU questions with an unrelated photo | 1,000 | 1.634 | 0.120 | 0.068 | 0.040 |
| MMLU text-only | 1,000 | 1.672 | 0.118 | 0.072 | 0.031 |
| Authored dev + p2b test (text, the shipped fit's neighbourhood) | 585 | 1.270 | 0.033 | 0.023 | 0.023 |
| Human dev, image rows | 351 | 1.326 | 0.078 | 0.042 | 0.040 |
| Human dev, text rows | 434 | 1.457 | 0.064 | 0.027 | 0.037 |

Two things follow. First, "image present" is not the split that matters: image requests that carry a record want the same
temperature as text (about 1.3), and only photo-only requests, an image with an empty state, are calibrated raw. Second, the
card's caveat still holds on the current adapter: on photo-only verification the shipped temperature raises ECE from 0.015 to
0.029, and a temperature of 1.03 brings it to 0.012. (Knowledge questions, MMLU-style, would want a softer 1.65, but that is a
task, not a request shape the server can see; not shipped.)

A single "image" temperature pooled over all image rows (1.225) changes nothing measurable on ImajevBench:

| ImajevBench split | Track | n | ECE raw | shipped | pooled image T |
|---|---|---:|---:|---:|---:|
| public test | image (visual + joint) | 242 | 0.068 | 0.044 | 0.044 |
| public test | text | 37 | 0.144 | 0.120 | 0.120 |
| hidden private-1 | image | 172 | 0.031 | 0.033 | 0.032 |
| hidden private-1 | text | 30 | 0.077 | 0.040 | 0.040 |

## What ships

`calibration-modality.json` and `calibration-rot4-modality.json` (schema 1.2): the shipped buckets unchanged, plus
`photo_only_temperatures` = 1.028 (823 rows) that the server applies only when a request carries images and an empty state.
`--calibration calibration-rot4-modality.json` therefore reproduces every published number for text and image-plus-record requests and
fixes the photo-only path. Argmax never changes. The older files keep working; a schema-1.0 file ignores the flag.

Server change: `scripts/playground/server.py` passes `photo_only = bool(images) and not request.state`;
`TemperatureCalibrator.temperature(..., photo_only=True)` picks the bucket. Tests: `tests/test_temperature_calibration.py`.
