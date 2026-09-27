# Fast torch serving path — H100 test, 2026-09-27

One H100 80GB (secure, US-MO-1, $3.49/h, ~36 min, terminated). Script: `cloud/pod_fast_path.sh`,
then a rerun (`fp/rerun.log`) for the unmerged variant. Public repo at the submitted server commit `a0134749` with the changed
files overlaid; adapter `mohit67890/imajev-4b@c9e5f132` (sha256 matches the phase-3 picked adapter); `--rotations 1
--calibration calibration.json`; flash-linear-attention 0.5.2 and causal-conv1d installed on every server.

- **A** = the submitted setting (current code path).
- **B** = `--fast --merge-lora` (LoRA folded into bf16 weights, one tokenization, CUDA graphs of the language model).
- **C** = `--fast` (LoRA unmerged, one tokenization, CUDA graphs). This is the default `--fast`.

## JevBench public tiers (JevBench's own runner, serial; latency = client wall time per decision)

| tier | A acc / ECE | A p50 / p95 | B acc / ECE | B p50 / p95 | C acc / ECE | C p50 / p95 |
|---|---|---|---|---|---|---|
| easy (48) | 100 / 0.006 | 118 / 138 ms | 100 / 0.006 | 16 / 18 ms | 100 / 0.006 | 24 / 26 ms |
| original (72) | 98.6 / 0.066 | 105 / 120 ms | 98.6 / 0.069 | 17 / 18 ms | 98.6 / 0.066 | 26 / 27 ms |
| hard (111) | 73.0 / 0.072 | 86 / 188 ms | 72.1 / 0.103 | 27 / 106 ms | 71.2 / 0.076 | 45 / 182 ms |

Every changed answer is a near-tie: B flips 2 hard items (A's top probability 0.36 and 0.34), C flips 2 (0.34 and 0.52);
largest per-item probability change 0.030 (B) / 0.034 (C). The same adapter measured 71.2 / 0.082 on hard in the phase-3
run, so these differences are bf16 run-to-run noise, not a change of model.

## ImajevBench public image requests (300, interleaved A/B and A/C, client wall time)

| | same answer | max prob diff | p50 | mean |
|---|---|---|---|---|
| A vs B (merged) | 296 / 300 | 0.169 | 260 → 200 ms | 254 → 178 ms |
| A vs C (unmerged) | 297 / 300 | 0.059 | 244 → 243 ms | 266 → 218 ms |

Where an image request's time goes on B (median of 12, 1,661-token prompt, 64x96 patch grid): HTTP + JSON parse of the ~390 KB
body ~48 ms, image decode 14.5, processor (resize/normalise/tokenize) 18.3, copy to GPU 8.0, vision encoder 34.4 (eager),
language model graph 41.8. The graphs remove launch overhead, which dominates short text prompts; image requests are
dominated by work before the language model, so they gain far less.

## Takeaways

- Text: 4-6x faster at the median with the same decisions within bf16 noise. On JevBench's Speed formula (standard+judge,
  self-hosted x2 + 0.15 s) that is roughly 88 -> 94.
- Images: 0-25 % faster. More needs work outside the model: request parsing, CPU preprocessing, a graphed vision encoder.
- `--merge-lora` is faster on long and image prompts. (Run 1 read its near-tie flips as a quality cost; run 2 below measured
  every path against a float32 reference and found the merged path as close to it as the submitted one.)

# Run 2 — against a float32 reference (same day, H100, EUR-NO-2, 8 vCPU, ~18 min, terminated)

Rule fixed before the run: a serving path keeps quality if its decisions are no further from a float32 server (`--float32`,
LoRA unmerged, eager) than the current bf16 path's are. `--fast` now also finishes image normalisation on the GPU (pixels
bit-identical to the processor's, tests/test_torch_fast_path.py). Everything `--rotations 1 --calibration calibration.json`.
Results: `run2/fp/` (`jevbench-vs-ref.json`, `images/summary.json`, `profile-image.txt`).

| vs float32 reference | JevBench hard flips | hard max prob diff | image flips (300) | image p90 / max prob diff |
|---|---|---|---|---|
| current (submitted) | 1 (ref top 0.34) | 0.030 | 4 (ref tops 0.41-0.55) | 0.030 / 0.186 |
| --fast | 1 (0.50) | 0.018 | 5 (0.42-0.70) | 0.029 / 0.208 |
| --fast --merge-lora | 1 (0.36) | 0.022 | 4 (0.41-0.51) | 0.025 / 0.164 |

Easy and original tiers: no flips anywhere; accuracy 100 / 98.6 on every server. Hard accuracy / ECE: ref 72.1 / 0.091,
current 73.0 / 0.072, fast 71.2 / 0.076, merged 72.1 / 0.103 (111 items: the float32 model itself sits between the bf16 runs).

| latency (serial) | original p50 / p95 | hard p50 / p95 | images p50 / p95 |
|---|---|---|---|
| ref (float32) | 54 / 56 ms | 149 / 750 ms | 572 / 590 ms |
| current | 59 / 68 ms | 75 / 182 ms | 149 / 154 ms |
| --fast | 19 / 19 ms | 39 / 172 ms | 129 / 134 ms |
| --fast --merge-lora | 11 / 11 ms | 22 / 96 ms | 91 / 95 ms |

This machine's CPU is faster than run 1's (current images 149 vs ~250 ms there); request handling outside the model is now
~10 ms of a --fast image request (profile-image.txt). Conclusion: `--fast --merge-lora` is as close to the float32 model as
the submitted path on both suites, and 1.6x faster on images, 5x on text.
