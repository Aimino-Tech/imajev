# Prefix scheduler benchmarks

These benchmarks isolate the remaining non-kernel work in the Torch shared-prefix
path. They do not install or change FLA, Triton, causal-conv1d, or other kernels.

## 1. Production showdown

Measure the current serving shape with CUDA synchronization, p50/p95, VRAM,
cold/warm gate, raw prefix, and canonical serial parity:

```bash
python scripts/bench_site_showdown.py \
  --runs 5 --warmups 1 --serial-runs 1 \
  --microbatch 16 \
  --suffix-bucket-width 8 \
  --question-prefill-batch 0
```

Run the same benchmark with bucketing disabled for the before/after comparison:

```bash
python scripts/bench_site_showdown.py \
  --runs 5 --warmups 1 --serial-runs 1 \
  --microbatch 16 \
  --suffix-bucket-width 0 \
  --question-prefill-batch 0
```

The production decision number is warm-gate p50 vs batch p50. Raw-prefix shows
the scheduler/model floor without one-time validation.

## 2. Scheduler sweep

Sweep suffix packing first, then wider question-prefix batches:

```bash
python scripts/bench_prefix_scheduler_sweep.py \
  --runs 3 --microbatch 16 \
  --suffix-buckets 0,8,16,24,32 \
  --prefill-batches 0,4,8,12,16,35
```

Each row reports p50/p95, peak VRAM, serial winner parity, parity against the
current prefix scheduler, suffix padding percentage, question-prefix batches,
suffix batches, LM calls, and cache-row selection branches. OOM configurations
are reported and skipped.

The sweep selects a candidate only when it introduces **no new winner flip
against the current prefix baseline** and does not reduce canonical-serial
parity. This intentionally tolerates the already-known ultra-low-margin Q17
serial/prefix disagreement instead of treating serial's 0.003 margin as ground
truth.

## Promotion rule

Promote a scheduler setting to the server default only when all of these hold:

- baseline parity is N/N against the current prefix scheduler;
- serial parity is not worse than the current prefix scheduler;
- p50 improves by at least ~3% across repeated runs;
- p95 does not regress materially;
- peak allocated VRAM stays inside the serving budget;
- the full production showdown confirms the gain after the isolated sweep.

The server exposes both knobs for testing:

```text
--prefix-suffix-bucket-width 8
--prefix-question-batch 0
```

`prefix-question-batch=0` keeps the conservative coupled scheduler. Positive
values enable the decoupled question-prefill cache path and should remain
experimental until the sweep picks a no-new-flip winner.
