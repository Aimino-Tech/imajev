# About this fork

This is a performance-focused community fork of [mohit67890/imajev](https://github.com/mohit67890/imajev). Upstream remains the source of the model family, training/evaluation lineage, model cards and canonical citation.

## What this fork adds

- exact per-question result caching, optionally persistent;
- one-time visual preprocessing per shared request;
- shared multimodal prefix/KV reuse across questions and rotations;
- hierarchical global-prefix → question-prefix → rotation-suffix execution;
- cross-question batching with bounded VRAM;
- length-aware suffix packing; 8-token buckets are the production default;
- winner-parity validation for the approximate prefix path;
- scheduler/production benchmarks with p50/p95, peak VRAM and parity;
- runtime metadata for LM calls, padding waste and prefix batches.

`prefix-question-batch=0` remains the production default because wider question-prefix batches introduced new winner flips.

## Current measured result

Repeated 35-question × 4-rotation visual showdowns on 6 October 2026 measured the bucket=8, qbatch=0 warm shared-prefix path at **4.39–4.46 s p50**, versus roughly **14.1–14.6 s** for production batching and **17.0–17.8 s** for canonical serial scoring. The latest repeat was 4.39 s: **3.21× faster than batch and 3.88× faster than serial**.

Winner parity against canonical serial was 34/35 on the latest repeat. The single disagreement is the documented ultra-low-margin Q17 case. Treat these numbers as benchmark-specific, not universal latency claims.

Reproduce:

```bash
python scripts/bench_prefix_scheduler_sweep.py --runs 3 --microbatch 16 --suffix-buckets 0,4,8,16,24,32 --prefill-batches 0,4,8,12,16,35
python scripts/bench_site_showdown.py --runs 5 --warmups 1 --serial-runs 1 --microbatch 16 --suffix-bucket-width 8 --question-prefill-batch 0
```

Shared-prefix inference is not bit-identical to a full B=1 Qwen3.5/HF bf16 forward. Consumers requiring bit-identical logits should use `--no-shared-prefix`.

FLA/causal-conv1d experiments are intentionally kept outside the known-good production environment and should be tested in a separate container.

See [docs/prefix-scheduler-benchmarks.md](docs/prefix-scheduler-benchmarks.md) and [docs/torch-prefix-parity-bisect.md](docs/torch-prefix-parity-bisect.md).
