# Changelog

All notable fork-specific changes are documented here. Upstream model/training
changes belong to [mohit67890/imajev](https://github.com/mohit67890/imajev).

## 2026-10-07 — Community performance fork

### Added

- exact in-memory and optional persistent result caching;
- one-time multimodal visual preprocessing per shared request;
- hierarchical shared-prefix/KV reuse across global evidence, questions and
  option-order rotations;
- cross-question batching with bounded VRAM;
- length-aware suffix packing with an 8-token production bucket;
- prefix winner-parity validation and explicit opt-out via
  `--no-shared-prefix`;
- production and scheduler benchmarks with p50/p95, peak VRAM, LM-call counts,
  padding waste and winner parity;
- community documentation, provenance/NOTICE, contribution guidance, security
  policy, issue/PR templates and lightweight Python 3.11/3.12 CI.

### Performance

On the documented 35-question × 4-rotation visual showdown, the promoted
configuration `microbatch=16 / suffix bucket=8 / question-prefill batch=0`
measured **4.39–4.46 s p50** for the warm shared-prefix path, versus roughly
**14.1–14.6 s** for production batching and **17.0–17.8 s** for canonical
serial scoring.

The latest repeat was 4.39 s: **3.21× faster than batch and 3.88× faster than
serial**. Winner parity was 34/35 against canonical serial; the remaining
disagreement is the documented ultra-low-margin Q17 case.

See [FORK.md](FORK.md) and
[docs/torch-prefix-parity-bisect.md](docs/torch-prefix-parity-bisect.md) for
methodology, limitations and rejected experiments.

### Compatibility

- Python 3.11 and 3.12 core CI passes.
- A pre-existing Python 3.11-incompatible nested f-string in
  `scripts/p2/p2_common.py` was rewritten without changing its output.
- Experimental FLA/causal-conv1d kernels are intentionally not part of the
  production dependency stack.
- Wider question-prefix batches remain experimental because they introduced new
  winner flips in the current validation panel.

### Provenance

This fork preserves the upstream Apache-2.0 licence, authorship, canonical
citation, model cards, weights and training/evaluation lineage. Fork-specific
serving and community changes are documented in [FORK.md](FORK.md) and Git
history.
