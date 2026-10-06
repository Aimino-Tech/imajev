# Contributing

Thanks for helping improve imajev. This repository is a community performance fork of [mohit67890/imajev](https://github.com/mohit67890/imajev). Upstream authorship, licence and citation are preserved.

## Development

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev,serve]"
pytest -q tests/test_contracts.py tests/test_temperature_calibration.py tests/test_playground_server.py
```

Open an issue before changes to public contracts, model semantics, benchmark methodology, defaults, dependencies or repository structure.

Performance PRs must include commit SHA, hardware, runtime versions, model/adapter, precision, rotations, microbatch, scheduler knobs, p50/p95, peak memory, winner parity and the exact command used.

Do not commit weights, datasets, private photos, prediction dumps, credentials or customer data. Public evaluation numbers must name model, adapter, calibration, interface, rotations, hardware and date.

Prefer upstream-compatible changes and isolate fork-specific serving logic. Intentional divergence belongs in [FORK.md](FORK.md) with regression coverage.

Contributions are accepted under Apache-2.0.
