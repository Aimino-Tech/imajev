# imajev-4b

The full model card lives on Hugging Face: https://huggingface.co/mohit67890/imajev-4b

Official boards: **#1 of 49 on Image JevBench v0.1.3** (released 28 Sep 2026; composite 76.39; https://benchmarkheaven.com/image-jev-bench; fast server `--fast --merge-lora`), **#1 of 91 on JevBench v1.4.2.2** (scored 27 Sep 2026; JevBench Score 67.37; https://benchmarkheaven.com/jev-models; adapter revision `c9e5f132`, served with
`--rotations 1 --calibration calibration.json`) and **#3 of 56 models on DecisionBench (eng, v1)** (79.65; https://huggingface.co/spaces/Hanno-Labs/decision-bench-leaderboard; results PR Hanno-Labs/decision-bench-results#68, merged 28 Sep 2026).

Summary (phase-3 version, 2026-09-26): Qwen3.5-4B + LoRA (r64/α128, language layers) + 256-code decision readout (255 options + unknown);
the recommended default (about 0.5 s per decision as shipped with four option orders on one H100 under load; runs on a Mac via MLX).
ImajevBench v2.0-lite test 83.9% (234/279), hidden split 85.6%; JevBench public hard 72.1% (our runs, as shipped); DecisionBench 1.0 full
suite 79.7% with the benchmark's own harness (every row scored; previous version 77.5%); fastino/fast-decisions dev split 60.4% domain
macro (their board uses a held-out test split); typed S1-Bench conversion 99.1% (212 items, saturated, not an S1-Bench score). Trained on top of the previous release with a round
of decisions that release got wrong (58k hard text, 32k hard image incl. 20k constructed chart / document / shelf / safety / geometry /
screen decisions, labelled by open-weight teachers and reviewed) plus a second round on the residual failures. Results: `results/phase3/`,
`results/benchmarks/`.
