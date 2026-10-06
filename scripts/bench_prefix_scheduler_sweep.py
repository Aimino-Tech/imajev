"""Sweep prefix scheduler knobs without touching kernel/runtime dependencies.

Safety criterion: a candidate must not introduce a new winner flip relative to
the current prefix scheduler baseline (bucket=0, qbatch=0). Canonical serial
parity is still reported, but the known ultra-low-margin Q17 disagreement does
not disqualify otherwise identical scheduler behavior.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch
from PIL import Image

from bench_site_showdown import (
    DEFAULT_IMAGE,
    DEFAULT_SNAP,
    _compile,
    _timed,
    _winner,
    serial_score,
    summarize_samples,
)
from torch_decision import TorchDecision
from torch_prefix_cache import PrefixScorer


def _csv_ints(value):
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def _winner_parity(left, right):
    return sum(
        _winner(a)[0] == _winner(b)[0]
        for a, b in zip(left, right)
    )


def _run_config(
    eng,
    images,
    compiled,
    serial_reference,
    baseline_reference,
    args,
    bucket,
    qbatch,
):
    samples, peaks = [], []
    last_meta = None
    last_result = None
    for _ in range(args.runs):
        scorer = PrefixScorer(
            eng,
            fast=not args.slow,
            microbatch=args.microbatch,
            suffix_bucket_width=bucket,
            question_prefill_batch=qbatch,
        )
        try:
            result, seconds, peak = _timed(
                lambda: scorer.score(images, compiled, args.rotations)
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {"oom": True}
        samples.append(seconds)
        peaks.append(peak)
        last_meta = scorer.metadata
        last_result = result

    stats = summarize_samples(samples)
    return {
        "oom": False,
        "p50": stats["p50"],
        "p95": stats["p95"],
        "peak": max(peaks),
        "serial_parity": _winner_parity(serial_reference, last_result),
        "baseline_parity": (
            len(compiled)
            if baseline_reference is None
            else _winner_parity(baseline_reference, last_result)
        ),
        "meta": last_meta or {},
        "result": last_result,
    }


def _print(label, bucket, qbatch, result, n):
    if result["oom"]:
        print(
            f"{label:8s} bucket={bucket:>3} qbatch={qbatch:>3} OOM",
            flush=True,
        )
        return
    meta = result["meta"]
    print(
        f"{label:8s} bucket={bucket:>3} qbatch={qbatch:>3} "
        f"p50={result['p50']:.3f}s p95={result['p95']:.3f}s "
        f"peak={result['peak']:.2f}GiB "
        f"serial={result['serial_parity']}/{n} "
        f"baseline={result['baseline_parity']}/{n} "
        f"pad={100*float(meta.get('suffix_padding_fraction', 0.0)):.1f}% "
        f"qpf={meta.get('question_prefill_batches')} "
        f"suffix={meta.get('suffix_batches')} calls={meta.get('lm_calls')} "
        f"select={meta.get('cache_select_branches')}",
        flush=True,
    )


def _safe_candidates(rows, n, baseline_serial_parity):
    return [
        (knob, result)
        for knob, result in rows
        if (
            not result["oom"]
            and result["baseline_parity"] == n
            and result["serial_parity"] >= baseline_serial_parity
        )
    ]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=DEFAULT_SNAP)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument("--suffix-buckets", default="0,8,16,24,32")
    parser.add_argument("--prefill-batches", default="0,4,8,12,16,35")
    parser.add_argument(
        "--prefill-suffix-bucket",
        type=int,
        default=-1,
        help="-1 = use fastest baseline-safe suffix bucket",
    )
    parser.add_argument("--slow", action="store_true")
    args = parser.parse_args(argv)

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    eng = TorchDecision(
        args.snapshot,
        device="cuda",
        dtype=torch.bfloat16,
        max_length=args.max_input_tokens,
    )
    image = Image.open(image_path).convert("RGB")
    image.thumbnail((448, 448))
    images = [image]
    compiled = _compile(eng, images)
    n = len(compiled)

    with torch.inference_mode():
        serial_reference, _, _ = _timed(
            lambda: serial_score(
                eng,
                images,
                compiled,
                args.rotations,
                fast=not args.slow,
            )
        )

        print("\nCURRENT PREFIX BASELINE", flush=True)
        baseline = _run_config(
            eng,
            images,
            compiled,
            serial_reference,
            None,
            args,
            0,
            0,
        )
        if baseline["oom"]:
            raise RuntimeError("current prefix baseline OOMed")
        baseline_reference = baseline["result"]
        baseline_serial_parity = baseline["serial_parity"]
        _print("baseline", 0, 0, baseline, n)

        print("\nSUFFIX LENGTH BUCKET SWEEP", flush=True)
        suffix_results = []
        for bucket in _csv_ints(args.suffix_buckets):
            if bucket == 0:
                result = baseline
            else:
                result = _run_config(
                    eng,
                    images,
                    compiled,
                    serial_reference,
                    baseline_reference,
                    args,
                    bucket,
                    0,
                )
            suffix_results.append((bucket, result))
            _print("suffix", bucket, 0, result, n)

        viable = _safe_candidates(
            suffix_results,
            n,
            baseline_serial_parity,
        )
        best_bucket = 0
        if viable:
            best_bucket, best_result = min(
                viable,
                key=lambda item: item[1]["p50"],
            )
            print(
                f"best no-new-flip suffix bucket: {best_bucket} "
                f"({best_result['p50']:.3f}s)",
                flush=True,
            )

        print("\nQUESTION PREFILL BATCH SWEEP", flush=True)
        prefill_bucket = (
            best_bucket
            if args.prefill_suffix_bucket < 0
            else args.prefill_suffix_bucket
        )
        prefill_results = []
        for qbatch in _csv_ints(args.prefill_batches):
            if qbatch == 0 and prefill_bucket == 0:
                result = baseline
            else:
                result = _run_config(
                    eng,
                    images,
                    compiled,
                    serial_reference,
                    baseline_reference,
                    args,
                    prefill_bucket,
                    qbatch,
                )
            prefill_results.append((qbatch, result))
            _print("prefill", prefill_bucket, qbatch, result, n)

        viable = _safe_candidates(
            prefill_results,
            n,
            baseline_serial_parity,
        )
        if viable:
            best_qbatch, best_result = min(
                viable,
                key=lambda item: item[1]["p50"],
            )
            print(
                f"best no-new-flip qprefill batch: {best_qbatch} "
                f"with bucket={prefill_bucket} "
                f"({best_result['p50']:.3f}s)",
                flush=True,
            )


if __name__ == "__main__":
    main()
