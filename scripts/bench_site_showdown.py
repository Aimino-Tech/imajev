"""Production-faithful prefix benchmark on a real website screenshot.

Measures four distinct costs over identical prompts:
  batch       - production no-prefix scheduler (question chunks, then rotation batches)
  raw-prefix  - PrefixScorer.score only; no validation and no margin fallback
  cold-gate   - fresh PrefixScorer; includes one-time parity validation + safety fallback
  warm-gate   - already-validated PrefixScorer + production batched margin fallback

Canonical serial scoring is measured separately for decision-parity only.  All
GPU wallclock timings synchronize CUDA before/after the measured region.
"""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import torch
from PIL import Image
from torch_decision import GRAPH_LENGTHS, TorchDecision
from torch_prefix_cache import PrefixScorer
from vision_decision.contracts import BooleanField, ChoiceField
from vision_decision.scoring import (
    combine_rotations,
    compile_question,
    cyclic_offsets,
    result_from_logits,
    rotate,
)

DEFAULT_SNAP = os.environ.get(
    "IMAJEV_QWEN_SNAPSHOT",
    "/mnt/work/repos/Operation/auto/imajev-src/.cache/huggingface/hub/"
    "models--Qwen--Qwen3.5-4B/snapshots/"
    "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
)
DEFAULT_IMAGE = os.environ.get("IMAJEV_BENCH_IMAGE", "/tmp/wiki_main.png")

def fields():
    B = lambda i, q, y="yes", n="no": BooleanField(
        type="boolean", id=f"b{i}", question=q, yes_description=y, no_description=n)
    C = lambda i, q, opts: ChoiceField(
        type="choice", id=f"c{i}", question=q,
        options=[{"value": v, "description": d} for v, d in opts])
    F = [
        B(0, "Does the page show a featured article section?", "featured article shown", "no featured article"),
        B(1, "Is there an 'In the news' section?", "news section shown", "no news section"),
        B(2, "Is a search box visible at the top?", "search box shown", "no search box"),
        B(3, "Does the page offer a dark appearance option?", "dark option shown", "no dark option"),
        C(4, "What sport is the featured article about?",
          [("cricket", "bat-and-ball game"), ("football", "soccer game"), ("tennis", "racket game"), ("rugby", "oval-ball game")]),
        C(5, "Who won the World Rally Championship per the news?",
          [("Elfyn Evans", "rally driver"), ("Scott Martin", "co-driver"), ("Charles Lennox", "nobleman"), ("John Bröcheler", "other")]),
        C(6, "Which flight had an incident over Saudi Arabia?",
          [("Flydubai 1073", "mentioned flight"), ("Flight 404", "other flight"), ("Flight 茨城 1", "other flight"), ("Flight 900", "other flight")]),
        C(7, "What is celebrated on October 6 per 'On this day'?",
          [("German-American Day", "heritage day"), ("Independence Day", "other day"), ("Cricket Day", "other day"), ("News Day", "other day")]),
    ]
    extra_b = [
        "Is a login link visible?", "Is a donate link visible?", "Is there an 'On this day' section?",
        "Is a portrait image shown?", "Is a rally car photo shown?", "Is a Talk tab visible?",
        "Is a View history tab visible?", "Is an appearance panel shown?",
        "Does the featured article mention Sussex?", "Does the news mention Spain?",
        "Does the news mention rugby league?", "Are recent deaths listed?",
    ]
    for i, q in enumerate(extra_b):
        F.append(B(10 + i, q))
    extra_c = [
        ("Which duke is pictured?", [("Richmond", "2nd Duke"), ("Wellington", "other duke"), ("York", "other duke"), ("Kent", "other duke")]),
        ("Which year is the earliest cricket reference?", [("1597", "16th century"), ("1697", "17th century"), ("1725", "18th century"), ("1598", "other year")]),
        ("Which war is listed as ongoing?", [("Russo-Ukrainian war", "listed war"), ("Iran war", "listed war"), ("Gulf war", "other war"), ("Balkan war", "other war")]),
        ("Who defeated Warrington Wolves?", [("Wakefield Trinity", "winning team"), ("Wigan Warriors", "other team"), ("Leeds Rhinos", "other team"), ("St Helens", "other team")]),
        ("What text size is selected?", [("Standard", "selected size"), ("Small", "other size"), ("Large", "other size"), ("Wide", "other size")]),
        ("What color mode is selected?", [("Light", "selected mode"), ("Dark", "other mode"), ("Automatic", "other mode"), ("Wide", "other mode")]),
        ("What width is selected?", [("Standard", "selected width"), ("Wide", "other width"), ("Small", "other width"), ("Large", "other width")]),
        ("How many articles are reported?", [("7248657", "reported count"), ("268352", "editor count"), ("1000000", "other count"), ("5000000", "other count")]),
    ]
    for i, (q, opts) in enumerate(extra_c):
        F.append(C(30 + i, q, opts))
    F.append(B(40, "Is a 'Nominate an article' link shown?"))
    F.append(B(41, "Is the welcome banner centered?"))
    F.append(B(42, "Are article counts reported under the welcome?"))
    F.append(C(43, "Which tab is leftmost?", [("Main Page", "first tab"), ("Talk", "second tab"), ("Read", "other tab"), ("History", "other tab")]))
    F.append(C(44, "Who is the pictured nobleman?", [("Charles Lennox", "named person"), ("Elfyn Evans", "other person"), ("Scott Martin", "other person"), ("John Bröcheler", "other person")]))
    F.append(B(45, "Is a portrait caption shown?"))
    F.append(B(46, "Is a rally photo caption shown?"))
    return F


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed(call):
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    value = call()
    _sync()
    seconds = time.perf_counter() - started
    peak = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
        if torch.cuda.is_available()
        else 0.0
    )
    return value, seconds, peak


def _timed_seconds(call):
    """Nested timing that does not reset the outer peak-memory counter."""
    _sync()
    started = time.perf_counter()
    value = call()
    _sync()
    return value, time.perf_counter() - started


def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(values):
    """Stable benchmark summary used by both CLI output and unit tests."""
    values = [float(value) for value in values]
    if not values:
        raise ValueError("at least one sample is required")
    return {
        "runs": len(values),
        "p50": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "min": min(values),
        "max": max(values),
    }


def serial_score(eng, images, compiled, rotations, *, fast):
    """Canonical B1-per-presentation reference for decision parity."""
    out = []
    for _, header, choices, texts, labels in compiled:
        per = []
        for offset in cyclic_offsets(len(choices), rotations):
            prompt = header + "\n".join(
                f"{label}: {text}"
                for label, text in zip(labels, rotate(texts, offset))
            )
            if fast:
                _, inputs, token_ids = eng.prepare_fast(images, prompt, labels)
                logits = eng.candidate_logits_fast(inputs, token_ids)
            else:
                _, inputs, token_ids = eng.prepare(images, prompt, labels)
                logits = eng.candidate_logits(inputs, token_ids)
            per.append((offset, [float(value) for value in logits.cpu().tolist()]))
        out.append(
            result_from_logits(choices, per[0][1])
            if len(per) == 1
            else combine_rotations(choices, per)
        )
    return out


def _production_batch_chunk(eng, images, compiled, rotations, *, fast):
    """Mirror TorchBackend._score_serial_batch exactly for one question chunk."""
    max_rots = max(
        len(cyclic_offsets(len(choices), rotations))
        for _, _, choices, _, _ in compiled
    )
    per_q = [[] for _ in compiled]
    for rotation_index in range(max_rots):
        examples = []
        owners = []
        for qi, (_, header, choices, texts, labels) in enumerate(compiled):
            offsets = cyclic_offsets(len(choices), rotations)
            if rotation_index >= len(offsets):
                continue
            offset = offsets[rotation_index]
            prompt = header + "\n".join(
                f"{label}: {text}"
                for label, text in zip(labels, rotate(texts, offset))
            )
            rendered, imgs, token_ids = eng.render_example(images, prompt, labels)
            examples.append((rendered, imgs, token_ids, None))
            owners.append((qi, offset))

        if fast:
            inputs, tids, _ = eng.collate_fast(examples)
            batch_logits = eng.candidate_logits_batch_fast(inputs, tids)
        else:
            inputs, tids, _ = eng.collate(examples)
            batch_logits = eng.candidate_logits_batch(inputs, tids)

        for (qi, offset), logits in zip(owners, batch_logits):
            per_q[qi].append(
                (offset, [float(value) for value in logits.cpu().tolist()])
            )

    results = []
    for qi, (_, _, choices, _, _) in enumerate(compiled):
        passes = per_q[qi]
        results.append(
            result_from_logits(choices, passes[0][1])
            if len(passes) == 1
            else combine_rotations(choices, passes)
        )
    return results


def production_batch_score(
    eng,
    images,
    compiled,
    rotations,
    *,
    microbatch,
    fast,
):
    """Mirror TorchBackend._score_serial_bounded + _score_serial_batch."""
    max_rots = max(
        len(cyclic_offsets(len(choices), rotations))
        for _, _, choices, _, _ in compiled
    )
    # VRAM scales with questions x rotations per forward: keep the physical
    # batch at microbatch rows regardless of rotation count.
    chunk = max(1, microbatch // max(1, max_rots))
    out = []
    for start in range(0, len(compiled), chunk):
        out.extend(
            _production_batch_chunk(
                eng,
                images,
                compiled[start : start + chunk],
                rotations,
                fast=fast,
            )
        )
    return out


class ProductionFallback:
    """Timed production-shaped fallback for PrefixScorer margin rescoring."""

    def __init__(self, eng, rotations, microbatch, fast):
        self.eng = eng
        self.rotations = rotations
        self.microbatch = microbatch
        self.fast = fast
        self.reset()

    def reset(self):
        self.calls = []
        self.seconds = 0.0

    def __call__(self, images, compiled):
        self.calls.append(len(compiled))
        value, seconds = _timed_seconds(
            lambda: production_batch_score(
                self.eng,
                images,
                compiled,
                self.rotations,
                microbatch=self.microbatch,
                fast=self.fast,
            )
        )
        self.seconds += seconds
        return value

    @property
    def questions(self):
        return sum(self.calls)


def _winner(result):
    ordered = sorted(result.scores.items(), key=lambda item: -item[1])
    margin = (
        ordered[0][1] - ordered[1][1]
        if len(ordered) > 1
        else float("inf")
    )
    return ordered[0][0], margin


def _parity(reference, candidate):
    flips = []
    for i, (left, right) in enumerate(zip(reference, candidate)):
        lw, lm = _winner(left)
        rw, rm = _winner(right)
        if lw != rw:
            flips.append((i, lw, lm, rw, rm))
    for i, lw, lm, rw, rm in flips:
        print(f"  FLIP Q{i}: serial={lw}@{lm:.3f} candidate={rw}@{rm:.3f}", flush=True)
    return len(reference) - len(flips)


def _compile(eng, images):
    state = {"page": "Wikipedia Main Page", "date": "2026-10-06"}
    compiled = []
    for field in fields():
        header_text, choices, texts = compile_question(field, state)
        compiled.append(
            (
                field,
                header_text,
                choices,
                texts,
                eng.labels(len(choices), len(images)),
            )
        )
    return compiled


def _print_summary(name, seconds, peaks, fallback_questions=None, fallback_seconds=None):
    stats = summarize_samples(seconds)
    peak = max(peaks) if peaks else 0.0
    extra = ""
    if fallback_questions is not None:
        extra += f" fallback_q_p50={statistics.median(fallback_questions):.0f}"
    if fallback_seconds is not None:
        extra += f" fallback_ms_p50={1000 * statistics.median(fallback_seconds):.1f}"
    print(
        f"{name:11s} p50={stats['p50']:.3f}s p95={stats['p95']:.3f}s "
        f"min={stats['min']:.3f}s max={stats['max']:.3f}s "
        f"peak={peak:.2f}GiB{extra}",
        flush=True,
    )
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", default=DEFAULT_SNAP)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--serial-runs", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--rotations", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--suffix-bucket-width", type=int, default=16)
    parser.add_argument("--question-prefill-batch", type=int, default=0)
    parser.add_argument("--max-input-tokens", type=int, default=4096)
    parser.add_argument(
        "--no-graphs",
        action="store_true",
        help="disable CUDA graph capture even on the fast path",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="benchmark the non-fast serving path; fast=True is the default",
    )
    args = parser.parse_args(argv)
    if args.runs < 1 or args.serial_runs < 1 or args.warmups < 0:
        parser.error("runs must be positive and warmups non-negative")

    fast = not args.slow
    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(
            f"benchmark image not found: {image_path}; set --image or IMAJEV_BENCH_IMAGE"
        )

    eng = TorchDecision(
        args.snapshot,
        device="cuda",
        dtype=torch.bfloat16,
        max_length=args.max_input_tokens,
    )
    print(f"[vram] after load: {torch.cuda.memory_allocated() / 1e9:.2f}GB", flush=True)
    if fast and not args.no_graphs:
        # Same 5 short lengths as the serving fast bench: 16 capture lengths
        # reserve ~4 GB and OOM the warmup batch on 16 GiB GPUs.
        lengths = [256, 384, 512, 768, 1024]
        try:
            graph_lengths = eng.capture_graphs(lengths).lengths
            print(f"[vram] after graph capture: "
                  f"{torch.cuda.memory_allocated() / 1e9:.2f}GB", flush=True)
        except Exception as exc:
            # Same serving contract as TorchBackend: eager path is the fallback.
            print(
                f"CUDA graph capture failed; benchmarking eager fast path: {exc}",
                flush=True,
            )
            eng.graphs = None

    image = Image.open(image_path).convert("RGB")
    image.thumbnail((448, 448))
    images = [image]
    compiled = _compile(eng, images)

    batch_call = lambda: production_batch_score(
        eng,
        images,
        compiled,
        args.rotations,
        microbatch=args.microbatch,
        fast=fast,
    )

    # No autograd anywhere: without inference_mode every forward keeps its
    # graph alive and OOMs the run after a few dozen rotations.
    with torch.inference_mode():
        _measure(eng, images, compiled, args, fast, graph_lengths, batch_call)


def _measure(eng, images, compiled, args, fast, graph_lengths, batch_call):
    """Warmup + timing race, all under the caller's inference_mode."""
    # Warm representative full shapes before measurement.  No result cache is
    # involved anywhere in this benchmark.
    for _ in range(args.warmups):
        batch_call()
        PrefixScorer(
            eng,
            fast=fast,
            microbatch=args.microbatch,
            suffix_bucket_width=args.suffix_bucket_width,
            question_prefill_batch=args.question_prefill_batch,
        ).score(images, compiled, args.rotations)

    # Canonical reference is deliberately separate from the main timing race.
    serial_samples, serial_peaks = [], []
    serial_result = None
    for _ in range(args.serial_runs):
        serial_result, seconds, peak = _timed(
            lambda: serial_score(
                eng,
                images,
                compiled,
                args.rotations,
                fast=fast,
            )
        )
        serial_samples.append(seconds)
        serial_peaks.append(peak)

    raw_scorer = PrefixScorer(
        eng,
        fast=fast,
        microbatch=args.microbatch,
        suffix_bucket_width=args.suffix_bucket_width,
        question_prefill_batch=args.question_prefill_batch,
    )

    # Validate a persistent scorer OUTSIDE the warm-production timer.  Two
    # same-type boolean questions are enough to exercise the real visual gate.
    warm_scorer = PrefixScorer(
        eng,
        fast=fast,
        microbatch=args.microbatch,
        suffix_bucket_width=args.suffix_bucket_width,
        question_prefill_batch=args.question_prefill_batch,
    )
    validation_fallback = ProductionFallback(
        eng, args.rotations, args.microbatch, fast
    )
    warm_scorer.maybe_validate_and_score(
        images,
        compiled[:2],
        args.rotations,
        validation_fallback,
    )
    if not warm_scorer.validated_visual:
        raise RuntimeError(
            f"visual prefix gate did not validate: {warm_scorer.error}"
        )

    samples = {
        name: {"seconds": [], "peaks": [], "fallback_q": [], "fallback_s": []}
        for name in ("batch", "raw-prefix", "cold-gate", "warm-gate")
    }
    last_results = {}

    # Alternate order to reduce thermal/allocator ordering bias.
    base_order = ["batch", "raw-prefix", "cold-gate", "warm-gate"]
    for run in range(args.runs):
        order = base_order if run % 2 == 0 else list(reversed(base_order))
        for name in order:
            fallback = None
            if name == "batch":
                call = batch_call
            elif name == "raw-prefix":
                call = lambda: raw_scorer.score(
                    images, compiled, args.rotations
                )
            elif name == "cold-gate":
                cold_scorer = PrefixScorer(
                    eng,
                    fast=fast,
                    microbatch=args.microbatch,
                    suffix_bucket_width=args.suffix_bucket_width,
                    question_prefill_batch=args.question_prefill_batch,
                )
                fallback = ProductionFallback(
                    eng, args.rotations, args.microbatch, fast
                )
                call = lambda scorer=cold_scorer, fb=fallback: (
                    scorer.maybe_validate_and_score(
                        images, compiled, args.rotations, fb
                    )
                )
            else:
                fallback = ProductionFallback(
                    eng, args.rotations, args.microbatch, fast
                )
                call = lambda fb=fallback: warm_scorer.maybe_validate_and_score(
                    images, compiled, args.rotations, fb
                )

            value, seconds, peak = _timed(call)
            last_results[name] = value
            bucket = samples[name]
            bucket["seconds"].append(seconds)
            bucket["peaks"].append(peak)
            if fallback is not None:
                bucket["fallback_q"].append(fallback.questions)
                bucket["fallback_s"].append(fallback.seconds)

    print(
        f"\nproduction benchmark: n={len(compiled)} rotations={args.rotations} "
        f"microbatch={args.microbatch} fast={fast} runs={args.runs} "
        f"suffix_bucket={args.suffix_bucket_width} "
        f"question_prefill_batch={args.question_prefill_batch} "
        f"cuda_graphs={graph_lengths}\n",
        flush=True,
    )
    serial_stats = _print_summary("serial", serial_samples, serial_peaks)
    summaries = {}
    for name in base_order:
        bucket = samples[name]
        summaries[name] = _print_summary(
            name,
            bucket["seconds"],
            bucket["peaks"],
            bucket["fallback_q"] or None,
            bucket["fallback_s"] or None,
        )

    batch_p50 = summaries["batch"]["p50"]
    print("\nproduction speedups (p50):", flush=True)
    for name in ("raw-prefix", "cold-gate", "warm-gate"):
        value = summaries[name]["p50"]
        print(
            f"  {name:11s}: {batch_p50 / value:.2f}x vs batch, "
            f"{serial_stats['p50'] / value:.2f}x vs serial",
            flush=True,
        )

    print("\ndecision parity vs canonical serial:", flush=True)
    for name in base_order:
        print(
            f"  {name:11s}: {_parity(serial_result, last_results[name])}/"
            f"{len(compiled)}",
            flush=True,
        )

    print("\nwarm prefix metadata:", warm_scorer.metadata, flush=True)


if __name__ == "__main__":
    main()
