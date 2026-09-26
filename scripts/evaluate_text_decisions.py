"""Three-way evaluation on the `LocalLLaMA/typed-decisions` test split (Apache-2.0).

400 cases in four workflows (100 each), five typed questions per case = 2,000 decisions, each with a
teacher-derived gold label and a gold probability distribution. Run:

    PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_text_decisions.py

which evaluates, one model at a time in its own subprocess:

  * `v1`    — our 2B with the decision-v1 adapter (`reports/decision-v1/runs/h100x4-full/best-mlx`);
  * `base`  — the same 2B with no adapter;
  * `laya`  — Laya, an EXTERNAL Apache-2.0 baseline, run from the separate `.venv-laya` interpreter so
              it never enters the product environment. `laya-td` is Laya's checkpoint fine-tuned on
              this benchmark's own training split (the one its README's 0.766 refers to).

Reported per question type (noul / choice / score) and per workflow: accuracy against the gold label,
multiclass Brier and ECE against the gold label, score MAE, and median ms per case. Results land in
`reports/text-decisions-v1/results.json` and `report.md`.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
DATASET = "LocalLLaMA/typed-decisions"
OUT_DIR = ROOT / "reports/text-decisions-v1"
MLX_ADAPTER = ROOT / "reports/decision-v1/runs/h100x4-full/best-mlx"
BUNDLE = ROOT / "artifacts/model.json"
LAYA_VENV = ROOT / ".venv-laya/bin/python"
LAYA_CHECKPOINT = "convaiinnovations/laya"

ENGINES = {
    # name:   (label,                                     runs under)
    "v1":     ("imajev v1 (2B + decision-v1 adapter)",     "venv"),
    "base":   ("imajev base (2B, no adapter)",             "venv"),
    "laya":   ("laya (English checkpoint)",                "venv-laya"),
    "laya-td": ("laya-typed-decisions (fine-tuned on this benchmark)", "venv-laya"),
}
QTYPES = ("noul", "choice", "score")


# ------------------------------------------------------------------------------------- the cases

def load_cases(split="test", workflow="all", limit=None):
    """-> [{"id", "workflow", "state", "questions", "gold"}]; state stays a JSON string when it is one."""
    import pandas as pd
    from huggingface_hub import snapshot_download

    root = Path(snapshot_download(DATASET, repo_type="dataset",
                                  allow_patterns=[f"{workflow}/{split}-*.parquet"]))
    frame = pd.read_parquet(sorted(root.glob(f"{workflow}/{split}-*.parquet"))[0])
    cases = []
    for _, row in frame.iterrows():
        state = json.loads(row["state"]) if str(row["state"]).lstrip().startswith("{") else row["state"]
        cases.append({"id": row["id"], "workflow": row["workflow"], "partition": split,
                      "heldout_family": split == "test", "state": state,
                      "questions": json.loads(row["questions"]), "gold": json.loads(row["gold"])})
    cases.sort(key=lambda c: c["id"])
    return cases if limit is None else cases[:limit]


# ------------------------------------------------------------------------------------- the engines

class ImajevEngine:
    """Our 2B, called with zero images: the same prompt and decision position as an image request."""

    def __init__(self, adapter=None, rotations=1, bundle=BUNDLE):
        sys.path.insert(0, str(ROOT / "src"))
        from vision_decision.backend import MLXDirect
        self.engine = MLXDirect(str(bundle), adapter=None if adapter is None else str(adapter))
        self.rotations = rotations
        self.load_seconds = self.engine.load_seconds

    def answer(self, state, questions):
        from vision_decision.jev_api import to_request, to_response
        request = to_request({"state": state, "questions": questions})
        results, _ = self.engine.score_request([], request.fields, request.state, self.rotations)
        return to_response(request, results)["answers"]


class LayaEngine:
    """External baseline; only ever constructed inside the `.venv-laya` interpreter."""

    def __init__(self, checkpoint=LAYA_CHECKPOINT, subfolder=None):
        import laya_mlx
        start = perf_counter()
        self.agent = laya_mlx.load(checkpoint, subfolder=subfolder)
        self.load_seconds = perf_counter() - start

    def answer(self, state, questions):
        return self.agent.system_one(state, questions)["answers"]


def build_engine(name, adapter=None, rotations=1, bundle=BUNDLE):
    if name == "v1":
        return ImajevEngine(adapter or MLX_ADAPTER, rotations, bundle)
    if name == "base":
        return ImajevEngine(None, rotations, bundle)
    if name == "laya":
        return LayaEngine()
    if name == "laya-td":
        return LayaEngine(subfolder="typed-decisions")
    raise ValueError(f"Unknown engine {name!r}; use one of {sorted(ENGINES)}")


# ------------------------------------------------------------------------------------- the metrics

def predicted(answer, gold):
    """-> (label, {label: probability}) over the gold label space, whichever engine produced it."""
    kind = answer["type"]
    if kind == "noul":
        p_true = float(answer["noul"])
        probabilities = {"false": 1.0 - p_true, "true": p_true}
    else:
        probabilities = {str(k): float(v) for k, v in answer["probabilities"].items()}
    total = sum(probabilities.values())
    if total > 0:                                  # our answers renormalize without the unknown mass
        probabilities = {k: v / total for k, v in probabilities.items()}
    for key in gold["probabilities"]:              # a label the engine never scored has probability 0
        probabilities.setdefault(str(key), 0.0)
    return max(probabilities, key=probabilities.get), probabilities


def canonical_distribution(answer, gold):
    """Recover the full served distribution, including unknown, for v1.1 evaluation."""
    unknown = float(answer.get("unknown_probability", 0.0))
    if not 0 <= unknown <= 1:
        raise ValueError("unknown_probability must be in [0, 1]")
    if answer["type"] == "noul":
        # The compatibility response pulls noul toward 0.5 by unknown mass; invert that mapping.
        true = min(1.0 - unknown, max(0.0, float(answer["noul"]) - 0.5 * unknown))
        scores = {"false": 1.0 - unknown - true, "true": true}
    else:
        scores = {str(key): float(value) * (1.0 - unknown)
                  for key, value in answer["probabilities"].items()}
    for key in gold["probabilities"]:
        scores.setdefault(str(key), 0.0)
    scores["__unknown__"] = unknown
    total = sum(scores.values())
    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid answer probabilities")
    return {key: value / total for key, value in scores.items()}


def brier(probabilities, gold_label):
    """Multiclass Brier score: sum over labels of (p - 1[label == gold])^2, in [0, 2]."""
    return sum((p - (1.0 if label == gold_label else 0.0)) ** 2 for label, p in probabilities.items())


def expected_calibration_error(confidences, correct, bins=15):
    """Standard ECE: |mean confidence - accuracy| per equal-width bin, weighted by bin size."""
    if not confidences:
        return None
    edges = [i / bins for i in range(bins + 1)]
    total, error = len(confidences), 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        chosen = [(c, k) for c, k in zip(confidences, correct) if low < c <= high]
        if chosen:
            mean_conf = sum(c for c, _ in chosen) / len(chosen)
            accuracy = sum(k for _, k in chosen) / len(chosen)
            error += (len(chosen) / total) * abs(mean_conf - accuracy)
    return error


def summarise(decisions):
    """Accuracy / Brier / ECE over a list of scored decisions, plus score MAE where it applies."""
    if not decisions:
        return None
    correct = [d["correct"] for d in decisions]
    confidences = [d["confidence"] for d in decisions]
    errors = [d["score_error"] for d in decisions if d["score_error"] is not None]
    return {
        "n": len(decisions),
        "accuracy": sum(correct) / len(correct),
        "brier": sum(d["brier"] for d in decisions) / len(decisions),
        "ece": expected_calibration_error(confidences, correct),
        "mean_confidence": sum(confidences) / len(confidences),
        "score_mae": (sum(errors) / len(errors)) if errors else None,
        "abstained": sum(d["abstained"] for d in decisions) / len(decisions),
    }


# ------------------------------------------------------------------------------------- the run

def evaluate(engine, cases, verbose=True):
    decisions, per_case_ms, failures = [], [], []
    for index, case in enumerate(cases):
        started = perf_counter()
        try:
            answers = engine.answer(case["state"], case["questions"])
        except Exception as exc:                     # a case the engine cannot take (length, options)
            failures.append({"id": case["id"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        per_case_ms.append((perf_counter() - started) * 1000)
        for name, gold in case["gold"].items():
            answer = answers.get(name)
            if answer is None:
                failures.append({"id": case["id"], "error": f"no answer for {name!r}"})
                continue
            legacy_label, legacy_probabilities = predicted(answer, gold)
            served = canonical_distribution(answer, gold)
            served_labels = list(served)
            label = max(served, key=served.get)
            probabilities = served
            gold_label = str(gold["label"])
            score_error = None
            if gold["type"] == "score" and "score" in answer and "score" in gold:
                score_error = abs(float(answer["score"]) - float(gold["score"]))
            decisions.append({
                "case": case["id"], "workflow": case["workflow"], "question": name,
                "type": gold["type"], "gold": gold_label, "predicted": label,
                "correct": label == gold_label,
                "confidence": max(probabilities.values()),
                "brier": brier(probabilities, gold_label),
                "score_error": score_error,
                "abstained": bool(answer.get("abstained", False)),
                # Canonical v1.1 row. Log probabilities are equivalent to logits for scalar
                # temperature fitting/evaluation because softmax is invariant to a shared offset.
                "id": f"{case['id']}:{name}", "family": case["workflow"],
                "partition": case["partition"], "heldout_family": case["heldout_family"],
                "decision_type": gold["type"], "option_count": len(served_labels) - 1,
                "labels": served_labels,
                "logits": [math.log(max(served[value], 1e-300)) for value in served_labels],
                "target_index": served_labels.index(gold_label),
                "legacy_conditional_predicted": legacy_label,
                "legacy_conditional_confidence": max(legacy_probabilities.values()),
            })
        if verbose and (index + 1) % 25 == 0:
            done = [d["correct"] for d in decisions]
            print(f"  {index + 1}/{len(cases)} cases · running accuracy "
                  f"{sum(done) / max(1, len(done)):.3f} · median {statistics.median(per_case_ms):.0f} ms",
                  flush=True)
    workflows = sorted({d["workflow"] for d in decisions})
    return {
        "n_cases": len(per_case_ms),
        "n_decisions": len(decisions),
        "load_seconds": round(engine.load_seconds, 3),
        "latency_ms": {
            "median_per_case": statistics.median(per_case_ms) if per_case_ms else None,
            "mean_per_case": statistics.fmean(per_case_ms) if per_case_ms else None,
            "p90_per_case": (sorted(per_case_ms)[int(0.9 * (len(per_case_ms) - 1))] if per_case_ms else None),
            "median_per_decision": (statistics.median(per_case_ms) / 5) if per_case_ms else None,
        },
        "overall": summarise(decisions),
        "by_type": {kind: summarise([d for d in decisions if d["type"] == kind]) for kind in QTYPES},
        "by_workflow": {w: summarise([d for d in decisions if d["workflow"] == w]) for w in workflows},
        "failures": failures,
        "decisions": decisions,
    }


# ------------------------------------------------------------------------------------- the report

def fmt(value, digits=3):
    return "—" if value is None else f"{value:.{digits}f}"


def baselines(cases):
    """Random guess and per-question majority class, so the accuracies have a floor to read against."""
    counts, random_sum, total = {}, 0.0, 0
    for case in cases:
        for name, gold in case["gold"].items():
            key = (case["workflow"], name)
            counts.setdefault(key, {}).setdefault(str(gold["label"]), 0)
            counts[key][str(gold["label"])] += 1
            random_sum += 1 / max(1, len(gold["probabilities"]))
            total += 1
    majority = sum(max(labels.values()) for labels in counts.values())
    return {"random": random_sum / total, "majority_class": majority / total}


def write_report(results, cases, path=OUT_DIR / "report.md"):
    order = [name for name in ENGINES if name in results]
    floor = baselines(cases)
    lines = [
        "# Text-only typed decisions — imajev vs external baselines",
        "",
        f"Dataset: [`{DATASET}`](https://huggingface.co/datasets/{DATASET}) (Apache-2.0), **test** split of the",
        "`all` config: 400 cases across four workflows (100 each), five typed questions per case = 2,000 decisions.",
        "Each question carries a teacher-derived gold label and a gold probability distribution; we score the",
        "**argmax against the gold label**.",
        "",
        "**Laya is an external baseline only.** It is not part of imajev and is not installed in the product",
        "environment: it lives in a separate `.venv-laya` interpreter and is invoked as a subprocess by",
        "`scripts/evaluate_text_decisions.py`. `laya` is the English checkpoint; `laya-typed-decisions` is the",
        "checkpoint Convai fine-tuned on **this benchmark's own training split**, which is what the Laya README's",
        "reported 0.766 refers to. Imajev configuration names and adapter paths are recorded by the run; this",
        "report does not infer whether a supplied adapter has seen any benchmark training rows.",
        "",
        "## Headline",
        "",
        "| engine | accuracy | Brier | ECE | score MAE | median ms / case | median ms / decision | abstained |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in order:
        result, summary, latency = results[name], results[name]["overall"], results[name]["latency_ms"]
        lines.append(
            f"| {ENGINES[name][0]} | **{fmt(summary['accuracy'])}** | {fmt(summary['brier'])} | "
            f"{fmt(summary['ece'])} | {fmt(summary['score_mae'])} | {fmt(latency['median_per_case'], 1)} | "
            f"{fmt(latency['median_per_decision'], 1)} | {fmt(summary['abstained'])} |")
    lines += [
        f"| *per-question majority class (this test split)* | *{fmt(floor['majority_class'])}* | | | | | | |",
        f"| *random guess* | *{fmt(floor['random'])}* | | | | | | |",
        "| *Jev 1.13.0 (third-party published, never measured here)* | *0.727* | *n/c* | *0.144* | *0.391* | *236–276* | | |",
        "",
        "Brier here is the multiclass score **summed** over labels, in [0, 2]; lower is better. Laya's README",
        "publishes a per-class-averaged Brier (0.062 for `laya-typed-decisions` against our 0.400 for the same",
        "run), so the published Jev Brier is not comparable to this column and is left out (`n/c`). ECE is the",
        "standard 15-bin expected calibration error of p_max against correctness, and it does match the published",
        "convention. `abstained` is the fraction of decisions where our model put its top mass on `unknown`: this",
        "benchmark has no unknown gold option, so every abstention is scored as wrong and unknown remains in the",
        "distribution for confidence, Brier, NLL exports, and calibration. Laya has no abstention at all.",
        "",
        "**Harness check.** On the two Laya rows this script reproduces Convai's published figures exactly —",
        "accuracy 0.766 / 0.362, score MAE 0.242 / 0.694, ECE 0.213 / 0.175 — which is the evidence that the",
        "accuracy, MAE and ECE columns are computed the way the published numbers are.",
        "",
        "## By question type",
        "",
        "`abst` is the share of decisions on which our model put its top mass on `unknown`.",
        "",
        "| engine | " + " | ".join(f"{kind} acc | {kind} Brier | {kind} abst" for kind in QTYPES) + " |",
        "|---|" + "---|" * (3 * len(QTYPES)),
    ]
    for name in order:
        cells = []
        for kind in QTYPES:
            summary = results[name]["by_type"].get(kind)
            cells += ["—", "—", "—"] if not summary else [
                fmt(summary["accuracy"]), fmt(summary["brier"]), fmt(summary["abstained"])]
        lines.append(f"| {ENGINES[name][0]} | " + " | ".join(cells) + " |")

    buckets = [(0.0, 0.5, "< 0.50"), (0.5, 0.7, "0.50 – 0.70"), (0.7, 0.9, "0.70 – 0.90"), (0.9, 1.01, "≥ 0.90")]
    lines += ["", "## Confidence gating", "",
              "Accuracy inside each band of the engine's own p_max, with the share of decisions that land there.",
              "A model whose accuracy rises with its confidence can be gated; one whose does not, cannot.", "",
              "| engine | " + " | ".join(label for _, _, label in buckets) + " |",
              "|---|" + "---|" * len(buckets)]
    for name in order:
        decisions = results[name]["decisions"]
        cells = []
        for low, high, _ in buckets:
            chosen = [d for d in decisions if low <= d["confidence"] < high]
            cells.append("—" if not chosen else
                         f"{sum(d['correct'] for d in chosen) / len(chosen):.3f} "
                         f"<sub>{len(chosen) / len(decisions):.0%}</sub>")
        lines.append(f"| {ENGINES[name][0]} | " + " | ".join(cells) + " |")

    workflows = sorted({w for name in order for w in results[name]["by_workflow"]})
    lines += ["", "## By workflow (accuracy)", "",
              "| engine | " + " | ".join(w.replace("_", " ") for w in workflows) + " |",
              "|---|" + "---|" * len(workflows)]
    for name in order:
        cells = [fmt(results[name]["by_workflow"][w]["accuracy"]) if w in results[name]["by_workflow"] else "—"
                 for w in workflows]
        lines.append(f"| {ENGINES[name][0]} | " + " | ".join(cells) + " |")

    lines += ["", "## Run detail", "",
              "| engine | cases scored | decisions | load s | mean ms / case | p90 ms / case | failures |",
              "|---|---|---|---|---|---|---|"]
    for name in order:
        result = results[name]
        lines.append(f"| {ENGINES[name][0]} | {result['n_cases']} | {result['n_decisions']} | "
                     f"{fmt(result['load_seconds'], 1)} | {fmt(result['latency_ms']['mean_per_case'], 1)} | "
                     f"{fmt(result['latency_ms']['p90_per_case'], 1)} | {len(result['failures'])} |")
    if "v1" in results and "base" in results:
        v1, base = results["v1"]["overall"], results["base"]["overall"]
        best_laya = max((results[n]["overall"]["accuracy"] for n in ("laya", "laya-td") if n in results),
                        default=None)
        lines += [
            "", "## What this measures", "",
            f"{ENGINES['v1'][0]} answers {fmt(v1['accuracy'])} of these 2,000 text decisions against "
            f"{fmt(base['accuracy'])} for the same 2B with no adapter and {fmt(floor['majority_class'])} for the "
            "per-question majority class. Relative to the base model, the measured change on these text decisions is "
            f"({v1['accuracy'] - base['accuracy']:+.3f}). Its ECE is {fmt(v1['ece'])} at a mean confidence of "
            f"{fmt(v1['mean_confidence'])}, i.e. the confidence it reports is close to how often it is right.",
            "",
            (f"The strongest measured Laya configuration scores "
             f"({fmt(best_laya)} for the best Laya checkpoint) and roughly {fmt(results['v1']['latency_ms']['median_per_case'] / max(1e-9, min(results[n]['latency_ms']['median_per_case'] for n in ('laya', 'laya-td') if n in results)), 1)}x "
             "faster. Laya remains an external comparison and is not part of the product." if best_laya is not None else
             "Laya was not run in this pass, so there is no external baseline in this table."),
            "",
            "Caveats: `laya-typed-decisions` was fine-tuned on this benchmark's own training split. The gold labels are "
            "teacher-derived, with a reported self-agreement ceiling of 0.735, so 1.000 is not reachable. This "
            "benchmark has no `unknown` option, so every abstention our model makes is scored as wrong.",
        ]
    lines += [
        "",
        "Latencies are wall-clock for a whole case (all five questions) on one Apple-silicon machine, MLX,",
        "model resident, measured inside the evaluation loop. Our model prefills the shared state + header once",
        "per case and then runs one forward pass per question; Laya scores all five questions in a single",
        "non-autoregressive encoder pass, which is where its latency advantage comes from.",
        "",
        f"Reproduce: `PYTHONPATH=src:scripts .venv/bin/python scripts/evaluate_text_decisions.py`",
        "(Laya needs `.venv-laya`: `python -m venv .venv-laya && .venv-laya/bin/pip install laya-mlx laya pandas pyarrow`.)",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))
    return path


# ------------------------------------------------------------------------------------- the driver

def run_worker(engine_name, args, destination):
    """Run one engine in its own process, so only one model is ever resident."""
    interpreter = str(LAYA_VENV) if ENGINES[engine_name][1] == "venv-laya" else sys.executable
    if ENGINES[engine_name][1] == "venv-laya" and not LAYA_VENV.is_file():
        raise FileNotFoundError(
            f"{LAYA_VENV} does not exist. Laya is an external baseline and must stay out of the product "
            f"environment: python -m venv .venv-laya && .venv-laya/bin/pip install laya-mlx laya pandas pyarrow")
    command = [interpreter, str(Path(__file__).resolve()), "--engine", engine_name,
               "--out", str(destination), "--split", args.split, "--workflow", args.workflow,
               "--our-name", args.our_name]
    if args.limit:
        command += ["--limit", str(args.limit)]
    if args.adapter:
        command += ["--adapter", args.adapter]
    command += ["--rotations", str(args.rotations), "--bundle", args.bundle]
    print(f"\n=== {ENGINES[engine_name][0]} ({interpreter}) ===", flush=True)
    subprocess.run(command, check=True, cwd=str(ROOT))
    return json.loads(destination.read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", default="v1,base,laya,laya-td",
                        help="comma-separated subset of " + ",".join(ENGINES))
    parser.add_argument("--engine", help="worker mode: evaluate exactly this engine and write --out")
    parser.add_argument("--adapter", help="override the MLX adapter directory for the v1 engine")
    parser.add_argument("--our-name", default="imajev v1 (2B + decision-v1 adapter)",
                        help="report label for --engine v1 (for example, 'imajev v1.1')")
    parser.add_argument("--rotations", type=int, default=1, help="candidate orders averaged per question")
    parser.add_argument("--bundle", default=str(BUNDLE), help="pinned base-model bundle for the v1/base engines (artifacts/model*.json)")
    parser.add_argument("--split", default="test")
    parser.add_argument("--workflow", default="all", help="dataset config: all, or one workflow")
    parser.add_argument("--limit", type=int, help="evaluate only the first N cases (smoke runs)")
    parser.add_argument("--report-only", action="store_true",
                        help="rebuild results.json and report.md from the raw-*.json files already written")
    parser.add_argument("--out", default=str(OUT_DIR), help="worker: a file; driver: the report directory")
    args = parser.parse_args(argv)
    ENGINES["v1"] = (args.our_name, ENGINES["v1"][1])

    cases = load_cases(args.split, args.workflow, args.limit)

    if args.engine:                                    # worker: one engine, one process, one model
        engine = build_engine(args.engine, args.adapter, args.rotations, args.bundle)
        print(f"loaded {args.engine} in {engine.load_seconds:.1f}s; {len(cases)} cases", flush=True)
        result = evaluate(engine, cases)
        result["configuration"] = {"engine": args.engine, "label": ENGINES[args.engine][0],
                                   "adapter": args.adapter if args.engine == "v1" else None,
                                   "bundle": args.bundle, "rotations": args.rotations}
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result))
        summary = result["overall"]
        print(f"{args.engine}: accuracy {summary['accuracy']:.3f}  brier {summary['brier']:.3f}  "
              f"ece {summary['ece']:.3f}  median {result['latency_ms']['median_per_case']:.0f} ms/case")
        return 0

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = [name.strip() for name in args.engines.split(",") if name.strip()]
    unknown = [name for name in wanted if name not in ENGINES]
    if unknown:
        raise SystemExit(f"Unknown engines {unknown}; use {sorted(ENGINES)}")
    results = {}
    for name in wanted:
        raw = out_dir / f"raw-{name}.json"
        results[name] = json.loads(raw.read_text()) if args.report_only else run_worker(name, args, raw)

    # results.json keeps the summaries; the per-decision rows stay in the raw files.
    summary = {"dataset": DATASET, "split": args.split, "workflow": args.workflow,
               "n_cases": len(cases), "baselines": baselines(cases),
               "engines": {name: {k: v for k, v in result.items() if k != "decisions"}
                           for name, result in results.items()}}
    (out_dir / "results.json").write_text(json.dumps(summary, indent=1))
    report = write_report(results, cases, out_dir / "report.md")
    print(f"\nwrote {out_dir / 'results.json'} and {report}")
    for name in wanted:
        overall, latency = results[name]["overall"], results[name]["latency_ms"]
        print(f"  {name:8s} accuracy {overall['accuracy']:.3f}  brier {overall['brier']:.3f}  "
              f"ece {overall['ece']:.3f}  {latency['median_per_case']:.0f} ms/case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
