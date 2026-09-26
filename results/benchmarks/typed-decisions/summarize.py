"""Summarise a typed-decisions run (scripts/evaluate_text_decisions.py worker output): raw and shipped-calibration metrics.

    python3 reports/benchmarks/typed-decisions/summarize.py reports/benchmarks/typed-decisions/imajev-4b-p3/raw-rot4.json 1.3051569717552742

Calibration is applied post hoc exactly as the server does it: softmax(log p / T) over every candidate including the
always-last unknown, which leaves the argmax unchanged. ECE is reported two ways: 15 equal-width bins (our evaluator)
and 10 equal-width bins with max(P) confidence (the definition in Intern-Decision's table and the jevbench harness).
"""
import json, math, sys
from collections import defaultdict

def softmax(logits, t):
    m = max(v / t for v in logits); e = [math.exp(v / t - m) for v in logits]; s = sum(e); return [v / s for v in e]

def ece(conf, ok, bins):
    tot = len(conf); out = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        idx = [i for i, c in enumerate(conf) if (lo < c <= hi) or (b == 0 and c <= hi)]
        if idx: out += len(idx) / tot * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return out

def metrics(rows, t):
    conf, ok, brier = [], [], []
    for r in rows:
        p = softmax(r["logits"], t); g = r["target_index"]
        conf.append(max(p)); ok.append(1.0 if p.index(max(p)) == g else 0.0)
        brier.append(sum((p[i] - (1.0 if i == g else 0.0)) ** 2 for i in range(len(p))))
    n = len(rows)
    return {"n": n, "accuracy": sum(ok) / n, "brier": sum(brier) / n, "ece15": ece(conf, ok, 15), "ece10": ece(conf, ok, 10),
            "mean_confidence": sum(conf) / n, "abstained": sum(1 for r in rows if r["abstained"]) / n}

def main(path, temperature):
    d = json.load(open(path)); rows = d["decisions"]; t = float(temperature)
    out = {"source": path, "configuration": d.get("configuration"), "n_cases": d["n_cases"], "failures": len(d["failures"]),
           "median_ms_per_case": d["latency_ms"]["median_per_case"], "temperature": t,
           "raw": metrics(rows, 1.0), "calibrated": metrics(rows, t), "by_type": {}, "by_workflow": {}}
    for key, name in (("decision_type", "by_type"), ("workflow", "by_workflow")):
        groups = defaultdict(list)
        for r in rows: groups[r[key]].append(r)
        out[name] = {k: {"raw": metrics(v, 1.0), "calibrated": metrics(v, t)} for k, v in sorted(groups.items())}
    summary_path = path.rsplit("/", 1)[0] + "/summary.json"
    json.dump(out, open(summary_path, "w"), indent=1)
    r, c = out["raw"], out["calibrated"]
    print(f"cases {out['n_cases']} decisions {r['n']} failures {out['failures']} median {out['median_ms_per_case']:.0f} ms/case")
    print(f"raw        acc {100*r['accuracy']:.2f}  brier {r['brier']:.3f}  ece15 {r['ece15']:.3f}  ece10 {r['ece10']:.3f}  abstain {100*r['abstained']:.1f}%")
    print(f"calibrated acc {100*c['accuracy']:.2f}  brier {c['brier']:.3f}  ece15 {c['ece15']:.3f}  ece10 {c['ece10']:.3f}  (T={t:.3f})")
    for k, v in out["by_type"].items(): print(f"  {k:7s} n {v['raw']['n']:4d} acc {100*v['raw']['accuracy']:.1f}  brier cal {v['calibrated']['brier']:.3f}")
    for k, v in out["by_workflow"].items(): print(f"  {k:28s} acc {100*v['raw']['accuracy']:.1f}")
    print("wrote", summary_path)

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "1.0")
