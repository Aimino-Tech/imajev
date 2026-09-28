"""Fit one temperature for text-only requests and one for image-present requests, on held-out phase-3 panels, and check both on
ImajevBench public and hidden splits by track. Nothing here changes an argmax.

    PYTHONPATH=src .venv/bin/python scripts/calibration/fit_modality_temperatures.py
"""
import json, math, sys
from pathlib import Path
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
from vision_decision.calibration import fit_temperature, softmax
E = ROOT / "reports/phase3/train-results/p3/run/eval/r2-s000291"; OUT = ROOT / "reports/calibration-modality"; OUT.mkdir(exist_ok=True)
jl = lambda p: [json.loads(l) for l in open(p)]
def image_ids(manifest):
    return {r["id"] for r in jl(ROOT / manifest) if r.get("images")}
def panel(name, image_set=None, image_all=None, keep=None):
    rows = []
    for r in jl(E / "panels" / name / "predictions.jsonl"):
        if keep and not keep(r): continue
        has_img = image_all if image_all is not None else (r["id"] in image_set)
        rows.append({"panel": name, "id": r["id"], "image": has_img, "z": r["logits"], "t": r["target_index"], "type": r["decision_type"]})
    return rows
fresh_img = image_ids("data/manifests/decision-p3-heldout-fresh.jsonl"); human_img = image_ids("data/manifests/decision-p3-human-dev.jsonl")
irr = jl(ROOT / "data/manifests/decision-v1.1-irrelevance-test.jsonl"); irr_src = {r["id"]: (r.get("source"), bool(r.get("images"))) for r in irr}
fit_pool = []
for g in ("heldout_charts", "heldout_docimg", "heldout_inventory", "heldout_safety", "heldout_geometry", "heldout_screens"): fit_pool += panel(g, image_all=True)
fit_pool += panel("state_probe", image_all=True) + panel("pairs_probe", image_all=True)
photo_only = panel("irrelevance", image_set={i for i, (s, im) in irr_src.items() if s in ("abo", "vizwiz") and im}, keep=lambda r: irr_src.get(r["id"], ("", False))[0] in ("abo", "vizwiz"))
mmlu_text = panel("irrelevance", image_all=False, keep=lambda r: r["id"].endswith(":text_only"))
mmlu_irrel = panel("irrelevance", image_all=True, keep=lambda r: r["id"].endswith(":irrelevant_image"))
text_pool = panel("authored_dev", image_all=False) + panel("p2b_test", image_all=False)
human = panel("human_dev", image_set=human_img)
def metrics(rows, T):
    conf, ok, nll = [], [], 0.0
    for r in rows:
        p = softmax(r["z"], T); i = max(range(len(p)), key=p.__getitem__); conf.append(p[i]); ok.append(1.0 if i == r["t"] else 0.0); nll -= math.log(max(p[r["t"]], 1e-12))
    n = len(rows); bins = 10; e = 0.0
    for b in range(bins):
        idx = [i for i, c in enumerate(conf) if (b / bins < c <= (b + 1) / bins) or (b == 0 and c <= 1 / bins)]
        if idx: e += len(idx) / n * abs(sum(ok[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    return {"n": n, "acc": sum(ok) / n, "ece": e, "nll": nll / n}
def fit(rows): return fit_temperature([(r["z"], r["t"]) for r in rows])
SHIPPED = 1.3051569717552742
sets = {"constructed+probes (image, 1,860)": fit_pool, "photo-only ABO+VizWiz (image)": photo_only, "MMLU with unrelated photo (image)": mmlu_irrel,
        "MMLU text-only": mmlu_text, "authored dev + p2b test (text)": text_pool, "human dev, image rows": [r for r in human if r["image"]], "human dev, text rows": [r for r in human if not r["image"]]}
report = {"shipped_T": SHIPPED, "fits": {}, "eval": {}}
for name, rows in sets.items():
    T = fit(rows); report["fits"][name] = {"T": T, "n": len(rows), "raw": metrics(rows, 1.0), "shipped": metrics(rows, SHIPPED), "own_T": metrics(rows, T)}
    print(f"{name:38s} n={len(rows):5d}  T*={T:.3f}   ECE raw {report['fits'][name]['raw']['ece']:.3f} → shipped {report['fits'][name]['shipped']['ece']:.3f} → own {report['fits'][name]['own_T']['ece']:.3f}   NLL raw {report['fits'][name]['raw']['nll']:.3f} shipped {report['fits'][name]['shipped']['nll']:.3f} own {report['fits'][name]['own_T']['nll']:.3f}")
image_fit_rows = fit_pool + photo_only
T_image = fit(image_fit_rows); T_text = SHIPPED
report["proposal"] = {"T_text": T_text, "T_image": T_image, "image_fit_rows": len(image_fit_rows), "image_fit_sets": ["constructed held-out groups", "state/pairs probes", "photo-only ABO+VizWiz controls"]}
print(f"\nproposed T_image = {T_image:.3f} (fit on {len(image_fit_rows)} image rows), T_text = {T_text:.3f} (shipped, authored dev)")
# validation on ImajevBench public + hidden (rot4 raw probabilities → log-probabilities as logits), by track
def bench(pred_path, rec_path):
    recs = {r["id"]: r for r in jl(ROOT / rec_path)}; out = defaultdict(list)
    for p in jl(ROOT / pred_path):
        r = recs.get(p["id"]); 
        if not r: continue
        probs = p["probabilities"]; labels = list(probs); z = [math.log(max(probs[k], 1e-12)) for k in labels]; gold = r["gold"]
        gk = "__unknown__" if gold is None else next((k for k in labels if k != "__unknown__" and str(k).lower() == str(gold).lower()), None)
        if gk is None: continue
        out["image" if r["track"] != "text" else "text"].append({"z": z, "t": labels.index(gk)})
    return out
for label, pp, rp in (("ImajevBench public test", "reports/phase3/train-results/p3/run/eval/r2-s000291/imajevbench/predictions.jsonl", "data/imajev-bench/v2-lite-v1/records-final-v2.jsonl"),
                      ("ImajevBench hidden private-1", "reports/phase3/release-check/private1/predictions.jsonl", "data/imajev-bench/private-1/records-audited.jsonl")):
    by = bench(pp, rp); report["eval"][label] = {}
    for mod, rows in by.items():
        Tm = T_image if mod == "image" else T_text
        report["eval"][label][mod] = {"n": len(rows), "raw": metrics(rows, 1.0), "shipped": metrics(rows, SHIPPED), "modality_T": metrics(rows, Tm), "T_used": Tm}
        m = report["eval"][label][mod]; print(f"{label:30s} {mod:5s} n={m['n']:3d}  ECE raw {m['raw']['ece']:.3f} → shipped {m['shipped']['ece']:.3f} → modality {m['modality_T']['ece']:.3f}   NLL {m['raw']['nll']:.3f} → {m['shipped']['nll']:.3f} → {m['modality_T']['nll']:.3f}")
json.dump(report, open(OUT / "fit-report.json", "w"), indent=1); print("wrote", OUT / "fit-report.json")
