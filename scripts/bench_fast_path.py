"""Serving-path check: how far each server's decisions are from a reference server, and how fast each one is.

    # ImajevBench image requests, sent serially to the reference and every server (order rotated per request)
    python scripts/bench_fast_path.py images --ref http://127.0.0.1:8760 --server current=http://127.0.0.1:8765 \
        --server fast=http://127.0.0.1:8766 --records <imajev-bench>/records/records-public.jsonl --root <imajev-bench> --out <dir>

    # JevBench runner outputs (results.jsonl per tier) against the reference run's
    python scripts/bench_fast_path.py jevbench --ref <dir>/jevbench-ref --run current=<dir>/jevbench-current --out <dir>

A serving path keeps quality if it is no further from the reference (a float32 server) than the current bf16 path is:
same answers, and per-decision probability differences of the same size.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def post(url, body):
    request = urllib.request.Request(url + "/v1/systemone", data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read())
    return payload, time.perf_counter() - start


def quantile(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(q * len(values)))] if values else None


def distance(ref, other):
    """(same answer, largest absolute probability difference) between two decoded decisions."""
    same = (ref["status"], ref["value"]) == (other["status"], other["value"])
    return same, max(abs(p - other["probabilities"].get(k, 0.0)) for k, p in ref["probabilities"].items())


def summarize(rows, name):
    diffs = [r[name]["diff"] for r in rows]
    flips = [{"id": r["id"], "ref_top": round(max(r["ref"]["probabilities"].values()), 3)} for r in rows if not r[name]["same"]]
    return {"n": len(rows), "same_answer": len(rows) - len(flips), "flips": flips,
            "prob_diff": {"median": statistics.median(diffs), "p90": quantile(diffs, 0.9), "max": max(diffs)}}


def images(a):
    from imajev_bench.runner import decode_jev, jev_payload
    root = Path(a.root)
    records = [json.loads(line) for line in Path(a.records).read_text().splitlines() if line.strip()]
    records = [r for r in records if r.get("images")][: a.limit or None]
    servers = [("ref", a.ref)] + [tuple(s.split("=", 1)) for s in a.server]
    for _, url in servers:  # one-off costs (first kernels, allocator growth) stay out of the numbers
        for record in records[: a.warmup]:
            post(url, json.dumps(jev_payload(record, root)).encode())
    rows, times = [], {name: [] for name, _ in servers}
    for index, record in enumerate(records):
        body = json.dumps(jev_payload(record, root)).encode()
        row = {"id": record["id"]}
        order = servers[index % len(servers):] + servers[: index % len(servers)]  # spread clock drift over every server
        for name, url in order:
            response, seconds = post(url, body)
            row[name] = {**decode_jev(response, record), "seconds": seconds}
            times[name].append(seconds)
        for name, _ in servers[1:]:
            same, diff = distance(row["ref"], row[name])
            row[name].update(same=same, diff=diff)
        rows.append(row)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("".join(json.dumps(r, default=str) + "\n" for r in rows))
    summary = {"vs_ref": {name: summarize(rows, name) for name, _ in servers[1:]},
               "latency_s": {name: {"p50": quantile(t, 0.5), "p95": quantile(t, 0.95), "mean": statistics.fmean(t)} for name, t in times.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def jevbench(a):
    def load(path):
        return {row["task_id"]: row for row in map(json.loads, Path(path).read_text().splitlines())}
    report = {}
    for tier in ("easy", "original", "hard"):
        ref = load(Path(a.ref) / tier / "results.jsonl")
        for spec in a.run:
            name, path = spec.split("=", 1)
            run = load(Path(path) / tier / "results.jsonl")
            rows = []
            for task, r in ref.items():
                x = run[task]
                diff = max(abs(p - x["probs"].get(k, 0.0)) for k, p in r["probs"].items())
                rows.append({"id": task, "ref": {"probabilities": r["probs"]}, name: {"same": r["predicted"] == x["predicted"], "diff": diff}})
            s = json.loads((Path(path) / tier / "summary.json").read_text())
            report.setdefault(name, {})[tier] = {**summarize(rows, name), "accuracy": s["accuracy"], "ece": s["ece"]["ece"],
                                                 "latency_p50_s": s["latency"]["p50_s"], "latency_p95_s": s["latency"]["p95_s"]}
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "jevbench-vs-ref.json").write_text(json.dumps(report, indent=2) + "\n")
    for name, tiers in report.items():
        for tier, s in tiers.items():
            print(f"{name:10s} {tier:8s} acc {100 * s['accuracy']:5.1f} ece {s['ece']:.3f} flips {len(s['flips'])} "
                  f"diff med {s['prob_diff']['median']:.4f} max {s['prob_diff']['max']:.3f} p50 {1000 * s['latency_p50_s']:.0f} ms")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    im = sub.add_parser("images")
    im.add_argument("--ref", required=True); im.add_argument("--server", action="append", required=True, help="name=url")
    im.add_argument("--records", required=True); im.add_argument("--root", required=True)
    im.add_argument("--limit", type=int, default=0); im.add_argument("--warmup", type=int, default=3); im.add_argument("--out", required=True)
    jb = sub.add_parser("jevbench")
    jb.add_argument("--ref", required=True); jb.add_argument("--run", action="append", required=True, help="name=dir")
    jb.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    images(a) if a.mode == "images" else jevbench(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
