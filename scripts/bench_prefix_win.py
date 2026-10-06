"""Decision-parity + wallclock: 4x rotations over 4 choices, serial vs prefix. Raw-logit level."""
import sys, time, math
sys.path.insert(0, "/mnt/work/repos/imajev/scripts")
sys.path.insert(0, "/mnt/work/repos/imajev/src")
import torch
from torch_decision import TorchDecision
import torch_prefix_cache as pc

SNAP = "/mnt/work/repos/Operation/auto/imajev-src/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
eng = TorchDecision(SNAP, device="cuda", dtype=torch.bfloat16)

rec = {"name": "Ada", "age": 36, "city": "Lagos", "member": True, "visits": 12}
pad = "Consider the record in full detail. " * 60
HQ = f"Answer from the record {rec}.\n{pad}Question: "
QUESTIONS = [
    ("Which city is named?", [("A", "Lagos."), ("B", "Abuja."), ("C", "Kano."), ("D", "Ibadan.")]),
    ("How old is Ada?", [("A", "36 years."), ("B", "29 years."), ("C", "41 years."), ("D", "22 years.")]),
    ("Member status?", [("A", "She is a member."), ("B", "She is not a member."), ("C", "Membership pending."), ("D", "Membership revoked.")]),
    ("Visit count?", [("A", "12 visits."), ("B", "3 visits."), ("C", "7 visits."), ("D", "20 visits.")]),
    ("Ada member since?", [("A", "2021."), ("B", "2020."), ("C", "2022."), ("D", "2019.")]),
    ("Ada age next year?", [("A", "37."), ("B", "36."), ("C", "38."), ("D", "35.")]),
    ("Lagos country?", [("A", "Nigeria."), ("B", "Niger."), ("C", "Ghana."), ("D", "Benin.")]),
    ("Visits doubled?", [("A", "24."), ("B", "22."), ("C", "26."), ("D", "12.")]),
    ("Name length?", [("A", "3 letters."), ("B", "4 letters."), ("C", "2 letters."), ("D", "5 letters.")]),
    ("Member for years?", [("A", "5 years."), ("B", "4 years."), ("C", "6 years."), ("D", "3 years.")]),
    ("City size rank?", [("A", "Largest."), ("B", "Second."), ("C", "Third."), ("D", "Fourth.")]),
    ("Age even?", [("A", "Yes, even."), ("B", "No, odd."), ("C", "Prime."), ("D", "Unknown.")]),
]
OFFS = (0, 1, 2, 3)

def agg(logit_lists, n):
    # mean log-prob per candidate index (rotation i has candidate (i+off)%n at position i -> invert)
    import math as m
    totals = [0.0] * n
    for off, lg in logit_lists:
        top = max(lg); norm = top + m.log(sum(m.exp(x - top) for x in lg))
        for pos, v in enumerate(lg):
            totals[(pos + off) % n] += v - norm
    avg = [t / len(logit_lists) for t in totals]
    exps = [m.exp(x - max(avg)) for x in avg]
    return [e / sum(exps) for e in exps]

with torch.inference_mode():
    groups, offs = [], []
    for q, ch in QUESTIONS:
        rots = []
        for off in OFFS:
            order = ch[off:] + ch[:off]
            prompt = HQ + q + "\n" + "\n".join(f"{l}: {t}" for l, t in order)
            rots.append((eng.render(prompt, 0), [l for l, _ in order]))
            offs.append(off)
        groups.append(rots)

    t0 = time.perf_counter()
    serial = []
    for rots in groups:
        for (r, labs) in rots:
            inputs, tids, _ = eng.collate([(r, [], eng.label_ids(r, labs), 0)])
            serial.append(eng.candidate_logits_batch(inputs, tids)[0].tolist())
    t_serial = time.perf_counter() - t0

    t0 = time.perf_counter()
    plogits, meta = pc.score_rendered_prefix_cached_hierarchical(eng, None, groups)
    t_prefix = time.perf_counter() - t0
    pre = [t.tolist() for t in plogits]

    print(f"serial={t_serial:.2f}s prefix={t_prefix:.2f}s speedup={t_serial/t_prefix:.2f}x meta=" +
          str({k: v for k, v in meta.items() if k != 'suffix_tokens'}), flush=True)
    agree = dmax = 0
    for qi in range(len(QUESTIONS)):
        s = agg([(offs[qi * 4 + j], serial[qi * 4 + j]) for j in range(4)], 4)
        p = agg([(offs[qi * 4 + j], pre[qi * 4 + j]) for j in range(4)], 4)
        dmax = max(dmax, max(abs(a - b) for a, b in zip(s, p)))
        sv = sorted(range(4), key=lambda i: -s[i]); pv = sorted(range(4), key=lambda i: -p[i])
        ok = sv[0] == pv[0]
        agree += ok
        print(f"Q{qi}: serial=c{sv[0]}@{s[sv[0]]-s[sv[1]]:.3f} prefix=c{pv[0]}@{p[pv[0]]-p[pv[1]]:.3f} agree={ok}", flush=True)
    print(f"decision-parity: {agree}/{len(QUESTIONS)} maxprobdelta={dmax:.4f}", flush=True)
