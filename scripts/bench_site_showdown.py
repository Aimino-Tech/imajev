"""Website showdown: real Wikipedia screenshot, ~35 grounded questions x4 rotations.

Three paths, same prompts:
  serial  - every rotation a full forward (one big microbatch)
  gate    - PrefixScorer.maybe_validate_and_score (argmax validation + per-question margin fallback)
  batch   - serial content but microbatched like production (question_microbatch=8)

Reports parity vs serial, fallback rate, wallclock speedups.
"""
import sys, time
sys.path.insert(0, "/mnt/work/repos/imajev/scripts")
sys.path.insert(0, "/mnt/work/repos/imajev/src")
import torch
from PIL import Image
from torch_decision import TorchDecision
from torch_prefix_cache import PrefixScorer
from vision_decision.contracts import BooleanField, ChoiceField
from vision_decision.scoring import combine_rotations, compile_question, cyclic_offsets, rotate

SNAP = "/mnt/work/repos/Operation/auto/imajev-src/.cache/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
ROT, MICRO = 4, 16


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


def serial_score(eng, images, compiled, rotations):
    out = []
    for field, header, choices, texts, labels in compiled:
        per = []
        for off in cyclic_offsets(len(choices), rotations):
            prompt = header + "\n".join(
                f"{label}: {text}" for label, text in zip(labels, rotate(texts, off)))
            rendered = eng.render(prompt, len(images))
            inputs = eng.processor(text=[rendered], images=images, return_tensors="pt")
            lg = eng.candidate_logits_batch(inputs, [eng.label_ids(rendered, labels)])[0]
            per.append((off, lg.tolist()))
        out.append(combine_rotations(choices, per))
    return out


def batched_score(eng, images, compiled, rotations, micro):
    # same content as serial, but forward in microbatches (production shape)
    rows = []
    owners = []
    for qi, (field, header, choices, texts, labels) in enumerate(compiled):
        for off in cyclic_offsets(len(choices), rotations):
            prompt = header + "\n".join(
                f"{label}: {text}" for label, text in zip(labels, rotate(texts, off)))
            rendered = eng.render(prompt, len(images))
            rows.append((rendered, images, eng.label_ids(rendered, labels), 0))
            owners.append((qi, off))
    per_q = {}
    for s in range(0, len(rows), micro):
        chunk = rows[s:s + micro]
        inputs, tids, _ = eng.collate([(r, im, t, 0) for r, im, t, _ in chunk])
        for k, lg in enumerate(eng.candidate_logits_batch(inputs, tids)):
            qi, off = owners[s + k]
            per_q.setdefault(qi, []).append((off, lg.tolist()))
    return [combine_rotations(compiled[qi][2], per_q[qi]) for qi in range(len(compiled))]


def main():
    eng = TorchDecision(SNAP, device="cuda", dtype=torch.bfloat16)
    img = Image.open("/tmp/wiki_main.png").convert("RGB")
    img.thumbnail((448, 448))
    images = [img]
    state = {"page": "Wikipedia Main Page", "date": "2026-10-06"}
    compiled = []
    for f in fields():
        header, choices, texts = compile_question(f, state)
        compiled.append((f, header, choices, texts, eng.labels(len(choices), len(images))))

    with torch.inference_mode():
        t0 = time.perf_counter()
        serial = serial_score(eng, images, compiled, ROT)
        t_serial = time.perf_counter() - t0
        torch.cuda.empty_cache()

        t0 = time.perf_counter()
        batched = batched_score(eng, images, compiled, ROT, MICRO)
        t_batch = time.perf_counter() - t0
        torch.cuda.empty_cache()

        scorer = PrefixScorer(eng, fast=False, microbatch=MICRO)
        fb = []

        def fallback(imgs, comp):
            fb.append(len(comp))
            return serial_score(eng, imgs, comp, ROT)

        t0 = time.perf_counter()
        gated = scorer.maybe_validate_and_score(images, compiled, ROT, fallback)
        t_gate = time.perf_counter() - t0

    def key(r):
        sv = sorted(r.scores.items(), key=lambda kv: -kv[1])
        return sv[0][0], sv[0][1] - sv[1][1]

    ag = sum(key(s)[0] == key(g)[0] for s, g in zip(serial, gated))
    ab = sum(key(s)[0] == key(b)[0] for s, b in zip(serial, batched))
    for i, (s, b, g) in enumerate(zip(serial, batched, gated)):
        ks, ms = key(s); kb, mb = key(b); kg, mg = key(g)
        if ks != kb or ks != kg:
            print(f"FLIP Q{i}: serial={ks}@{ms:.3f} batch={kb}@{mb:.3f} gate={kg}@{mg:.3f}", flush=True)
    import statistics
    m = [key(s)[1] for s in serial]
    print(f"n={len(compiled)} serial={t_serial:.1f}s batch={t_batch:.1f}s gate={t_gate:.1f}s "
          f"gate_speedup={t_serial / t_gate:.2f}x batch_speedup={t_serial / t_batch:.2f}x "
          f"gate_vs_batch={t_batch / t_gate:.2f}x", flush=True)
    print(f"parity_gate={ag}/{len(compiled)} parity_batch={ab}/{len(compiled)} "
          f"fallbacks={fb} margin_min={min(m):.3f} margin_med={statistics.median(m):.3f}", flush=True)
    print(f"meta={scorer.metadata}", flush=True)


if __name__ == "__main__":
    main()
