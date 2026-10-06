# Torch Prefix-KV Parity Bisect (2026-10-06)

Setup: Qwen3.5-4B base, bf16, CUDA 16G, transformers 5.18.0, text-only,
2 same-type choice prompts (Ada/Lagos-Record), Referenz
`candidate_logits_batch`. Skripte: `/tmp/gpu_bisect{,2,3,4,5}.py` (nicht im Repo).

## Runden

- **r1** (`gpu_bisect.py`): flat + hier, shared=53, suffix je 14.
  `match=False max_delta=0.067`, argmax identisch. question_shared=0 →
  Frage-Branch unschuldig. Gate-0.219 NICHT reproduziert (Gate nutzt
  rotierte Panels ungleicher Länge + microbatch).
- **r2** (`gpu_bisect2.py`): V1 ohne attention_mask 0.093 (Maske unschuldig);
  V3 suffix-ohne-cache 7.5 (erwartbar kaputt, kein Signal); V2 fehlt
  (Skriptfehler, keine Aussage).
- **r3** (`gpu_bisect3.py`): direct-reuse 0.63 vs repeat-1 0.41.
  Scheinbar Repeat-Effekt — aber r4 widerlegt (s.u.).
- **r4** (`gpu_bisect4.py`): prefill mit/ohne Maske identisch 0.093;
  **shared-copy-reuse 0.63** = Suffix-Call mutiert Cache trotz
  `use_cache=False`. Frische Kopie pro Branch PFLICHT. r3-0.41 erklärt:
  zweiter Branch las mutierten Cache.
- **r5** (`gpu_bisect5.py`): shared=44, suffix 19/45/100 →
  delta 14.8/13.3/0.016. Je mehr durch Prefix läuft, desto größer der
  Fehler. Stärkste Spur: Prefix-Prefill ohne Maske vs seriell mit
  (left-pad) bei ungleichen Längen. r1–r4 gleiche Längen → kein Pad →
  Residuum 0.067.

## Offene Hypothesen (priorisiert)

1. **Hybrid-States (ZURÜCKGEZOGEN 11:20 — trifft falsche Klasse):**
   Qwen3.5-`layer_types` = getrennt `linear_attention`/`full_attention`,
   nie `hybrid`. Re-Bisect mit f55b7bc: same-B2 0.067 UNVERÄNDERT,
   mixed-B2 0.108 (r5-14.8 kam vom falsch rechts-gepaddeten Ref, nicht
   vom Prefix). Fix No-Op für dieses Modell, aber harmlos + getestet.
   Laufzeit-Typen/Shape-Check am GPU-Cache steht noch aus.
2. **Prefill-Maske (GESCHWÄCHT):** MLX prefillt ebenfalls ohne Maske und
   funktioniert; Einzel-Prompts sind vor dem Split ungepaddet. r4 zeigte
   Prefill mit/ohne Maske identisch (0.093). r5-Muster damit NICHT erklärt
   (war Ref-Artefakt) — kein aktiver Fix-Kandidat mehr.
3. **Referenz-Padding (GEKLÄRT):** r5 nutzte `processor(padding=True)`
   = rechts, `collate` = links. Re-Bisect mit collate-Ref: mixed 0.108
   statt 14.8. Lehre: Bisect-Refs immer via `collate` bauen.
4. **Residuum 0.07–0.11 (OFFEN, einzige aktive Spur):** same 0.067,
   mixed 0.108, argmax stabil, Logits ~21. Kandidaten: bf16-Rundung vs
   Fallback-Kernel-Chunking (causal_conv1d/flash-Warnings). fp32-Kontrolle
   OOM-unmöglich (15.6G). Nächste Sonden: Laufzeit-Cache-Typen/Shapes am
   GPU-Cache; Gate-0.219 vs Bisect-0.067 Diskrepanz (rotierte Panels).

## Neue Priorisierung nach Codevergleich (2026-10-06)

1. **GDN-Split an 64er Chunkgrenze** — jetzt im Code umgesetzt, GPU-Parity
   noch messen. HF Transformers 5.18.0 nutzt für Qwen3.5 `chunk_size=64`;
   r1/r5 trennten bei 53/44 mitten im Chunk.
2. **Cache branch state / B>1** — weiter prüfen; Branches müssen frische
   Cache-Kopien bekommen und alle Linear-Attention-Zustände pro Row tragen.
3. **bf16/kernel residue** — erst nach aligned split neu messen.
4. **Prefill-Maske / Referenz-Padding** — herabgestuft: MLX prefillt ebenfalls
   unmasked; der Live-Gate-Referenzpfad ist pro Prompt ungepaddet.

Erwarteter nächster GPU-Test: exakt denselben Live-Gate-Probe mit
`shared_prefix_tokens % 64 == 0` ausgeben und Candidate-vs-Reference delta +
argmax messen. Keine Toleranz lockern.
## Befund 11:35 — numerischer Parity-Floor B1 vs B=N (korrigiert, ex-Super-KI)

Seriell B=1 vs B=2, identische Prompts: delta 0.177/0.071. Das ist KEINE
Instabilität/Non-Determinismus (ungeprüft), sondern batch-shape-/kernel-
Sensitivität: f(x,B=1) != exakt f(x,B=2) bei bf16+GDN. Korrekt formuliert:
canonical-B1 und batched full-forward haben einen numerischen Parity-Floor
von 0.07–0.18. GDN-aligned shared=128: delta 0.121, argmax stabil.
Fazit: Gate verglich R1 (B=1 seriell) mit P (Prefix, suffix-Batch B=N) und
mass damit cache-split + B=1→B=N + Kernel-Geometrie + bf16 aufsummiert.
Fehlende Messung: R2 (full-forward B=N via collate) als Baseline, dann
prefix_error = Δ(R2,P) statt total_error = Δ(R1,P).
Messung 11:50 (shared=128, aligned): rep-B1=0, rep-B2=0 → deterministisch,
"instabil" endgültig falsch. Aber: batch_noise 0.111/0.141, prefix_error
Δ(R2,P) 0.083/0.121, total Δ(R1,P) 0.035/0.021. Super-KIs guter Fall
(prefix_error ~0.008) trifft NICHT zu — Prefix hat echten Fehler ~0.1
gegenüber gleichem Execution-Mode. argmax überall stabil. Fazit: kein
Messartefakt mehr übrig; Restfehler sitzt im Prefix-Pfad selbst (split bei
Messung 12:0x (5-Stufen-Leiter, frischer Prefill pro Stufe — Suffix-Calls
mutieren Cache trotz use_cache=False, drift 26.7!): S1 no-copy 0.033, S2
deepcopy 0.033 (Clone unschuldig), S3 repeat-B2-identisch 0.009 (Repeat
unschuldig), S4 verschieden rowA 0.009/rowB 0.086. Sprung nur bei
inhaltlich abweichendem Suffix. Split-Sweep rowB: 128→0.089, 130→0.030,
133→0.013, 135→0.078, 143→0.015 — exakt reproduzierbar, aber ohne
Chunk-Raster. Layer-Hooks: Drift 0.001 (l0) → 0.625 (l31), monoton
akkumulierend, kein Sprung-Layer. Deutung: GDN-Recurrence setzt über Split
nicht exakt fort (Fallback-Kernel, causal_conv1d+FLA fehlen); kleiner
Anfangsfehler verstärkt sich pro Layer. Deterministisch, argmax-stabil.
Messung 12:4x (FLA 0.5.2 aktiv via triton 3.8.0, causal_conv1d fehlt weiter —
CUDA-Mismatch, nicht baubar): S1 0.033→0.137 (SCHLECHTER), S3 0.009→0.006,
S4-rowB 0.086→0.075. Micro-Test synthetisch: chunk-split==full (1.5e-5),
Chunk-Decomposition unschuldig; qkv-preconv exakt 0, Drift entsteht in
Conv/GDN. Fazit: Split-Fehler kernel-abhängig, aber kein Kernel-Pfad führt
zu Parity. Gate-Entscheidung nötig (Pfad meiden vs Floor-Toleranz), keine
weitere Messung.
Rollback 12:5x: triton 3.8.0→3.1.0, FLA/fla-core deinstalliert, Smoke grün
(Fallback-Warnungen zurück, Logit-Max 18.4 plausibel). Beschluss: kein
experimenteller Kernel-Pfad im Serving-Stack; ggf. separater Container.
Win-Messung 12:6x (4 Fragen/4 Rotationen, shared-prefix 448 Tok, text-only):
serial 3.32s vs prefix 0.64s = 5.20x; 12 Fragen: 9.65s vs 1.56s = 6.20x.
Decision-Parity 12/12 (inkl. knapper Margins Q8/Q10, Drift ohne Flip),
max prob-delta 0.05. Gate (argmax + MARGIN_FLOOR) trägt: klare Fälle prefix,
knappe seriell. Fazit: Production-Win bei 4x-Sampling.

Messung 13:x (HEAD 9803309, Cross-Question-Batching, Wiki 35Qx4 fast=True):
serial 17.2s / batch 13.4s / gate 9.2s = gate_vs_batch 1.46x (vorher 1.26x),
parity_gate 34/35, batch 32/35, fb 8. Q-Prefills gleiche-Laenge-gepackt
(8 Batches statt 35xB1), Suffixe pro Frage aus In-Place-expandiertem Cache
(OOM-Fix: kein Deepcopy). Alle q_shared=64 (raw 65-73). B2-Q-Prefill-Drift
0.09 (Sonden) traegt nicht bis Logit/Argmax durch — Tiling-Rauschen, kein
Gate-Risiko. 50.2-Ausreisser war Sonden-Artefakt (Padding+Inhalt gemischt).

Messung 14:x (267a022, Margin-Fallback entfernt, Showdown p50 n=35 rot4 mb16):
serial 17.3s / batch 14.6s / warm-gate 5.1s = 2.84x vs batch, 3.37x vs serial,
Paritaet 34/35 (Q17: serial true@0.003 vs prefix false@0.001 — beidseitiges
Raten, kein Prefix-Fehler). Phasen (sync): global 1xB1x128 0.07s + Q-Prefills
8xB4-5x64 0.99s (19%) + Suffixe 8xB13-16x45-87 3.83s (74%) + repeat 0.05s.
Faktoren: 1. Suffix GDN-Fallback-Kernel (FLA/causal_conv1d fehlen), 2. Q-Prefill
Launches B4-5 (alle q_shared=64, ein B35-Batch moeglich?), 3. Suffix-Laengen
45-87T, 4. Vision/CPU marginal. Showdown-OOMs unterwegs: fehlendes
inference_mode (Autograd hielt 140 Graphen/4.6GB) + 16 Graph-Laengen (~4GB).
