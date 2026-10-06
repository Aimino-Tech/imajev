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
## Regeln bis Parity grün

- Gate disablt korrekt; Prefix-Speedup unquotable (aktuell 0×).
- Result-Cache (identischer Request): cold 1345ms → warm 0.7ms (~1900×).
- Keine Toleranz-Diskussion vor geklärtem Residuum (H4).
