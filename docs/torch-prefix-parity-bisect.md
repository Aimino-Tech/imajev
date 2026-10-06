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

1. **Hybrid-States (BESTÄTIGT am Code, Fix in Arbeit):**
   `LinearAttentionAndFullAttentionLayer` hat keinen eigenen
   `batch_repeat_interleave` — MRO gewinnt `DynamicLayer` (nur keys/values),
   conv/recurrent bleiben bei B>1 auf B=1. `_repeat_cache` hielt das für
   Erfolg und lief nie in den Linear-Fallback. Fix: nach Top-Level-repeat
   immer `_repeat_linear_layer` über alle Layer (idempotent via Guard).
2. **Prefill-Maske (GESCHWÄCHT):** MLX prefillt ebenfalls ohne Maske und
   funktioniert; einzelne Prompts sind vor dem Split ungepaddet. r5-Muster
   damit nicht erklärt — nach Hybrid-Fix neu bewerten.
3. **Referenz-Padding:** Bisect-Ref nutzte `processor(..., padding=True)`
   (default rechts) statt `eng.collate` (links, decision bei -1).
   Vor Fix erst Ref mit `collate` verifizieren.
4. **Residuum 0.067:** nach Fix neu messen; Kandidaten bf16-Rundung
   (Logits ~21) vs Fallback-Kernel-Chunking. float32-Kontrolllauf trennt
   Numerik von Logik.
## Regeln bis Parity grün

- Gate disablt korrekt; Prefix-Speedup unquotable (aktuell 0×).
- Result-Cache (identischer Request): cold 1345ms → warm 0.7ms (~1900×).
- Keine Toleranz-Diskussion vor Fix 1+3.
