"""Fold the imajev PEFT adapter into Qwen3.5 weights and save a plain checkpoint (for vLLM thought generation).

Same arithmetic as `server.py --merge-lora`: the low-rank sum is formed in float32 and rounded once to bfloat16. The decision
readout is not part of this checkpoint (vLLM only writes thoughts; the decision is read by the torch server).

    python scripts/export_merged.py --adapter <peft adapter dir> --out <dir> [--model-bundle artifacts/model-qwen4b.json]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--model-bundle", default=str(ROOT / "artifacts/model-qwen4b.json"))
    a = ap.parse_args(argv)
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    base = json.loads(Path(a.model_bundle).read_text())["path"]
    model = Qwen3_5ForConditionalGeneration.from_pretrained(base, local_files_only=True, dtype=torch.float32)
    model = PeftModel.from_pretrained(model, a.adapter).merge_and_unload().to(torch.bfloat16).eval()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    AutoProcessor.from_pretrained(base, local_files_only=True).save_pretrained(out)
    (out / "MERGED_FROM.json").write_text(json.dumps({"base": base, "adapter": str(Path(a.adapter).resolve()),
                                                      "merge": "float32 sum, rounded once to bfloat16"}, indent=2) + "\n")
    print(f"merged checkpoint written to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
