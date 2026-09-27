"""server --fast tokenizes once: the cached label ids and the tokenizer-only text path must give exactly the ids of the
original path (verified_label_ids over the whole prompt, and the multimodal processor). Uses the real Qwen3.5 tokenizer
and processor from the local snapshot; no model weights are loaded."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from torch_decision import TorchDecision  # noqa: E402
from vision_decision.scoring import verified_label_ids  # noqa: E402

BUNDLE = ROOT / "artifacts/model-qwen4b.json"
HARD = ROOT / ".cache/external/jevbench/datasets/public/hard.jsonl"


@pytest.fixture(scope="module")
def engine():
    if not BUNDLE.is_file() or not Path(json.loads(BUNDLE.read_text())["path"]).is_dir():
        pytest.skip("local Qwen3.5-4B snapshot is missing")
    from transformers import AutoProcessor
    engine = TorchDecision.__new__(TorchDecision)  # tokenizer and processor only
    engine.processor = AutoProcessor.from_pretrained(json.loads(BUNDLE.read_text())["path"], local_files_only=True)
    engine.max_length, engine.codes, engine._codebook = 1 << 20, 256, None
    return engine


def raw_prompts():
    rows = [json.loads(line) for line in HARD.read_text().splitlines()[:6]] if HARD.is_file() else []
    texts = [f"State:\n{json.dumps(row, ensure_ascii=False)}\nWhich label fits?\n" for row in rows]
    return texts + ["Is the parcel damaged?\nA: yes\nB: no", "Ends with spaces and a newline  \n", "Émoji 🙂 and 中文 text\n\n"]


def test_cached_label_ids_match_whole_prompt_verification(engine):
    codes = engine.labels(256)  # A..Z then two-letter codes: the full extended readout
    for text in raw_prompts():
        for n_images in (0, 1):
            rendered = engine.render(text, n_images)
            assert rendered.endswith(TorchDecision.DECISION_TAIL)
            for labels in (codes[:2], codes[:26], codes):
                assert engine.label_ids(rendered, labels) == verified_label_ids(engine.processor.tokenizer, rendered, labels)


def test_image_path_matches_the_processor_bit_for_bit(engine):
    from PIL import Image
    bench = ROOT.parent / "imajev-bench-public/data"
    paths = sorted((bench / "assets").glob("*"))[:12] if bench.is_dir() else []
    images = [Image.open(p).convert("RGB") for p in paths] or [Image.new("RGB", (1000, 700), (40, 120, 200))]
    images.append(images[0].resize((517, 389)))  # a size that is not a multiple of 32, so the resize really resamples
    labels = engine.labels(3)
    for image in images:
        _, fast, fast_ids = engine.prepare_fast([image], "What is shown?\nA: x\nB: y\nC: z", labels)
        _, slow, slow_ids = engine.prepare([image], "What is shown?\nA: x\nB: y\nC: z", labels)
        assert fast["pixel_values"].dtype == torch.uint8
        assert torch.equal(engine.normalize_patches(fast["pixel_values"]), slow["pixel_values"])
        for key in ("input_ids", "image_grid_thw", "mm_token_type_ids", "attention_mask"):
            if key in slow:
                assert torch.equal(fast[key], slow[key]), key
        assert fast_ids == slow_ids


def test_text_only_ids_match_the_processor(engine):
    labels = engine.labels(4)
    for text in raw_prompts():
        _, fast, fast_ids = engine.prepare_fast([], text, labels)
        _, slow, slow_ids = engine.prepare([], text, labels)
        assert fast["input_ids"].tolist() == slow["input_ids"].tolist()
        assert fast_ids == slow_ids
