"""Thought generation for think-if-unsure through a vLLM OpenAI-compatible server (fast, batched across concurrent calls).

The server runs the same weights the decision is read with (the merged imajev-4b, scripts/export_merged.py). Messages are
built exactly as TorchDecision.render_thinking builds them (images first, then the prompt; enable_thinking), so vLLM renders the
same chat template; images go as lossless PNG of the already-decoded image, so the pixels are the ones the decision sees.
Generation is greedy and stops at '</think>'. The thought comes back as text and is re-tokenized with the same tokenizer.

    engine = VLLMThoughts("http://127.0.0.1:8000", "imajev-4b", tokenizer)
    tokens, closed, prompt_tokens = engine.think(images, prompt, max_tokens=256)
"""
from __future__ import annotations

import base64
import io
import json
import urllib.request


def png_data_url(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


class VLLMThoughts:
    def __init__(self, url, model, tokenizer, timeout=600):
        self.url, self.model, self.tok, self.timeout = url.rstrip("/"), model, tokenizer, timeout
        ids = tokenizer.encode("</think>", add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError("'</think>' is not a single token")
        self.think_end = ids[0]

    def think(self, images, prompt, max_tokens):
        """-> (thought token ids without '</think>', closed by the model?, prompt tokens as vLLM counted them)."""
        images = [] if images is None else images if isinstance(images, list) else [images]
        content = [{"type": "image_url", "image_url": {"url": png_data_url(image)}} for image in images] + [{"type": "text", "text": prompt}]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": int(max_tokens),
                "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": True}, "stop_token_ids": [self.think_end],
                "skip_special_tokens": False}
        request = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read())
        choice = payload["choices"][0]
        text = choice["message"].get("content") or ""
        if choice["message"].get("reasoning_content"):  # a server started with a reasoning parser splits the thought off
            text = choice["message"]["reasoning_content"]
        text = text.split("</think>")[0]
        closed = choice.get("finish_reason") == "stop"
        return self.tok.encode(text, add_special_tokens=False), closed, payload.get("usage", {}).get("prompt_tokens")
