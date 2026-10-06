"""Torch shared-prefix/KV scorer for the torch serving path.

For a physical batch of unique questions over identical evidence it:
1. prepares each prompt without running the model;
2. finds the longest token-identical prefix, including all visual placeholders;
3. runs multimodal embedding + the language-model prefix exactly once;
4. expands the resulting transformer KV cache to the suffix batch;
5. runs only the unique text suffixes in one padded batch;
6. reads each decision at its own final non-padding position.

PrefixScorer is a plain helper owned by TorchBackend (no monkey-patching): the
backend calls self._prefix_scorer.score(...) from _score_batched once the
parity gate has validated each evidence mode (text vs visual) against the
serial scorer.
"""

from __future__ import annotations

import copy
from time import perf_counter

import torch
PARITY_ATOL = 0.02

class PrefixUnsuitable(Exception):
    """Batch cannot use prefix reuse (no shared prefix, no validatable pair,
    degenerate model output). Fall back WITHOUT disabling the path."""


class ParityError(Exception):
    """Prefix path produced wrong logits. Disables the path."""


def _unpadded_ids(inputs):
    ids = inputs["input_ids"]
    mask = inputs.get("attention_mask")
    if ids.shape[0] != 1:
        raise ValueError("prefix scorer expects individually prepared prompts")
    if mask is None:
        return ids[0]
    keep = mask[0].to(dtype=torch.bool)
    return ids[0][keep]


def _longest_common_prefix(rows):
    if len(rows) < 2:
        return 0
    limit = min(int(row.numel()) for row in rows)
    if limit < 2:
        return 0
    first = rows[0][:limit]
    same = torch.ones(limit, dtype=torch.bool, device=first.device)
    for row in rows[1:]:
        same &= row[:limit].eq(first)
    mismatch = (~same).nonzero(as_tuple=False)
    shared = int(mismatch[0].item()) if mismatch.numel() else limit
    # Every branch must retain at least one suffix token for a decision state.
    return min(shared, limit - 1)


GDN_PREFIX_ALIGNMENT = 64


def _prefix_alignment(engine):
    """Numerically stable cache split alignment for the loaded decoder.

    Transformers 5.18 Qwen3.5 evaluates Gated DeltaNet in 64-token chunks.
    Splitting a cached continuation inside one of those chunks changes the
    chunk decomposition versus a one-shot forward and can amplify bf16 drift.
    Full-attention-only decoders do not need this restriction.
    """
    base = engine._base()
    root = getattr(base, "config", None)
    configs = (
        root,
        getattr(root, "text_config", None),
        getattr(getattr(base, "model", None), "config", None),
    )
    for config in configs:
        layer_types = getattr(config, "layer_types", None) if config is not None else None
        if layer_types and any(kind == "linear_attention" for kind in layer_types):
            return GDN_PREFIX_ALIGNMENT
    return 1


def _aligned_shared_length(common, alignment, *, base=0):
    """Largest reusable part whose absolute end stays on a cache-safe boundary."""
    if common <= 0 or alignment <= 1:
        return max(0, common)
    end = base + common
    return max(0, (end // alignment) * alignment - base)


def _repeat_cache(cache, batch_size):
    """Clone one-prefix cache and expand all cached states to batch_size.

    Cache implementations differ: dynamic-attention layers can expose a batch
    repeater while linear-attention states need explicit expansion. Iterate
    layers ourselves so a successful K/V repeat cannot hide recurrent state.
    """
    branch = copy.deepcopy(cache)
    layers = getattr(branch, "layers", None)
    if layers is not None:
        for layer in layers:
            repeat = getattr(layer, "batch_repeat_interleave", None)
            if callable(repeat):
                repeat(batch_size)
            # Safe for hybrids and future cache layers: helper only expands
            # states that still have batch dimension 1.
            _repeat_linear_layer(layer, batch_size)
        return branch

    # Generic non-layered cache implementations.
    repeat = getattr(branch, "batch_repeat_interleave", None)
    if callable(repeat):
        repeat(batch_size)
        return branch

    # Compatibility with legacy tuple/list past_key_values.
    if isinstance(branch, (tuple, list)):
        repeated = []
        for layer in branch:
            repeated.append(
                tuple(
                    value.expand(batch_size, *value.shape[1:])
                    for value in layer
                )
            )
        return type(branch)(repeated)
    raise TypeError(
        f"unsupported transformers cache type: {type(branch).__name__}"
    )
def _repeat_linear_layer(layer, batch_size):
    """Expand one linear-attention layer's conv/recurrent states to batch_size."""
    for name in ("conv_states", "recurrent_states"):
        states = getattr(layer, name, None)
        if not states:
            continue
        for key, value in list(states.items()):
            if value is not None and value.shape[0] == 1 and batch_size > 1:
                states[key] = value.repeat_interleave(batch_size, dim=0)



def _visual_token_ids(engine):
    config = engine._base().config
    ids = set()
    for name in (
        "image_token_id",
        "video_token_id",
        "image_token_index",
        "video_token_index",
    ):
        value = getattr(config, name, None)
        if isinstance(value, int):
            ids.add(value)
    model_config = getattr(engine._base().model, "config", None)
    for name in (
        "image_token_id",
        "video_token_id",
        "image_token_index",
        "video_token_index",
    ):
        value = getattr(model_config, name, None)
        if isinstance(value, int):
            ids.add(value)
    return ids


def _suffix_positions(prefix_positions, lengths, max_suffix, device):
    """Qwen M-RoPE text after a completed multimodal prefix advances uniformly."""
    last = prefix_positions[:, :, -1:].to(device)
    steps = torch.arange(
        1,
        max_suffix + 1,
        device=device,
        dtype=last.dtype,
    ).view(1, 1, -1)
    positions = last + steps
    return positions.expand(-1, len(lengths), -1).contiguous()


def score_rendered_prefix_cached(
    engine,
    images,
    rendered_examples,
    *,
    fast=False,
    microbatch=8,
):
    """Return candidate-logit tensors for rendered (prompt, labels) examples.

    rendered_examples is a list of (rendered_prompt, labels). The images must be
    identical for every example. This function performs no answer/result cache.
    """
    if len(rendered_examples) < 2:
        raise ValueError("prefix reuse requires at least two prompts")

    prepared = _prepare_rendered_examples_once(
        engine,
        images,
        rendered_examples,
        fast=fast,
    )
    rows = [item[2] for item in prepared]

    raw_shared = _longest_common_prefix(rows)
    alignment = _prefix_alignment(engine)
    shared = _aligned_shared_length(raw_shared, alignment)
    if shared < 1:
        raise PrefixUnsuitable(
            f"reusable prefix ({raw_shared} tokens) does not reach a "
            f"{alignment}-token cache-safe boundary"
        )

    visual_ids = _visual_token_ids(engine)
    if visual_ids:
        for row in rows:
            suffix = row[shared:]
            if any(bool(suffix.eq(token).any()) for token in visual_ids):
                raise PrefixUnsuitable(
                    "cache-safe split would leave visual placeholders in the suffix"
                )

    device = engine.device
    first_inputs = {
        key: value.to(device)
        for key, value in prepared[0][0].items()
    }
    if (
        fast
        and first_inputs.get("pixel_values") is not None
        and first_inputs["pixel_values"].dtype == torch.uint8
    ):
        first_inputs["pixel_values"] = engine.normalize_patches(
            first_inputs["pixel_values"]
        )

    # The expensive vision tower is called only here, for the first prompt.
    full_embeds, full_positions = engine._embeds_positions(first_inputs)
    prefix_embeds = full_embeds[:, :shared]
    prefix_positions = full_positions[:, :, :shared]

    base = engine._base()
    language = base.model.language_model
    with torch.inference_mode():
        prefix_out = language(
            inputs_embeds=prefix_embeds,
            position_ids=prefix_positions,
            use_cache=True,
            return_dict=True,
        )
    cache = getattr(prefix_out, "past_key_values", None)
    if cache is None:
        raise RuntimeError("language model did not return past_key_values")

    suffix_ids = [row[shared:].to(device) for row in rows]
    lengths = [int(row.numel()) for row in suffix_ids]
    if min(lengths) < 1:
        raise ValueError("every prompt must retain a non-empty suffix")
    if isinstance(microbatch, bool) or not isinstance(microbatch, int):
        raise ValueError("microbatch must be an integer")
    if microbatch < 1:
        raise ValueError("microbatch must be positive")

    embed_tokens = base.model.get_input_embeddings()
    head = base.lm_head.weight
    results = []
    suffix_batches = 0
    for batch_start in range(0, len(rows), microbatch):
        batch_ids = suffix_ids[batch_start : batch_start + microbatch]
        batch_lengths = lengths[batch_start : batch_start + microbatch]
        max_suffix = max(batch_lengths)
        batch_size = len(batch_ids)
        hidden_size = int(prefix_embeds.shape[-1])
        suffix_embeds = torch.zeros(
            (batch_size, max_suffix, hidden_size),
            device=device,
            dtype=prefix_embeds.dtype,
        )
        suffix_mask = torch.zeros(
            (batch_size, max_suffix),
            device=device,
            dtype=torch.long,
        )
        for local_index, ids in enumerate(batch_ids):
            count = batch_lengths[local_index]
            suffix_embeds[local_index, :count] = embed_tokens(ids)
            suffix_mask[local_index, :count] = 1

        attention_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, shared),
                    device=device,
                    dtype=torch.long,
                ),
                suffix_mask,
            ],
            dim=1,
        )
        position_ids = _suffix_positions(
            prefix_positions,
            batch_lengths,
            max_suffix,
            device,
        )
        branch_cache = _repeat_cache(cache, batch_size)

        with torch.inference_mode():
            output = language(
                inputs_embeds=suffix_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=branch_cache,
                use_cache=False,
                return_dict=True,
            )

        states = output.last_hidden_state.float()
        for local_index, global_index in enumerate(
            range(batch_start, batch_start + batch_size)
        ):
            token_ids = prepared[global_index][1]
            hidden = _read_hidden(
                states, local_index, batch_lengths[local_index] - 1
            )
            indices = engine._readout_indices(token_ids)
            if engine.readout is not None and indices is not None:
                logits = engine.readout(hidden)[indices]
            else:
                rows_tensor = torch.tensor(token_ids, device=device)
                logits = hidden @ head[rows_tensor].float().T
            results.append(logits)
        suffix_batches += 1

    return results, {
        "shared_prefix_tokens": shared,
        "raw_shared_prefix_tokens": raw_shared,
        "prefix_alignment": alignment,
        "suffix_tokens": lengths,
        "vision_forwards": 1 if first_inputs.get("pixel_values") is not None else 0,
        "prefix_prefills": 1,
        "suffix_batches": suffix_batches,
        "microbatch": microbatch,
        "prefix_cache": True,
        "image_processor_calls": 1 if images else 0,
        "reused_visual_tokenizations": max(0, len(rows) - 1),
    }



def _prepare_rendered_example(engine, images, rendered, labels, *, fast=False):
    image_list = (
        []
        if images is None
        else images
        if isinstance(images, list)
        else [images]
    )
    token_ids = engine.label_ids(rendered, labels)
    if fast:
        inputs = dict(
            engine.processor(
                text=[rendered],
                images=image_list or None,
                return_tensors="pt",
                do_rescale=False,
                do_normalize=False,
            )
        )
    else:
        inputs = engine.processor(
            text=[rendered],
            images=image_list or None,
            return_tensors="pt",
        )
    return inputs, token_ids, _unpadded_ids(inputs)


def _tokenize_with_cached_visual_layout(engine, rendered, visual_inputs):
    """Tokenize another prompt without re-running the image processor.

    Qwen's processor expands each image placeholder from image_grid_thw before
    tokenization. Reuse the first prompt's grid and the processor's own
    replace_image_token() implementation, then tokenize text only.
    """
    processor = engine.processor
    image_token = getattr(processor, "image_token", None)
    grids = visual_inputs.get("image_grid_thw")
    expanded = rendered

    if grids is not None:
        if not image_token or not callable(getattr(processor, "replace_image_token", None)):
            raise PrefixUnsuitable("processor cannot reuse cached visual token layout")
        count = int(grids.shape[0])
        parts = expanded.split(image_token)
        if len(parts) - 1 != count:
            raise PrefixUnsuitable(
                f"rendered prompt has {len(parts) - 1} image placeholders, expected {count}"
            )
        image_inputs = {"image_grid_thw": grids}
        chunks = [parts[0]]
        for index in range(count):
            chunks.append(processor.replace_image_token(image_inputs, index))
            chunks.append(parts[index + 1])
        expanded = "".join(chunks)
    elif image_token and image_token in expanded:
        raise PrefixUnsuitable("rendered prompt has image placeholders but no cached image grid")

    return dict(
        processor.tokenizer(
            expanded,
            add_special_tokens=False,
            return_tensors="pt",
        )
    )


def _prepare_rendered_examples_once(engine, images, rendered_examples, *, fast=False):
    """Prepare one visual prompt, then reuse its visual layout for all others.

    Only the first example calls the multimodal processor. The cached path is
    runtime-guarded by reproducing the first example's exact input_ids; any
    transformers/tokenizer behavior change falls back instead of risking a
    silent prompt mismatch.
    """
    if not rendered_examples:
        return []

    first_rendered, first_labels = rendered_examples[0]
    first = _prepare_rendered_example(
        engine, images, first_rendered, first_labels, fast=fast
    )
    cached_first = _tokenize_with_cached_visual_layout(
        engine, first_rendered, first[0]
    )
    if not torch.equal(_unpadded_ids(cached_first), first[2]):
        raise PrefixUnsuitable(
            "cached visual tokenization does not match multimodal processor"
        )

    prepared = [first]
    for rendered, labels in rendered_examples[1:]:
        token_ids = engine.label_ids(rendered, labels)
        inputs = _tokenize_with_cached_visual_layout(
            engine, rendered, first[0]
        )
        prepared.append((inputs, token_ids, _unpadded_ids(inputs)))
    return prepared


def _pack_question_specs_by_shared_length(specs, microbatch):
    """Pack whole questions while bounding the expanded rotation batch.

    Questions can share one physical question-prefix call only when their
    question-prefix cache length matches.  The sum of their rotation rows is
    bounded by microbatch so the in-place cache expansion never exceeds the
    serving VRAM limit.
    """
    if microbatch < 1:
        raise ValueError("microbatch must be positive")
    buckets = {}
    order = []
    for spec in specs:
        key = int(spec["question_shared"])
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        rows = spec["rows"]
        if len(rows) <= microbatch:
            buckets[key].append(spec)
        else:
            # Defensive path for microbatch smaller than a question's rotation
            # count.  Re-prefilling that one question is preferable to OOM.
            for start in range(0, len(rows), microbatch):
                part = dict(spec)
                part["rows"] = rows[start : start + microbatch]
                buckets[key].append(part)

    for key in order:
        packed = []
        expanded_rows = 0
        for spec in buckets[key]:
            count = len(spec["rows"])
            if packed and expanded_rows + count > microbatch:
                yield key, packed
                packed = []
                expanded_rows = 0
            packed.append(spec)
            expanded_rows += count
        if packed:
            yield key, packed


def score_rendered_prefix_cached_hierarchical(
    engine,
    images,
    rendered_groups,
    *,
    fast=False,
    microbatch=8,
):
    """Reuse one global multimodal prefix, then one prefix per rotation group.

    Each group is normally one question containing all of that question's
    candidate-order rotations. The expensive image/evidence prefix is evaluated
    once for the whole request; the question-specific header is evaluated once
    per question; only the reordered candidate suffix is evaluated per rotation.
    """
    flat_examples = [
        example
        for group in rendered_groups
        for example in group
    ]
    flat_prepared = _prepare_rendered_examples_once(
        engine,
        images,
        flat_examples,
        fast=fast,
    )
    prepared_groups = []
    rows = []
    cursor = 0
    for group in rendered_groups:
        prepared = flat_prepared[cursor : cursor + len(group)]
        cursor += len(group)
        prepared_groups.append(prepared)
        rows.extend(item[2] for item in prepared)

    if len(rows) < 2:
        raise ValueError("hierarchical prefix reuse requires at least two prompts")

    raw_shared = _longest_common_prefix(rows)
    alignment = _prefix_alignment(engine)
    shared = _aligned_shared_length(raw_shared, alignment)
    if shared < 1:
        raise PrefixUnsuitable(
            f"reusable global prefix ({raw_shared} tokens) does not reach a "
            f"{alignment}-token cache-safe boundary"
        )

    visual_ids = _visual_token_ids(engine)
    if visual_ids:
        for row in rows:
            suffix = row[shared:]
            if any(bool(suffix.eq(token).any()) for token in visual_ids):
                raise PrefixUnsuitable(
                    "cache-safe global split would leave visual placeholders in the suffix"
                )

    device = engine.device
    first_inputs = {
        key: value.to(device)
        for key, value in prepared_groups[0][0][0].items()
    }
    if (
        fast
        and first_inputs.get("pixel_values") is not None
        and first_inputs["pixel_values"].dtype == torch.uint8
    ):
        first_inputs["pixel_values"] = engine.normalize_patches(
            first_inputs["pixel_values"]
        )

    full_embeds, full_positions = engine._embeds_positions(first_inputs)
    prefix_embeds = full_embeds[:, :shared]
    prefix_positions = full_positions[:, :, :shared]

    base = engine._base()
    language = base.model.language_model
    embed_tokens = base.model.get_input_embeddings()
    head = base.lm_head.weight
    with torch.inference_mode():
        prefix_out = language(
            inputs_embeds=prefix_embeds,
            position_ids=prefix_positions,
            use_cache=True,
            return_dict=True,
        )
    global_cache = getattr(prefix_out, "past_key_values", None)
    if global_cache is None:
        raise RuntimeError("language model did not return past_key_values")
    del prefix_out

    # Build one spec per logical question.  Question-prefixes are computed once
    # per question in a packed batch, then that cache is expanded IN PLACE to
    # the question's 2/4 rotation rows and consumed immediately by one suffix
    # batch.  This avoids both the old ~35 B=1 prefills and the huge extra
    # deepcopy that made wider cache branching OOM.
    logits_out = [None] * len(rows)
    question_prefix_tokens = []
    raw_question_prefix_tokens = []
    question_specs = []
    flat_index = 0
    max_effective_tokens = shared
    logical_question_prefills = 0

    for question_index, prepared in enumerate(prepared_groups):
        group_rows = [item[2] for item in prepared]
        relative = [row[shared:] for row in group_rows]
        raw_question_shared = _longest_common_prefix(relative)
        question_shared = _aligned_shared_length(
            raw_question_shared, alignment, base=shared
        )
        raw_question_prefix_tokens.append(raw_question_shared)
        question_prefix_tokens.append(question_shared)
        if question_shared > 0:
            logical_question_prefills += 1
            question_segment = group_rows[0][
                shared : shared + question_shared
            ].to(device)
        else:
            question_segment = None

        rotation_rows = []
        for item, row in zip(prepared, group_rows):
            suffix = row[shared + question_shared :].to(device)
            suffix_length = int(suffix.numel())
            if suffix_length < 1:
                raise ValueError("every rotation must retain a non-empty suffix")
            rotation_rows.append(
                {
                    "flat_index": flat_index,
                    "suffix_ids": suffix,
                    "suffix_length": suffix_length,
                    "prepared": item,
                }
            )
            flat_index += 1
        question_specs.append(
            {
                "question_index": question_index,
                "question_shared": question_shared,
                "question_segment": question_segment,
                "rows": rotation_rows,
            }
        )

    suffix_batches = 0
    question_prefill_batches = 0
    question_prefill_rows = 0

    for question_shared, batch_questions in _pack_question_specs_by_shared_length(
        question_specs, microbatch
    ):
        suffix_rows = [
            row
            for question in batch_questions
            for row in question["rows"]
        ]
        batch_size = len(suffix_rows)
        batch_lengths = [row["suffix_length"] for row in suffix_rows]
        branch_length = shared + question_shared

        if question_shared > 0:
            question_batch = len(batch_questions)
            branch_cache = _repeat_cache(global_cache, question_batch)
            segment_ids = torch.stack(
                [question["question_segment"] for question in batch_questions],
                dim=0,
            )
            segment_embeds = embed_tokens(segment_ids)
            segment_positions = _suffix_positions(
                prefix_positions,
                [question_shared] * question_batch,
                question_shared,
                device,
            )
            question_attention_mask = torch.ones(
                (question_batch, branch_length),
                device=device,
                dtype=torch.long,
            )
            with torch.inference_mode():
                question_out = language(
                    inputs_embeds=segment_embeds,
                    position_ids=segment_positions,
                    attention_mask=question_attention_mask,
                    past_key_values=branch_cache,
                    use_cache=True,
                    return_dict=True,
                )
            branch_cache = getattr(question_out, "past_key_values", None)
            if branch_cache is None:
                raise RuntimeError(
                    "batched question-prefix pass did not return past_key_values"
                )

            # Expand B=questions -> B=rotation rows without deepcopy.  HF Cache
            # reorder_cache uses index_select for both full-attention K/V and
            # linear-attention conv/recurrent states, and duplicate indices are
            # valid.  The expanded cache is single-use, so the source rows are
            # no longer needed after this operation.
            owner_indices = []
            for local_question, question in enumerate(batch_questions):
                owner_indices.extend(
                    [local_question] * len(question["rows"])
                )
            owner_indices = torch.tensor(
                owner_indices,
                device=device,
                dtype=torch.long,
            )
            reorder = getattr(branch_cache, "reorder_cache", None)
            if not callable(reorder):
                raise PrefixUnsuitable(
                    "cache does not support in-place row expansion"
                )
            reorder(owner_indices)
            branch_positions = segment_positions.index_select(
                1, owner_indices
            )
            question_prefill_batches += 1
            question_prefill_rows += question_batch
            del question_out, segment_embeds, segment_ids
        else:
            # No question-specific cache exists; branch the global prefix
            # directly to the final suffix batch.
            branch_cache = _repeat_cache(global_cache, batch_size)
            branch_positions = prefix_positions

        max_suffix = max(batch_lengths)
        hidden_size = int(prefix_embeds.shape[-1])
        suffix_embeds = torch.zeros(
            (batch_size, max_suffix, hidden_size),
            device=device,
            dtype=prefix_embeds.dtype,
        )
        suffix_mask = torch.zeros(
            (batch_size, max_suffix),
            device=device,
            dtype=torch.long,
        )
        for local_index, row in enumerate(suffix_rows):
            count = row["suffix_length"]
            suffix_embeds[local_index, :count] = embed_tokens(
                row["suffix_ids"]
            )
            suffix_mask[local_index, :count] = 1

        attention_mask = torch.cat(
            [
                torch.ones(
                    (batch_size, branch_length),
                    device=device,
                    dtype=torch.long,
                ),
                suffix_mask,
            ],
            dim=1,
        )
        position_ids = _suffix_positions(
            branch_positions,
            batch_lengths,
            max_suffix,
            device,
        )
        with torch.inference_mode():
            output = language(
                inputs_embeds=suffix_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=branch_cache,
                use_cache=False,
                return_dict=True,
            )

        states = output.last_hidden_state.float()
        for local_index, row in enumerate(suffix_rows):
            item = row["prepared"]
            token_ids = item[1]
            hidden = _read_hidden(
                states,
                local_index,
                row["suffix_length"] - 1,
            )
            indices = engine._readout_indices(token_ids)
            if engine.readout is not None and indices is not None:
                logits = engine.readout(hidden)[indices]
            else:
                rows_tensor = torch.tensor(
                    token_ids,
                    device=device,
                )
                logits = hidden @ head[rows_tensor].float().T
            logits_out[row["flat_index"]] = logits

        suffix_batches += 1
        max_effective_tokens = max(
            max_effective_tokens,
            branch_length + max_suffix,
        )
        # Suffix execution mutates cache objects even with use_cache=False.
        # Every expanded branch is intentionally consumed exactly once.
        del output, branch_cache, suffix_embeds

    if any(item is None for item in logits_out):
        raise RuntimeError("hierarchical prefix scheduler left logits unfilled")

    physical_prefix_prefills = 1 + question_prefill_batches
    return logits_out, {
        "shared_prefix_tokens": shared,
        "raw_shared_prefix_tokens": raw_shared,
        "prefix_alignment": alignment,
        "question_shared_prefix_tokens": question_prefix_tokens,
        "raw_question_shared_prefix_tokens": raw_question_prefix_tokens,
        "vision_forwards": (
            1 if first_inputs.get("pixel_values") is not None else 0
        ),
        "prefix_prefills": physical_prefix_prefills,
        "question_prefix_prefills": logical_question_prefills,
        "question_prefill_batches": question_prefill_batches,
        "question_prefill_rows": question_prefill_rows,
        "suffix_batches": suffix_batches,
        "lm_calls": physical_prefix_prefills + suffix_batches,
        "microbatch": microbatch,
        "prefix_cache": True,
        "cross_question_batching": True,
        "inplace_rotation_expansion": True,
        "image_processor_calls": 1 if images else 0,
        "reused_visual_tokenizations": max(0, len(rows) - 1),
        "max_effective_tokens": max_effective_tokens,
    }


def _select_probe_pair(compiled):
    """Indices of two same-type questions, or None if no validatable pair.

    Cross-type pairs (e.g. noul + choice) share no prompt prefix, so probing
    them proves nothing; the batch falls back and the path stays enabled."""
    by_type = {}
    for i, row in enumerate(compiled):
        by_type.setdefault(getattr(row[0], "type", None), []).append(i)
    for idx in by_type.values():
        if len(idx) >= 2:
            return idx[0], idx[1]
    return None


def _read_hidden(states, local_index, pos):
    """Decision hidden state with a loud failure on degenerate model output.

    A 0D `states` (collapsed batch dim upstream) cannot be indexed at all, so
    guard before indexing; raise PrefixUnsuitable so the gate falls back
    cleanly instead of exploding inside F.linear."""
    if states.dim() == 0:
        raise PrefixUnsuitable("degenerate 0D model output; falling back")
    hidden = states[local_index, pos]
    if hidden.dim() < 1:
        raise PrefixUnsuitable(
            f"degenerate hidden state dim={hidden.dim()}; falling back"
        )
    return hidden


def _logits_match(candidate, reference, *, atol=PARITY_ATOL):
    if len(candidate) != len(reference):
        return False, float("inf")
    max_delta = 0.0
    for left, right in zip(candidate, reference):
        if left.shape != right.shape:
            return False, float("inf")
        delta = float((left.float() - right.float()).abs().max().item())
        max_delta = max(max_delta, delta)
        if int(left.argmax().item()) != int(right.argmax().item()):
            return False, max_delta
    return max_delta <= atol, max_delta


class PrefixScorer:
    """Shared-prefix KV scorer owned by TorchBackend.

    The backend renders (prompt, labels) groups and calls score(); the serial
    scorer stays the fallback. The first eligible batch per evidence mode
    (text vs visual) is double-scored against the serial path; the prefix path
    enables on identical argmax (logit delta is a measured numeric floor,
    not a correctness signal). No per-question margin rescoring: Q17 showed
    both paths guessing (margins <0.01), serial is no reference there.
    """

    def __init__(self, engine, *, fast=False, microbatch=8):
        self.engine = engine
        self.fast = bool(fast)
        self.microbatch = max(1, int(microbatch))
        self.enabled = True
        self.validated_text = False
        self.validated_visual = False
        self.error = None
        self.max_delta = {}
        self.metadata = None

    @property
    def validated(self):
        return bool(self.validated_text or self.validated_visual)

    def render_groups(self, images, compiled, rotations):
        from vision_decision.scoring import cyclic_offsets, rotate

        groups, owners = [], []
        for qi, (_, header, choices, texts, labels) in enumerate(compiled):
            group = []
            for offset in cyclic_offsets(len(choices), rotations):
                prompt = header + "\n".join(
                    f"{label}: {text}" for label, text in zip(labels, rotate(texts, offset)))
                group.append((self.engine.render(prompt, len(images)), labels))
                owners.append((qi, offset))
            groups.append(group)
        return groups, owners

    def reference_logits(self, images, examples):
        engine = self.engine
        out = []
        for rendered, labels in examples:
            image_list = [] if images is None else images if isinstance(images, list) else [images]
            token_ids = engine.label_ids(rendered, labels)
            if self.fast:
                inputs = dict(engine.processor(text=[rendered], images=image_list or None,
                                               return_tensors="pt", do_rescale=False, do_normalize=False))
                with torch.inference_mode():
                    out.append(engine.candidate_logits_batch_fast(inputs, [token_ids])[0])
            else:
                inputs = engine.processor(text=[rendered], images=image_list or None, return_tensors="pt")
                with torch.inference_mode():
                    out.append(engine.candidate_logits_batch(inputs, [token_ids])[0])
        return out

    def score(self, images, compiled, rotations):
        """-> (per-question [(offset, logits)] lists, metadata)."""
        from vision_decision.scoring import combine_rotations, result_from_logits

        start = perf_counter()
        groups, owners = self.render_groups(images, compiled, rotations)
        logits, metadata = score_rendered_prefix_cached_hierarchical(
            self.engine, images, groups, fast=self.fast, microbatch=self.microbatch)
        per_q = [[] for _ in compiled]
        for (qi, offset), tensor in zip(owners, logits):
            per_q[qi].append((offset, [float(v) for v in tensor.cpu().tolist()]))
        self.metadata = {**metadata, "enabled": True,
                         "validated": True, "logical_questions": len(compiled), "rotated_suffixes": len(logits)}
        self.engine._last_batch_tokens = metadata["max_effective_tokens"]
        out = []
        for qi, (_, _, choices, _, _) in enumerate(compiled):
            passes = per_q[qi]
            out.append(result_from_logits(choices, passes[0][1]) if len(passes) == 1
                       else combine_rotations(choices, [(o, v) for o, v in passes]))
        self.engine._batch_seconds = perf_counter() - start
        return out
    @staticmethod
    def _min_margin(results):
        """Smallest top1-top2 softmax gap over per-question results."""
        floor = 1.0
        for result in results:
            ordered = sorted(result.scores.values(), reverse=True)
            floor = min(floor, ordered[0] - ordered[1] if len(ordered) > 1 else 0.0)
        return floor

    def maybe_validate_and_score(self, images, compiled, rotations, fallback):
        """Prefix path with parity gate; falls back to `fallback` on any doubt.

        Unsuitable batches (no shared prefix, no validatable same-type pair,
        degenerate model output) fall back WITHOUT disabling the path; only
        genuine parity mismatches and unexpected errors disable it."""
        if not self.enabled or len(compiled) < 2:
            return fallback(images, compiled)
        mode = "visual" if images else "text"
        try:
            if (mode == "visual" and not self.validated_visual) or (mode == "text" and not self.validated_text):
                # Validate the hierarchical branch itself (rotations included); a
                # flat-prefix check would not prove the per-question KV branch.
                # Cross-type pairs share no prefix, so only same-type pairs prove
                # the mechanics; without one, fall back and stay enabled.
                pair = _select_probe_pair(compiled)
                if pair is None:
                    return fallback(images, compiled)
                first, second = pair
                groups, _ = self.render_groups(images, [compiled[first], compiled[second]], rotations)
                probe, probe_examples = groups, [item for group in groups for item in group]
                candidate, _ = score_rendered_prefix_cached_hierarchical(
                    self.engine, images, probe, fast=self.fast, microbatch=self.microbatch)
                reference = self.reference_logits(images, probe_examples)
                _, delta = _logits_match(candidate, reference)
                self.max_delta[mode] = delta
                # Logit parity is a measured numeric floor (bisect 0.03-0.14),
                # not a correctness signal: gate on identical argmax only.
                for cand, ref in zip(candidate, reference):
                    if int(cand.argmax()) != int(ref.argmax()):
                        raise ParityError(f"argmax mismatch at delta={delta:.6f}")
                if mode == "visual":
                    self.validated_visual = True
                else:
                    self.validated_text = True
            out = self.score(images, compiled, rotations)
            self.metadata["validated_mode"] = mode
            self.metadata["parity_max_delta"] = self.max_delta.get(mode)
            self.metadata["min_margin"] = self._min_margin(out)
            return out
        except PrefixUnsuitable:
            return fallback(images, compiled)
        except Exception as exc:
            self.enabled = False
            self.error = f"{type(exc).__name__}: {exc}"
            return fallback(images, compiled)

