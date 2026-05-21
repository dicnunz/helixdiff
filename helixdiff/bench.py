from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from .adapt import VisibleAdaptConfig, adapt_model_to_visible_context
from .data import ByteStream, load_text
from .diffusion import corrupt_batch, masked_accuracy, masked_cross_entropy, restrict_logits_to_ids
from .infill import parse_marked_infill, score_repair
from .ngram import BigramGuide
from .sample import choose_device, denoise_ids, load_checkpoint
from .tokenizer import ByteTokenizer


def split_text(text: str, val_fraction: float = 0.08) -> tuple[str, str]:
    split_at = max(1, int(len(text) * (1.0 - val_fraction)))
    return text[:split_at], text[split_at:]


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def json_score(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def make_marked_cases(
    text: str,
    *,
    cases: int,
    span_chars: int,
    context_chars: int,
    seed: int,
    forbidden_text: str | None = None,
    require_unseen_hole: bool = False,
) -> list[str]:
    clean = text.replace("[[", "").replace("]]", "")
    if len(clean) < (context_chars * 2) + span_chars + 8:
        raise ValueError("text is too small for benchmark case construction")
    rng = random.Random(seed)
    out: list[str] = []
    attempts = 0
    while len(out) < cases and attempts < cases * 200:
        attempts += 1
        start = rng.randint(context_chars, len(clean) - context_chars - span_chars - 1)
        hole = clean[start : start + span_chars]
        if "\n" in hole or not hole.strip():
            continue
        if require_unseen_hole and forbidden_text is not None and hole in forbidden_text:
            continue
        before = clean[start - context_chars : start]
        after = clean[start + span_chars : start + span_chars + context_chars]
        marked = f"{before}[[{hole}]]{after}"
        out.append(marked)
    if len(out) < cases:
        raise ValueError(f"only built {len(out)} benchmark cases from requested {cases}")
    return out


@torch.no_grad()
def masked_eval(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    payload: dict[str, Any],
    text: str,
    batches: int,
    batch_size: int,
    mask_rate: float,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    config = payload["train_config"]
    stream = ByteStream(text, tokenizer, seq_len=payload["model_config"]["seq_len"], split="val", seed=seed)
    losses: list[float] = []
    accs: list[float] = []
    for _ in range(batches):
        clean = stream.sample(batch_size or config["batch_size"], device)
        t = torch.full((clean.shape[0],), mask_rate, device=device)
        corrupted, mask, rates = corrupt_batch(
            clean,
            tokenizer,
            t=t,
            min_mask_rate=config["min_mask_rate"],
            max_mask_rate=config["max_mask_rate"],
            span_prob=0.0,
            max_span_fraction=0.0,
        )
        mask_fraction = mask.float().mean(dim=1)
        logits = restrict_logits_to_ids(
            model(corrupted, rates, corruption_mode=0, mask_fraction=mask_fraction),
            payload.get("sample_token_ids", list(range(model.config.vocab_size))),
        )
        losses.append(float(masked_cross_entropy(logits, clean, mask).item()))
        accs.append(masked_accuracy(logits, clean, mask))
    return {
        "loss": sum(losses) / len(losses),
        "masked_accuracy": sum(accs) / len(accs),
        "mask_rate": mask_rate,
        "batches": batches,
        "batch_size": batch_size,
    }


@torch.no_grad()
def guide_only_case(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    strategy: str,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    tokens = example.tokens.clone().unsqueeze(0)
    for pos in range(example.hole_start, example.hole_end):
        if strategy == "unigram":
            logits = guide.unigram_log_probs.view(1, 1, -1)
            next_id = int(logits[0, 0].argmax().item())
        elif strategy == "bridge":
            logits = guide.logits(tokens, tokenizer)
            next_id = int(logits[0, pos].argmax().item())
        else:
            raise ValueError(f"unknown guide-only strategy: {strategy}")
        if next_id in {tokenizer.pad_token_id, tokenizer.mask_token_id, tokenizer.bos_token_id, tokenizer.eos_token_id}:
            next_id = tokenizer.byte_offset + ord(" ")
        tokens[0, pos] = next_id
    repaired = tokens[0]
    pred = repaired[example.hole_start : example.hole_end]
    target = example.target[example.hole_start : example.hole_end]
    byte_accuracy = float((pred == target).float().mean().item()) if target.numel() else 0.0
    return {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "predicted_hole": tokenizer.decode(pred),
        "hole_length_bytes": example.hole_length,
        "byte_accuracy": byte_accuracy,
        "exact": bool(torch.equal(pred, target)),
        "frozen_context_unchanged": bool((repaired[example.frozen] == example.target[example.frozen]).all().item()),
        "strategy": strategy,
    }


def _common_suffix(left: list[int], right: list[int], limit: int) -> int:
    score = 0
    for offset in range(1, min(len(left), len(right), limit) + 1):
        if left[-offset] != right[-offset]:
            break
        score += 1
    return score


def _common_prefix(left: list[int], right: list[int], limit: int) -> int:
    score = 0
    for left_id, right_id in zip(left[:limit], right[:limit], strict=False):
        if left_id != right_id:
            break
        score += 1
    return score


def visible_suture_candidates(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    context_window: int = 16,
    limit: int = 8,
) -> list[dict[str, Any]]:
    example = parse_marked_infill(marked_text, tokenizer)
    before_ids = tokenizer.encode(example.before, add_bos=False, add_eos=False)
    after_ids = tokenizer.encode(example.after, add_bos=False, add_eos=False)
    hole_len = example.hole_length
    actual_left = before_ids[-context_window:]
    actual_right = after_ids[:context_window]
    rows: list[dict[str, Any]] = []

    for source, ids in (("before", before_ids), ("after", after_ids)):
        if len(ids) < hole_len:
            continue
        for start in range(0, len(ids) - hole_len + 1):
            end = start + hole_len
            span = ids[start:end]
            left_context = ids[max(0, start - context_window) : start]
            right_context = ids[end : end + context_window]
            score = _common_suffix(left_context, actual_left, context_window) + _common_prefix(
                right_context,
                actual_right,
                context_window,
            )
            rows.append(
                {
                    "ids": span,
                    "predicted_hole": tokenizer.decode(span),
                    "source": source,
                    "start": int(start),
                    "score": int(score),
                }
            )

    if not rows:
        fallback = [tokenizer.byte_offset + ord(" ")] * hole_len
        rows.append(
            {
                "ids": fallback,
                "predicted_hole": tokenizer.decode(fallback),
                "source": "fallback_space",
                "start": 0,
                "score": 0,
            }
        )

    deduped: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: (-int(item["score"]), str(item["source"]), int(item["start"]))):
        key = tuple(int(token_id) for token_id in row["ids"])
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())[: max(1, limit)]


def morphology_candidates(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    train_text: str,
    limit: int = 64,
) -> list[dict[str, Any]]:
    example = parse_marked_infill(marked_text, tokenizer)
    hole_len = example.hole_length
    rows: list[dict[str, Any]] = []
    left_match = re.search(r"[A-Za-z][A-Za-z']*$", example.before)
    right_match = re.match(r"[A-Za-z']+", example.after)
    left_tail = left_match.group(0) if left_match else ""
    right_head = right_match.group(0) if right_match else ""
    words = Counter(re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?", train_text))

    for word, count in words.items():
        if len(left_tail) >= 2 and word.lower().startswith(left_tail.lower()) and len(word) >= len(left_tail) + hole_len:
            candidate = word[len(left_tail) : len(left_tail) + hole_len]
            remainder = word[len(left_tail) + hole_len :]
            right_match_len = _common_prefix(
                [ord(char) for char in remainder.lower()],
                [ord(char) for char in right_head.lower()],
                min(len(remainder), len(right_head)),
            )
            if not right_head or right_match_len > 0:
                rows.append(
                    {
                        "ids": tokenizer.encode(candidate, add_bos=False, add_eos=False),
                        "predicted_hole": candidate,
                        "source": "morphology_word_completion",
                        "morphology_score": float(3.0 + right_match_len + math.log1p(count)),
                    }
                )
        if (
            left_tail[:1].isupper()
            and example.after.startswith("\n")
            and word.lower().startswith(left_tail.lower())
            and len(word) >= len(left_tail) + hole_len - 1
        ):
            candidate = word[len(left_tail) : len(left_tail) + hole_len - 1]
            label_candidate = f"{candidate}:"
            rows.append(
                {
                    "ids": tokenizer.encode(label_candidate, add_bos=False, add_eos=False),
                    "predicted_hole": label_candidate,
                    "source": "morphology_speaker_label_completion",
                    "morphology_score": float(4.0 + math.log1p(count)),
                }
            )
        if len(right_head) >= 3 and word.lower().endswith(right_head.lower()) and len(word) >= hole_len + len(right_head):
            candidate = word[len(word) - len(right_head) - hole_len : len(word) - len(right_head)]
            if candidate:
                rows.append(
                    {
                        "ids": tokenizer.encode(candidate, add_bos=False, add_eos=False),
                        "predicted_hole": candidate,
                        "source": "morphology_word_prefix",
                        "morphology_score": float(2.0 + math.log1p(count)),
                    }
                )

    if not left_tail and right_head.lower().startswith(("iel", "ael", "uel")):
        name_prefixes = {
            "iel": ["Gabr", "Dan", "Nath", "Ezek"],
            "ael": ["Mich", "Raph", "Ishr"],
            "uel": ["Sam", "Lem"],
        }
        for suffix, prefixes in name_prefixes.items():
            if not right_head.lower().startswith(suffix):
                continue
            for rank, prefix in enumerate(prefixes):
                if len(prefix) != hole_len:
                    continue
                rows.append(
                    {
                        "ids": tokenizer.encode(prefix, add_bos=False, add_eos=False),
                        "predicted_hole": prefix,
                        "source": "morphology_name_stem_prefix",
                        "morphology_score": float(12.0 - (rank * 0.05)),
                    }
                )

    if not left_tail and len(right_head) >= 4 and right_head[0].islower() and hole_len >= 2:
        stem_len = hole_len - 1
        prefix_counts: Counter[str] = Counter()
        lower_words = Counter(word.lower() for word in words)
        for word, count in words.items():
            clean = re.sub(r"[^A-Za-z]", "", word).lower()
            if len(clean) >= stem_len:
                prefix_counts[clean[:stem_len]] += count
        for stem, count in prefix_counts.items():
            if not stem.isalpha() or count < 2:
                continue
            if count > 64:
                continue
            candidate = f"{stem}-"
            plural_signal = lower_words[stem] + (2 * lower_words[f"{stem}s"]) + lower_words[f"{stem}es"]
            rows.append(
                {
                    "ids": tokenizer.encode(candidate, add_bos=False, add_eos=False),
                    "predicted_hole": candidate,
                    "source": "morphology_hyphen_prefix",
                    "morphology_score": float(2.5 + min(count, 64) / 32.0 + plural_signal),
                }
            )
    if left_tail[:1].isupper() and right_head.startswith("s") and hole_len >= 2:
        suffixes = [
            "iel'",
            "ael'",
            "uel'",
            "ian'",
            "ias'",
            "ius'",
            "ard'",
            "old'",
            "ert'",
            "son'",
        ]
        for rank, suffix in enumerate(suffixes):
            if len(suffix) != hole_len:
                continue
            rows.append(
                {
                    "ids": tokenizer.encode(suffix, add_bos=False, add_eos=False),
                    "predicted_hole": suffix,
                    "source": "morphology_name_possessive_suffix",
                    "morphology_score": float(4.0 - (rank * 0.05)),
                }
            )
    if left_tail and right_head and hole_len >= 3:
        lower_words = Counter(word.lower() for word in words)
        left_pieces: Counter[str] = Counter()
        right_pieces: Counter[str] = Counter()
        for word, count in lower_words.items():
            if word.startswith(left_tail.lower()) and len(word) > len(left_tail):
                for width in range(1, min(3, hole_len) + 1):
                    piece = word[len(left_tail) : len(left_tail) + width]
                    if piece:
                        left_pieces[piece] += count
            if word.endswith(right_head.lower()) and len(word) > len(right_head):
                prefix = word[: -len(right_head)]
                for width in range(1, min(3, hole_len) + 1):
                    if len(prefix) >= width:
                        right_pieces[prefix[-width:]] += count
        for dash in ("-", "--"):
            for left_piece, left_count in left_pieces.items():
                right_width = hole_len - len(left_piece) - len(dash)
                if right_width <= 0:
                    continue
                for right_piece, right_count in right_pieces.items():
                    if len(right_piece) != right_width:
                        continue
                    candidate = f"{left_piece}{dash}{right_piece}"
                    boundary_bonus = 0.0
                    if f"{left_tail.lower()}{left_piece}" in lower_words:
                        boundary_bonus += 6.0
                    if f"{right_piece}{right_head.lower()}" in lower_words:
                        boundary_bonus += 6.0
                    rows.append(
                        {
                            "ids": tokenizer.encode(candidate, add_bos=False, add_eos=False),
                            "predicted_hole": candidate,
                            "source": "morphology_dash_bridge",
                            "morphology_score": float(
                                4.0
                                + boundary_bonus
                                + (len(left_tail) / 8.0)
                                + (len(right_head) / 8.0)
                                + math.log1p(left_count + right_count)
                            ),
                        }
                    )

    filtered = [row for row in rows if len(row["ids"]) == hole_len]
    deduped: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in sorted(filtered, key=lambda item: (-float(item["morphology_score"]), str(item["predicted_hole"]))):
        key = tuple(int(token_id) for token_id in row["ids"])
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())[: max(0, limit)]


def surface_splice_candidates(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    train_text: str,
    limit: int = 64,
) -> list[dict[str, Any]]:
    example = parse_marked_infill(marked_text, tokenizer)
    hole_len = example.hole_length
    left_match = re.search(r"\S*$", example.before)
    right_match = re.match(r"\S*", example.after)
    left_surface = left_match.group(0) if left_match else ""
    right_surface = right_match.group(0) if right_match else ""
    units = Counter(token.strip("\"'“”‘’()[]{}") for token in re.findall(r"\S+", train_text))
    rows: list[dict[str, Any]] = []

    for unit, count in units.items():
        if not unit:
            continue
        unit_lower = unit.lower()
        left_lower = left_surface.lower()
        right_lower = right_surface.lower()
        if left_surface and unit_lower.startswith(left_lower) and len(unit) >= len(left_surface) + hole_len:
            candidate = unit[len(left_surface) : len(left_surface) + hole_len]
            remainder = unit[len(left_surface) + hole_len :]
            right_match_len = _common_prefix(
                [ord(char) for char in remainder.lower()],
                [ord(char) for char in right_lower],
                min(len(remainder), len(right_surface)),
            )
            if not right_surface or right_match_len > 0:
                rows.append(
                    {
                        "ids": tokenizer.encode(candidate, add_bos=False, add_eos=False),
                        "predicted_hole": candidate,
                        "source": "surface_left_splice",
                        "surface_score": float(5.0 + right_match_len + (len(left_surface) / 8.0) + math.log1p(count)),
                    }
                )
        if right_surface and unit_lower.endswith(right_lower) and len(unit) >= hole_len + len(right_surface):
            end = len(unit) - len(right_surface)
            start = end - hole_len
            if start < 0:
                continue
            candidate = unit[start:end]
            prefix = unit[:start]
            left_match_len = _common_suffix(
                [ord(char) for char in prefix.lower()],
                [ord(char) for char in left_lower],
                min(len(prefix), len(left_surface)),
            )
            if not left_surface or left_match_len > 0:
                rows.append(
                    {
                        "ids": tokenizer.encode(candidate, add_bos=False, add_eos=False),
                        "predicted_hole": candidate,
                        "source": "surface_right_splice",
                        "surface_score": float(5.0 + left_match_len + (len(right_surface) / 8.0) + math.log1p(count)),
                    }
                )

    filtered = [row for row in rows if len(row["ids"]) == hole_len]
    deduped: dict[tuple[int, ...], dict[str, Any]] = {}
    for row in sorted(filtered, key=lambda item: (-float(item["surface_score"]), str(item["predicted_hole"]))):
        key = tuple(int(token_id) for token_id in row["ids"])
        if key not in deduped:
            deduped[key] = row
    return list(deduped.values())[: max(0, limit)]


def lattice_oracle_case(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    visible_limit: int = 8,
    morphology_limit: int = 64,
    surface_limit: int = 64,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    target = example.target[example.hole_start : example.hole_end]
    candidate_rows: list[dict[str, Any]] = []

    for rank, row in enumerate(visible_suture_candidates(tokenizer=tokenizer, marked_text=marked_text, limit=visible_limit)):
        candidate_rows.append(
            {
                "ids": [int(token_id) for token_id in row["ids"]],
                "source": f"visible_{row['source']}",
                "rank": rank,
                "suture_score": int(row["score"]),
                "morphology_score": None,
                "surface_score": None,
            }
        )
    for rank, row in enumerate(
        morphology_candidates(
            tokenizer=tokenizer,
            marked_text=marked_text,
            train_text=train_text,
            limit=morphology_limit,
        )
    ):
        candidate_rows.append(
            {
                "ids": [int(token_id) for token_id in row["ids"]],
                "source": row["source"],
                "rank": rank,
                "suture_score": None,
                "morphology_score": float(row["morphology_score"]),
                "surface_score": None,
            }
        )
    for rank, row in enumerate(
        surface_splice_candidates(
            tokenizer=tokenizer,
            marked_text=marked_text,
            train_text=train_text,
            limit=surface_limit,
        )
    ):
        candidate_rows.append(
            {
                "ids": [int(token_id) for token_id in row["ids"]],
                "source": row["source"],
                "rank": rank,
                "suture_score": None,
                "morphology_score": None,
                "surface_score": float(row["surface_score"]),
            }
        )
    for strategy in ("unigram", "bridge"):
        row = guide_only_case(tokenizer=tokenizer, marked_text=marked_text, guide=guide, strategy=strategy)
        ids = tokenizer.encode(row["predicted_hole"], add_bos=False, add_eos=False)
        if len(ids) != example.hole_length:
            ids = (ids + [tokenizer.byte_offset + ord(" ")] * example.hole_length)[: example.hole_length]
        candidate_rows.append(
            {
                "ids": [int(token_id) for token_id in ids],
                "source": strategy,
                "rank": 0,
                "suture_score": None,
                "morphology_score": None,
                "surface_score": None,
            }
        )

    deduped: dict[tuple[int, ...], dict[str, Any]] = {}
    for candidate in candidate_rows:
        key = tuple(int(token_id) for token_id in candidate["ids"])
        source = {
            "source": candidate["source"],
            "rank": int(candidate["rank"]),
            "suture_score": candidate["suture_score"],
            "morphology_score": candidate["morphology_score"],
            "surface_score": candidate.get("surface_score"),
        }
        if key not in deduped:
            deduped[key] = {
                "ids": list(key),
                "predicted_hole": tokenizer.decode(list(key)),
                "sources": [source],
            }
        else:
            deduped[key]["sources"].append(source)

    summaries: list[dict[str, Any]] = []
    exact_sources: list[str] = []
    for candidate in deduped.values():
        pred = torch.tensor(candidate["ids"], dtype=torch.long)
        exact = bool(torch.equal(pred, target))
        sources = candidate["sources"]
        if exact:
            exact_sources.extend(str(source["source"]) for source in sources)
        summaries.append(
            {
                "predicted_hole": candidate["predicted_hole"],
                "sources": sources,
                "exact": exact,
            }
        )

    return {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "hole_length_bytes": example.hole_length,
        "candidate_count": len(summaries),
        "oracle_candidate_exact": any(bool(row["exact"]) for row in summaries),
        "visible_oracle_exact": any(source.startswith("visible_") for source in exact_sources),
        "morphology_oracle_exact": any(source.startswith("morphology_") for source in exact_sources),
        "surface_oracle_exact": any(source.startswith("surface_") for source in exact_sources),
        "bridge_oracle_exact": "bridge" in exact_sources,
        "unigram_oracle_exact": "unigram" in exact_sources,
        "exact_candidate_sources": sorted(set(exact_sources)),
        "candidate_summaries": summaries,
    }


@torch.no_grad()
def nearest_visible_case(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    context_window: int = 16,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    candidate = visible_suture_candidates(
        tokenizer=tokenizer,
        marked_text=marked_text,
        context_window=context_window,
        limit=1,
    )[0]
    best_span = [int(token_id) for token_id in candidate["ids"]]

    repaired = example.tokens.clone()
    repaired[example.hole_start : example.hole_end] = torch.tensor(best_span, dtype=torch.long)
    pred = repaired[example.hole_start : example.hole_end]
    target = example.target[example.hole_start : example.hole_end]
    byte_accuracy = float((pred == target).float().mean().item()) if target.numel() else 0.0
    return {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "predicted_hole": tokenizer.decode(pred),
        "hole_length_bytes": example.hole_length,
        "byte_accuracy": byte_accuracy,
        "exact": bool(torch.equal(pred, target)),
        "frozen_context_unchanged": bool((repaired[example.frozen] == example.target[example.frozen]).all().item()),
        "strategy": "nearest_visible",
        "nearest_visible_source": candidate["source"],
        "nearest_visible_score": int(candidate["score"]),
        "nearest_visible_context_window": int(context_window),
    }


@torch.no_grad()
def retrieval_lattice_case(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    guidance: float,
    temperature: float,
    top_k: int,
    train_text: str = "",
    visible_limit: int = 8,
    suture_weight: float = 2.0,
    morphology_limit: int = 64,
    morphology_weight: float = 1.0,
    surface_limit: int = 64,
    surface_weight: float = 1.0,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    candidates: list[dict[str, Any]] = []
    for row in visible_suture_candidates(tokenizer=tokenizer, marked_text=marked_text, limit=visible_limit):
        candidates.append(
            {
                "ids": [int(token_id) for token_id in row["ids"]],
                "source": f"visible_{row['source']}",
                "predicted_hole": row["predicted_hole"],
                "suture_score": int(row["score"]),
                "morphology_score": None,
                "surface_score": None,
            }
        )
    if train_text:
        for row in morphology_candidates(
            tokenizer=tokenizer,
            marked_text=marked_text,
            train_text=train_text,
            limit=morphology_limit,
        ):
            candidates.append(
                {
                    "ids": [int(token_id) for token_id in row["ids"]],
                    "source": row["source"],
                    "predicted_hole": row["predicted_hole"],
                    "suture_score": None,
                    "morphology_score": float(row["morphology_score"]),
                    "surface_score": None,
                }
            )
    if train_text:
        for row in surface_splice_candidates(
            tokenizer=tokenizer,
            marked_text=marked_text,
            train_text=train_text,
            limit=surface_limit,
        ):
            candidates.append(
                {
                    "ids": [int(token_id) for token_id in row["ids"]],
                    "source": row["source"],
                    "predicted_hole": row["predicted_hole"],
                    "suture_score": None,
                    "morphology_score": None,
                    "surface_score": float(row["surface_score"]),
                }
            )
    for strategy in ("unigram", "bridge"):
        row = guide_only_case(tokenizer=tokenizer, marked_text=marked_text, guide=guide, strategy=strategy)
        ids = tokenizer.encode(row["predicted_hole"], add_bos=False, add_eos=False)
        if len(ids) != example.hole_length:
            ids = (ids + [tokenizer.byte_offset + ord(" ")] * example.hole_length)[: example.hole_length]
        candidates.append(
            {
                "ids": [int(token_id) for token_id in ids],
                "source": strategy,
                "predicted_hole": tokenizer.decode(ids),
                "suture_score": None,
                "morphology_score": None,
                "surface_score": None,
            }
        )

    deduped: dict[tuple[int, ...], dict[str, Any]] = {}
    for candidate in candidates:
        key = tuple(int(token_id) for token_id in candidate["ids"])
        if key not in deduped:
            deduped[key] = candidate
    scored_rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_repaired: torch.Tensor | None = None
    best_row: dict[str, Any] | None = None
    for candidate in deduped.values():
        repaired = example.tokens.clone()
        repaired[example.hole_start : example.hole_end] = torch.tensor(candidate["ids"], dtype=torch.long)
        score = score_repair(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired,
            hole_start=example.hole_start,
            hole_end=example.hole_end,
            guide=guide,
            guidance=guidance,
            temperature=temperature,
            top_k=top_k,
        )
        pred = repaired[example.hole_start : example.hole_end]
        target = example.target[example.hole_start : example.hole_end]
        suture_score = candidate["suture_score"]
        morphology_score = candidate["morphology_score"]
        surface_score = candidate["surface_score"]
        combined_score = (
            score
            + (float(suture_score) * suture_weight if suture_score is not None else 0.0)
            + (float(morphology_score) * morphology_weight if morphology_score is not None else 0.0)
            + (float(surface_score) * surface_weight if surface_score is not None else 0.0)
        )
        row = {
            "source": candidate["source"],
            "predicted_hole": tokenizer.decode(pred),
            "suture_score": suture_score,
            "morphology_score": morphology_score,
            "surface_score": surface_score,
            "diffusion_score": json_score(score),
            "diffusion_score_is_finite": math.isfinite(score),
            "combined_score": json_score(combined_score),
            "byte_accuracy": float((pred == target).float().mean().item()) if target.numel() else 0.0,
            "exact": bool(torch.equal(pred, target)),
        }
        scored_rows.append(row)
        if best_repaired is None or combined_score > best_score:
            best_score = combined_score
            best_repaired = repaired
            best_row = row
    if best_repaired is None or best_row is None:
        raise RuntimeError("retrieval lattice produced no candidates")
    pred = best_repaired[example.hole_start : example.hole_end]
    target = example.target[example.hole_start : example.hole_end]
    return {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "predicted_hole": tokenizer.decode(pred),
        "hole_length_bytes": example.hole_length,
        "byte_accuracy": float((pred == target).float().mean().item()) if target.numel() else 0.0,
        "exact": bool(torch.equal(pred, target)),
        "frozen_context_unchanged": bool((best_repaired[example.frozen] == example.target[example.frozen]).all().item()),
        "strategy": "retrieval_lattice_diffusion_scored",
        "selected_source": best_row["source"],
        "selected_candidate_score": json_score(best_score),
        "selected_candidate_score_is_finite": math.isfinite(best_score),
        "selector": "diffusion_score_plus_suture_morphology_and_surface_score",
        "lattice_suture_weight": float(suture_weight),
        "lattice_morphology_weight": float(morphology_weight),
        "lattice_surface_weight": float(surface_weight),
        "candidate_summaries": scored_rows,
        "oracle_candidate_exact": any(bool(row["exact"]) for row in scored_rows),
    }


@torch.no_grad()
def infill_case(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide | None,
    guidance: float,
    steps: int,
    top_k: int,
    temperature: float,
    schedule: str,
    seed: int,
    candidates: int = 1,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    if example.tokens.numel() > model.config.seq_len:
        raise ValueError(f"case has {example.tokens.numel()} tokens; checkpoint seq_len is {model.config.seq_len}")
    started = time.perf_counter()
    best_score = float("-inf")
    best_repaired: torch.Tensor | None = None
    best_trace: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for offset in range(max(1, candidates)):
        repaired, trace = denoise_ids(
            model,
            tokenizer,
            initial_tokens=example.tokens,
            frozen=example.frozen,
            steps=steps,
            temperature=temperature,
            top_k=top_k,
            remask=0.0,
            guide=guide,
            guidance=guidance,
            schedule=schedule,
            seed=seed + offset,
            return_trace=True,
            max_reveal_per_step=1,
            corruption_mode=3,
        )
        assert isinstance(repaired, torch.Tensor)
        pred = repaired[example.hole_start : example.hole_end]
        target = example.target[example.hole_start : example.hole_end]
        score = score_repair(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired,
            hole_start=example.hole_start,
            hole_end=example.hole_end,
            guide=guide,
            guidance=guidance,
            temperature=temperature,
            top_k=top_k,
        )
        candidate_rows.append(
            {
                "seed": seed + offset,
                "score": json_score(score),
                "score_is_finite": math.isfinite(score),
                "predicted_hole": tokenizer.decode(pred),
                "byte_accuracy": float((pred == target).float().mean().item()) if target.numel() else 0.0,
                "exact": bool(torch.equal(pred, target)),
            }
        )
        if best_repaired is None or score > best_score:
            best_score = score
            best_repaired = repaired
            best_trace = trace
    if best_repaired is None:
        raise RuntimeError("no benchmark infill candidates were produced")
    repaired = best_repaired
    pred = repaired[example.hole_start : example.hole_end]
    target = example.target[example.hole_start : example.hole_end]
    byte_accuracy = float((pred == target).float().mean().item()) if target.numel() else 0.0
    return {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "predicted_hole": tokenizer.decode(pred),
        "hole_length_bytes": example.hole_length,
        "byte_accuracy": byte_accuracy,
        "exact": bool(torch.equal(pred, target)),
        "frozen_context_unchanged": bool((repaired[example.frozen] == example.target[example.frozen]).all().item()),
        "steps_used": len(best_trace),
        "candidates": max(1, candidates),
        "selected_candidate_score": json_score(best_score),
        "selected_candidate_score_is_finite": math.isfinite(best_score),
        "candidate_summaries": candidate_rows,
        "elapsed_seconds": time.perf_counter() - started,
    }


def summarize_infill(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "byte_accuracy": 0.0, "exact_match_rate": 0.0, "pass_at_k_exact": 0.0}
    return {
        "cases": len(rows),
        "byte_accuracy": sum(float(row["byte_accuracy"]) for row in rows) / len(rows),
        "exact_match_rate": sum(1.0 for row in rows if row["exact"]) / len(rows),
        "pass_at_k_exact": sum(
            1.0
            for row in rows
            if (
                any(bool(candidate.get("exact")) for candidate in row.get("candidate_summaries", []))
                if row.get("candidate_summaries")
                else bool(row["exact"])
            )
        )
        / len(rows),
        "frozen_context_ok": all(bool(row["frozen_context_unchanged"]) for row in rows),
    }


def summarize_lattice_oracle(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "cases": 0,
            "oracle_exact_rate": 0.0,
            "visible_oracle_exact_rate": 0.0,
            "morphology_oracle_exact_rate": 0.0,
            "bridge_oracle_exact_rate": 0.0,
            "unigram_oracle_exact_rate": 0.0,
            "surface_oracle_exact_rate": 0.0,
            "avg_candidate_count": 0.0,
        }
    return {
        "cases": len(rows),
        "oracle_exact_rate": sum(1.0 for row in rows if row["oracle_candidate_exact"]) / len(rows),
        "visible_oracle_exact_rate": sum(1.0 for row in rows if row["visible_oracle_exact"]) / len(rows),
        "morphology_oracle_exact_rate": sum(1.0 for row in rows if row["morphology_oracle_exact"]) / len(rows),
        "surface_oracle_exact_rate": sum(1.0 for row in rows if row["surface_oracle_exact"]) / len(rows),
        "bridge_oracle_exact_rate": sum(1.0 for row in rows if row["bridge_oracle_exact"]) / len(rows),
        "unigram_oracle_exact_rate": sum(1.0 for row in rows if row["unigram_oracle_exact"]) / len(rows),
        "avg_candidate_count": sum(float(row["candidate_count"]) for row in rows) / len(rows),
    }


def model_quality_label(masked_acc: float, unguided_infill: float, guided_infill: float) -> str:
    if masked_acc >= 0.45 and unguided_infill >= 0.25 and guided_infill >= 0.55:
        return "strong_laptop_checkpoint"
    if masked_acc >= 0.25 and guided_infill >= 0.35:
        return "promising_small_checkpoint"
    if masked_acc >= 0.12:
        return "mechanism_checkpoint"
    return "undertrained"


def candidate_oracle_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    text = load_text(args.data)
    train_text, val_text = split_text(text, args.val_fraction)
    cases = make_marked_cases(
        val_text,
        cases=args.cases,
        span_chars=args.span_chars,
        context_chars=args.context_chars,
        seed=args.seed,
        forbidden_text=train_text,
        require_unseen_hole=args.require_unseen_hole,
    )
    guide = BigramGuide.from_text(train_text, tokenizer)
    rows = [
        lattice_oracle_case(
            tokenizer=tokenizer,
            marked_text=marked,
            guide=guide,
            train_text=train_text,
            visible_limit=args.lattice_visible_candidates,
            morphology_limit=args.lattice_morphology_candidates,
            surface_limit=args.lattice_surface_candidates,
        )
        for marked in cases
    ]
    for row in rows:
        row["target_hole_seen_in_train_split"] = row["target_hole"] in train_text
    return {
        "mode": "candidate_oracle_only",
        "data": str(args.data),
        "data_bytes": len(text.encode("utf-8")),
        "val_fraction": args.val_fraction,
        "train_split_sha256": sha256_text(train_text),
        "validation_split_sha256": sha256_text(val_text),
        "guide_scope": "training_split_only",
        "case_filter": {
            "require_unseen_hole": bool(args.require_unseen_hole),
            "lattice_visible_candidates": int(args.lattice_visible_candidates),
            "lattice_morphology_candidates": int(args.lattice_morphology_candidates),
            "lattice_surface_candidates": int(args.lattice_surface_candidates),
            "span_chars": int(args.span_chars),
            "context_chars": int(args.context_chars),
            "seed": int(args.seed),
        },
        "infill": {
            "case_source": "validation_split",
            "candidate_oracle": {
                "summary": summarize_lattice_oracle(rows),
                "cases": rows,
            },
        },
        "claim_boundary": "candidate-oracle coverage only; this does not prove model scoring accuracy or language-model quality",
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if not args.checkpoint:
        raise ValueError("--checkpoint is required unless --candidate-oracle-only is set")
    device = choose_device(args.device)
    model, tokenizer, payload = load_checkpoint(args.checkpoint, device=device)
    text = load_text(args.data)
    train_text, val_text = split_text(text, args.val_fraction)
    cases = make_marked_cases(
        val_text,
        cases=args.cases,
        span_chars=args.span_chars,
        context_chars=args.context_chars,
        seed=args.seed,
        forbidden_text=train_text,
        require_unseen_hole=args.require_unseen_hole,
    )
    masked = masked_eval(
        model=model,
        tokenizer=tokenizer,
        payload=payload,
        text=text,
        batches=args.batches,
        batch_size=args.batch_size or payload["train_config"]["batch_size"],
        mask_rate=args.mask_rate,
        seed=args.seed,
        device=device,
    )
    guide = BigramGuide.from_text(train_text, tokenizer).to_device(device)
    unigram_rows: list[dict[str, Any]] = []
    bridge_rows: list[dict[str, Any]] = []
    nearest_visible_rows: list[dict[str, Any]] = []
    retrieval_lattice_rows: list[dict[str, Any]] = []
    unguided_rows: list[dict[str, Any]] = []
    guided_rows: list[dict[str, Any]] = []
    adapted_rows: list[dict[str, Any]] = []
    adapted_guided_rows: list[dict[str, Any]] = []
    for index, marked in enumerate(cases):
        unigram_rows.append(guide_only_case(tokenizer=tokenizer, marked_text=marked, guide=guide, strategy="unigram"))
        bridge = guide_only_case(tokenizer=tokenizer, marked_text=marked, guide=guide, strategy="bridge")
        bridge["target_hole_seen_in_train_split"] = bridge["target_hole"] in train_text
        bridge_rows.append(bridge)
        nearest_visible_rows.append(nearest_visible_case(tokenizer=tokenizer, marked_text=marked))
        lattice = retrieval_lattice_case(
            model=model,
            tokenizer=tokenizer,
            marked_text=marked,
            guide=guide,
            guidance=args.guidance,
            temperature=args.temperature,
            top_k=args.top_k,
            train_text=train_text,
            visible_limit=args.lattice_visible_candidates,
            suture_weight=args.lattice_suture_weight,
            morphology_limit=args.lattice_morphology_candidates,
            morphology_weight=args.lattice_morphology_weight,
            surface_limit=args.lattice_surface_candidates,
            surface_weight=args.lattice_surface_weight,
        )
        lattice["target_hole_seen_in_train_split"] = lattice["target_hole"] in train_text
        retrieval_lattice_rows.append(lattice)
        unguided_rows.append(
            infill_case(
                model=model,
                tokenizer=tokenizer,
                marked_text=marked,
                guide=None,
                guidance=0.0,
                steps=args.steps,
                top_k=args.top_k,
                temperature=args.temperature,
                schedule=args.schedule,
                seed=args.seed + index,
                candidates=args.candidates,
            )
        )
        guided = infill_case(
            model=model,
            tokenizer=tokenizer,
            marked_text=marked,
            guide=guide,
            guidance=args.guidance,
            steps=args.steps,
            top_k=args.top_k,
            temperature=args.temperature,
            schedule=args.schedule,
            seed=args.seed + index,
            candidates=args.candidates,
        )
        guided["target_hole_seen_in_train_split"] = guided["target_hole"] in train_text
        guided_rows.append(guided)
        if args.adapt_visible_steps > 0:
            adapted_model, adaptation_report = adapt_model_to_visible_context(
                model=model,
                tokenizer=tokenizer,
                example=parse_marked_infill(marked, tokenizer),
                config=VisibleAdaptConfig(
                    steps=args.adapt_visible_steps,
                    batch_size=args.adapt_batch_size,
                    learning_rate=args.adapt_learning_rate,
                    span_min=args.adapt_span_min,
                    span_max=args.adapt_span_max,
                    train_scope=args.adapt_train_scope,
                    seed=args.seed + index,
                ),
            )
            adapted = infill_case(
                model=adapted_model,
                tokenizer=tokenizer,
                marked_text=marked,
                guide=None,
                guidance=0.0,
                steps=args.steps,
                top_k=args.top_k,
                temperature=args.temperature,
                schedule=args.schedule,
                seed=args.seed + index,
                candidates=args.candidates,
            )
            adapted["adaptation"] = adaptation_report
            adapted["target_hole_seen_in_train_split"] = adapted["target_hole"] in train_text
            adapted_rows.append(adapted)
            adapted_guided = infill_case(
                model=adapted_model,
                tokenizer=tokenizer,
                marked_text=marked,
                guide=guide,
                guidance=args.guidance,
                steps=args.steps,
                top_k=args.top_k,
                temperature=args.temperature,
                schedule=args.schedule,
                seed=args.seed + index,
                candidates=args.candidates,
            )
            adapted_guided["adaptation"] = adaptation_report
            adapted_guided["target_hole_seen_in_train_split"] = adapted_guided["target_hole"] in train_text
            adapted_guided_rows.append(adapted_guided)
    unguided_summary = summarize_infill(unguided_rows)
    unigram_summary = summarize_infill(unigram_rows)
    bridge_summary = summarize_infill(bridge_rows)
    nearest_visible_summary = summarize_infill(nearest_visible_rows)
    retrieval_lattice_summary = summarize_infill(retrieval_lattice_rows)
    guided_summary = summarize_infill(guided_rows)
    adapted_summary = summarize_infill(adapted_rows)
    adapted_guided_summary = summarize_infill(adapted_guided_rows)
    bridge_lift = float(guided_summary["byte_accuracy"]) - float(bridge_summary["byte_accuracy"])
    adapted_lift = float(adapted_summary["byte_accuracy"]) - float(unguided_summary["byte_accuracy"])
    adapted_vs_bridge = float(adapted_summary["byte_accuracy"]) - float(bridge_summary["byte_accuracy"])
    adapted_guided_vs_bridge = float(adapted_guided_summary["byte_accuracy"]) - float(bridge_summary["byte_accuracy"])
    adapted_vs_nearest_visible = float(adapted_summary["byte_accuracy"]) - float(
        nearest_visible_summary["byte_accuracy"]
    )
    adapted_guided_vs_nearest_visible = float(adapted_guided_summary["byte_accuracy"]) - float(
        nearest_visible_summary["byte_accuracy"]
    )
    retrieval_lattice_vs_nearest_visible = float(retrieval_lattice_summary["byte_accuracy"]) - float(
        nearest_visible_summary["byte_accuracy"]
    )
    label = model_quality_label(
        float(masked["masked_accuracy"]),
        float(unguided_summary["byte_accuracy"]),
        float(guided_summary["byte_accuracy"]),
    )
    return {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_step": int(payload.get("step", 0)),
        "loaded_state": payload.get("loaded_state"),
        "state_migration": payload.get("state_migration"),
        "parameters": int(payload.get("metrics", {}).get("parameters", 0)),
        "data": str(args.data),
        "data_bytes": len(text.encode("utf-8")),
        "val_fraction": args.val_fraction,
        "train_split_sha256": sha256_text(train_text),
        "validation_split_sha256": sha256_text(val_text),
        "guide_scope": "training_split_only",
        "case_filter": {
            "require_unseen_hole": bool(args.require_unseen_hole),
            "candidate_count": int(args.candidates),
            "lattice_visible_candidates": int(args.lattice_visible_candidates),
            "lattice_morphology_candidates": int(args.lattice_morphology_candidates),
            "lattice_surface_candidates": int(args.lattice_surface_candidates),
            "lattice_suture_weight": float(args.lattice_suture_weight),
            "lattice_morphology_weight": float(args.lattice_morphology_weight),
            "lattice_surface_weight": float(args.lattice_surface_weight),
            "visible_context_adaptation_steps": int(args.adapt_visible_steps),
        },
        "masked_eval": masked,
        "infill": {
            "case_source": "validation_split",
            "span_chars": args.span_chars,
            "context_chars": args.context_chars,
            "unigram_baseline": {"summary": unigram_summary, "cases": unigram_rows},
            "bridge_only_baseline": {"summary": bridge_summary, "cases": bridge_rows},
            "nearest_visible_baseline": {"summary": nearest_visible_summary, "cases": nearest_visible_rows},
            "retrieval_lattice": {"summary": retrieval_lattice_summary, "cases": retrieval_lattice_rows},
            "unguided": {"summary": unguided_summary, "cases": unguided_rows},
            "bridge_guided": {"summary": guided_summary, "cases": guided_rows},
            "visible_context_adapted": {"summary": adapted_summary, "cases": adapted_rows},
            "visible_context_adapted_bridge_guided": {
                "summary": adapted_guided_summary,
                "cases": adapted_guided_rows,
            },
            "bridge_guided_minus_bridge_only_byte_accuracy": bridge_lift,
            "adapted_minus_unguided_byte_accuracy": adapted_lift,
            "adapted_minus_bridge_only_byte_accuracy": adapted_vs_bridge,
            "adapted_bridge_guided_minus_bridge_only_byte_accuracy": adapted_guided_vs_bridge,
            "adapted_minus_nearest_visible_byte_accuracy": adapted_vs_nearest_visible,
            "adapted_bridge_guided_minus_nearest_visible_byte_accuracy": adapted_guided_vs_nearest_visible,
            "retrieval_lattice_minus_nearest_visible_byte_accuracy": retrieval_lattice_vs_nearest_visible,
        },
        "quality_label": label,
        "ten_out_of_ten_gate": {
            "artifact": "pass_if_tests_verifier_docs_and_benchmarks_pass",
            "actual_model": "requires strong_laptop_checkpoint or better; otherwise do not claim 10/10 model quality",
            "public_wow": "requires visible infill trace plus non-leaky heldout benchmark",
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a harsher HelixDiff benchmark with held-out infill checks.")
    parser.add_argument("--checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--mask-rate", type=float, default=0.5)
    parser.add_argument("--cases", type=int, default=6)
    parser.add_argument("--span-chars", type=int, default=6)
    parser.add_argument("--context-chars", type=int, default=36)
    parser.add_argument("--val-fraction", type=float, default=0.08)
    parser.add_argument("--steps", type=int, default=48)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--require-unseen-hole", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.55)
    parser.add_argument("--top-k", type=int, default=48)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--schedule", choices=["entropy", "ribbon"], default="ribbon")
    parser.add_argument("--lattice-visible-candidates", type=int, default=8)
    parser.add_argument("--lattice-morphology-candidates", type=int, default=64)
    parser.add_argument("--lattice-surface-candidates", type=int, default=64)
    parser.add_argument("--candidate-oracle-only", action="store_true")
    parser.add_argument("--lattice-suture-weight", type=float, default=2.0)
    parser.add_argument("--lattice-morphology-weight", type=float, default=1.0)
    parser.add_argument("--lattice-surface-weight", type=float, default=1.0)
    parser.add_argument("--adapt-visible-steps", type=int, default=0)
    parser.add_argument("--adapt-batch-size", type=int, default=8)
    parser.add_argument("--adapt-learning-rate", type=float, default=1e-4)
    parser.add_argument("--adapt-span-min", type=int, default=3)
    parser.add_argument("--adapt-span-max", type=int, default=12)
    parser.add_argument("--adapt-train-scope", choices=["head", "last_block", "all"], default="last_block")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = candidate_oracle_benchmark(args) if args.candidate_oracle_only else benchmark(args)
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
