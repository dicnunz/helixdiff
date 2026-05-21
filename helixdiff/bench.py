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
    result = {
        "marked_text": marked_text,
        "target_hole": example.hole,
        "predicted_hole": tokenizer.decode(pred),
        "hole_length_bytes": example.hole_length,
        "byte_accuracy": byte_accuracy,
        "exact": bool(torch.equal(pred, target)),
        "frozen_context_unchanged": bool((repaired[example.frozen] == example.target[example.frozen]).all().item()),
        "strategy": strategy,
    }
    return result


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


def lattice_source_prior_score(
    source: dict[str, Any],
    *,
    suture_weight: float = 2.0,
    morphology_weight: float = 1.0,
    surface_weight: float = 1.0,
) -> float:
    score = 0.0
    if source.get("suture_score") is not None:
        score += float(source["suture_score"]) * suture_weight
    if source.get("morphology_score") is not None:
        score += float(source["morphology_score"]) * morphology_weight
    if source.get("surface_score") is not None:
        score += float(source["surface_score"]) * surface_weight
    if source.get("source") == "bridge":
        score += 0.25
    rank = source.get("rank")
    if isinstance(rank, int):
        score -= rank * 0.01
    return score


def lattice_candidate_prior_score(
    sources: list[dict[str, Any]],
    *,
    suture_weight: float = 2.0,
    morphology_weight: float = 1.0,
    surface_weight: float = 1.0,
) -> tuple[float, str | None]:
    if not sources:
        return float("-inf"), None
    scored = [
        (
            lattice_source_prior_score(
                source,
                suture_weight=suture_weight,
                morphology_weight=morphology_weight,
                surface_weight=surface_weight,
            ),
            str(source.get("source")),
        )
        for source in sources
    ]
    return max(scored, key=lambda item: (item[0], item[1]))


def build_lattice_candidate_rows(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str = "",
    visible_limit: int = 8,
    morphology_limit: int = 64,
    surface_limit: int = 64,
) -> list[dict[str, Any]]:
    example = parse_marked_infill(marked_text, tokenizer)
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
    if train_text:
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
    return list(deduped.values())


def rank_lattice_candidates_by_prior(
    candidates: list[dict[str, Any]],
    *,
    suture_weight: float = 2.0,
    morphology_weight: float = 1.0,
    surface_weight: float = 1.0,
) -> list[dict[str, Any]]:
    ranked: list[dict[str, Any]] = []
    for candidate in candidates:
        prior_score, prior_source = lattice_candidate_prior_score(
            candidate["sources"],
            suture_weight=suture_weight,
            morphology_weight=morphology_weight,
            surface_weight=surface_weight,
        )
        ranked.append(
            {
                **candidate,
                "prior_score": json_score(prior_score),
                "prior_source": prior_source,
            }
        )
    ranked.sort(
        key=lambda row: (
            -(float(row["prior_score"]) if row["prior_score"] is not None else float("-inf")),
            str(row["predicted_hole"]),
        )
    )
    for rank, row in enumerate(ranked):
        row["prior_rank"] = rank
    return ranked


def score_lattice_verifier(
    *,
    model: torch.nn.Module,
    tokenizer: ByteTokenizer,
    repaired_ids: torch.Tensor,
    hole_start: int,
    hole_end: int,
    guide: BigramGuide | None,
    guidance: float,
    temperature: float,
    top_k: int,
    verifier_mode: str,
) -> tuple[float, dict[str, float | None]]:
    if verifier_mode == "dual":
        modes = ("suture_loo", "full_hole")
    elif verifier_mode in {"suture_loo", "full_hole"}:
        modes = (verifier_mode,)
    else:
        raise ValueError(f"unknown lattice verifier mode: {verifier_mode}")

    scores: dict[str, float | None] = {}
    finite_scores: list[float] = []
    for mode in modes:
        score = score_repair(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired_ids,
            hole_start=hole_start,
            hole_end=hole_end,
            guide=guide,
            guidance=guidance,
            temperature=temperature,
            top_k=top_k,
            mode=mode,
        )
        scores[mode] = json_score(score)
        if math.isfinite(score):
            finite_scores.append(score)
    if not finite_scores:
        return float("-inf"), scores
    return sum(finite_scores) / len(finite_scores), scores


def classify_selector_effect(*, raw_best_exact: bool, anchor_exact: bool, margin_applied: bool) -> str:
    if margin_applied:
        if anchor_exact and not raw_best_exact:
            return "margin_rescued_exact_anchor"
        if raw_best_exact and not anchor_exact:
            return "margin_blocked_exact_raw"
        if anchor_exact:
            return "margin_kept_exact_anchor"
        return "margin_kept_prior_anchor"
    if raw_best_exact:
        return "raw_verifier_selected_exact"
    if anchor_exact:
        return "raw_verifier_overrode_exact_anchor"
    return "raw_verifier_selected_nonexact"


def format_selector_margin(margin: float) -> str:
    return f"{float(margin):g}"


def parse_selector_margin_sweep(value: str) -> list[float]:
    margins: list[float] = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        margins.append(float(stripped))
    return sorted(set(margins))


def parse_prior_weight_grid(value: str) -> list[float]:
    weights: list[float] = []
    for chunk in value.split(","):
        stripped = chunk.strip()
        if not stripped:
            continue
        weight = float(stripped)
        if weight < 0:
            raise ValueError("prior weights must be non-negative")
        weights.append(weight)
    return sorted(set(weights))


def visible_self_calibration_cases(
    marked_text: str,
    *,
    span_chars: int,
    context_chars: int = 12,
    limit: int = 12,
) -> list[str]:
    before, rest = marked_text.split("[[", 1)
    _, after = rest.split("]]", 1)
    rows: list[tuple[int, str]] = []
    for side, segment in (("before", before), ("after", after)):
        if len(segment) < span_chars + 2:
            continue
        local_context = max(1, min(context_chars, (len(segment) - span_chars) // 2))
        if len(segment) < (local_context * 2) + span_chars:
            continue
        for start in range(local_context, len(segment) - local_context - span_chars + 1):
            hole = segment[start : start + span_chars]
            if "\n" in hole or not hole.strip():
                continue
            left = segment[start - local_context : start]
            right = segment[start + span_chars : start + span_chars + local_context]
            if not left or not right:
                continue
            boundary_distance = (len(segment) - start) if side == "before" else start
            rows.append((int(boundary_distance), f"{left}[[{hole}]]{right}"))
    deduped: dict[str, int] = {}
    for distance, marked in sorted(rows, key=lambda item: (item[0], item[1])):
        deduped.setdefault(marked, distance)
    return list(deduped.keys())[: max(0, limit)]


def exact_rank_for_prior_weights(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    visible_limit: int,
    morphology_limit: int,
    surface_limit: int,
    suture_weight: float,
    morphology_weight: float,
    surface_weight: float,
) -> tuple[int | None, int]:
    example = parse_marked_infill(marked_text, tokenizer)
    target = example.target[example.hole_start : example.hole_end]
    ranked = rank_lattice_candidates_by_prior(
        build_lattice_candidate_rows(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
        ),
        suture_weight=suture_weight,
        morphology_weight=morphology_weight,
        surface_weight=surface_weight,
    )
    ranks = [
        int(candidate["prior_rank"])
        for candidate in ranked
        if torch.equal(torch.tensor(candidate["ids"], dtype=torch.long), target)
    ]
    return (min(ranks) if ranks else None), len(ranked)


def evaluate_prior_weight_combo(
    *,
    tokenizer: ByteTokenizer,
    calibration_cases: list[str],
    guide: BigramGuide,
    train_text: str,
    visible_limit: int,
    morphology_limit: int,
    surface_limit: int,
    suture_weight: float,
    morphology_weight: float,
    surface_weight: float,
) -> dict[str, Any]:
    ranks: list[int | None] = []
    candidate_counts: list[int] = []
    for marked in calibration_cases:
        rank, candidate_count = exact_rank_for_prior_weights(
            tokenizer=tokenizer,
            marked_text=marked,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
            suture_weight=suture_weight,
            morphology_weight=morphology_weight,
            surface_weight=surface_weight,
        )
        ranks.append(rank)
        candidate_counts.append(candidate_count)
    exact_ranks = [rank for rank in ranks if rank is not None]
    cases = len(calibration_cases)
    return {
        "suture_weight": float(suture_weight),
        "morphology_weight": float(morphology_weight),
        "surface_weight": float(surface_weight),
        "cases": cases,
        "oracle_exact_rate": (len(exact_ranks) / cases) if cases else 0.0,
        "prior_top1_exact_rate": (sum(1 for rank in exact_ranks if rank == 0) / cases) if cases else 0.0,
        "prior_top4_exact_rate": (sum(1 for rank in exact_ranks if rank < 4) / cases) if cases else 0.0,
        "avg_prior_exact_rank": (sum(float(rank) for rank in exact_ranks) / len(exact_ranks)) if exact_ranks else None,
        "avg_candidate_count": (sum(float(count) for count in candidate_counts) / cases) if cases else 0.0,
    }


def calibrate_lattice_prior_weights_on_visible_context(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    visible_limit: int,
    morphology_limit: int,
    surface_limit: int,
    default_suture_weight: float,
    default_morphology_weight: float,
    default_surface_weight: float,
    weight_grid: list[float],
    calibration_cases: int = 12,
    calibration_context_chars: int = 12,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    cases = visible_self_calibration_cases(
        marked_text,
        span_chars=len(example.hole),
        context_chars=calibration_context_chars,
        limit=calibration_cases,
    )
    if not cases or not weight_grid:
        return {
            "enabled": True,
            "status": "no_visible_calibration_cases",
            "calibration_cases": len(cases),
            "selected_weights": {
                "suture_weight": float(default_suture_weight),
                "morphology_weight": float(default_morphology_weight),
                "surface_weight": float(default_surface_weight),
            },
            "grid_rows": [],
            "claim_boundary": "visible self-calibration is per-case calibration, not held-out model proof",
        }
    rows: list[dict[str, Any]] = []
    for suture in weight_grid:
        for morphology in weight_grid:
            for surface in weight_grid:
                rows.append(
                    evaluate_prior_weight_combo(
                        tokenizer=tokenizer,
                        calibration_cases=cases,
                        guide=guide,
                        train_text=train_text,
                        visible_limit=visible_limit,
                        morphology_limit=morphology_limit,
                        surface_limit=surface_limit,
                        suture_weight=suture,
                        morphology_weight=morphology,
                        surface_weight=surface,
                    )
                )

    def default_distance(row: dict[str, Any]) -> float:
        return (
            abs(float(row["suture_weight"]) - default_suture_weight)
            + abs(float(row["morphology_weight"]) - default_morphology_weight)
            + abs(float(row["surface_weight"]) - default_surface_weight)
        )

    def row_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        avg_rank = float(row["avg_prior_exact_rank"]) if row["avg_prior_exact_rank"] is not None else 1e9
        return (
            float(row["oracle_exact_rate"]),
            float(row["prior_top4_exact_rate"]),
            float(row["prior_top1_exact_rate"]),
            -avg_rank,
            -default_distance(row),
        )

    selected = max(rows, key=row_key)
    default_row = next(
        (
            row
            for row in rows
            if float(row["suture_weight"]) == float(default_suture_weight)
            and float(row["morphology_weight"]) == float(default_morphology_weight)
            and float(row["surface_weight"]) == float(default_surface_weight)
        ),
        None,
    )
    return {
        "enabled": True,
        "status": "selected_visible_context_weights",
        "calibration_cases": len(cases),
        "selected_weights": {
            "suture_weight": float(selected["suture_weight"]),
            "morphology_weight": float(selected["morphology_weight"]),
            "surface_weight": float(selected["surface_weight"]),
        },
        "selected_summary": selected,
        "default_summary": default_row,
        "top_grid_rows": sorted(rows, key=row_key, reverse=True)[:8],
        "claim_boundary": "visible self-calibration is per-case calibration, not held-out model proof",
    }


def select_lattice_row_with_margin(
    scored_options: list[tuple[float, torch.Tensor, dict[str, Any]]],
    *,
    selector_margin: float,
) -> tuple[float, torch.Tensor, dict[str, Any], dict[str, Any]]:
    if not scored_options:
        raise RuntimeError("retrieval lattice produced no candidates")
    raw_best = max(scored_options, key=lambda item: (item[0], -int(item[2]["prior_rank"]), str(item[2]["predicted_hole"])))
    anchor = scored_options[0]
    anchor_margin_gap = max(0.0, float(raw_best[0] - anchor[0]))
    margin_applied = selector_margin > 0 and raw_best is not anchor and raw_best[0] < anchor[0] + selector_margin
    selected = anchor if margin_applied else raw_best
    raw_best_exact = bool(raw_best[2].get("exact", False))
    anchor_exact = bool(anchor[2].get("exact", False))
    return (
        selected[0],
        selected[1],
        selected[2],
        {
            "selector_margin": float(selector_margin),
            "selector_margin_applied": bool(margin_applied),
            "anchor_margin_gap": json_score(anchor_margin_gap),
            "selector_margin_clears_anchor_gap": bool(raw_best is not anchor and selector_margin > anchor_margin_gap),
            "selector_margin_shortfall": json_score(max(0.0, anchor_margin_gap - float(selector_margin))),
            "selector_effect": classify_selector_effect(
                raw_best_exact=raw_best_exact,
                anchor_exact=anchor_exact,
                margin_applied=bool(margin_applied),
            ),
            "raw_best_hole": raw_best[2]["predicted_hole"],
            "raw_best_score": json_score(raw_best[0]),
            "raw_best_exact": raw_best_exact,
            "raw_best_byte_accuracy": float(raw_best[2].get("byte_accuracy", 0.0)),
            "anchor_hole": anchor[2]["predicted_hole"],
            "anchor_score": json_score(anchor[0]),
            "anchor_exact": anchor_exact,
            "anchor_byte_accuracy": float(anchor[2].get("byte_accuracy", 0.0)),
        },
    )


def classify_retrieval_lattice_outcome(row: dict[str, Any]) -> str:
    if row.get("exact"):
        return "selected_exact"
    if not row.get("oracle_candidate_exact"):
        return "oracle_missing_from_lattice"
    if not row.get("oracle_candidate_exact_in_scored_set"):
        return "oracle_outside_scored_set"
    selector_effect = row.get("selector_effect")
    if selector_effect == "raw_verifier_overrode_exact_anchor":
        return "raw_verifier_overrode_exact_anchor"
    if selector_effect == "margin_blocked_exact_raw":
        return "margin_blocked_exact_raw"
    return "scored_exact_not_selected"


def selector_margin_sweep_report(
    scored_options: list[tuple[float, torch.Tensor, dict[str, Any]]],
    *,
    selector_margins: list[float],
    oracle_candidate_exact: bool,
    oracle_candidate_exact_in_scored_set: bool,
) -> list[dict[str, Any]]:
    report: list[dict[str, Any]] = []
    for margin in selector_margins:
        selected_score, _, selected_row, selector_report = select_lattice_row_with_margin(
            scored_options,
            selector_margin=margin,
        )
        exact = bool(selected_row.get("exact", False))
        outcome_category = classify_retrieval_lattice_outcome(
            {
                "exact": exact,
                "oracle_candidate_exact": oracle_candidate_exact,
                "oracle_candidate_exact_in_scored_set": oracle_candidate_exact_in_scored_set,
                "selector_effect": selector_report["selector_effect"],
            }
        )
        report.append(
            {
                "selector_margin": float(margin),
                "selected_hole": selected_row["predicted_hole"],
                "selected_score": json_score(selected_score),
                "exact": exact,
                "byte_accuracy": float(selected_row.get("byte_accuracy", 0.0)),
                "selector_margin_applied": bool(selector_report["selector_margin_applied"]),
                "anchor_margin_gap": selector_report["anchor_margin_gap"],
                "selector_margin_clears_anchor_gap": bool(selector_report["selector_margin_clears_anchor_gap"]),
                "selector_margin_shortfall": selector_report["selector_margin_shortfall"],
                "selector_effect": selector_report["selector_effect"],
                "outcome_category": outcome_category,
                "raw_best_hole": selector_report["raw_best_hole"],
                "raw_best_exact": bool(selector_report["raw_best_exact"]),
                "anchor_hole": selector_report["anchor_hole"],
                "anchor_exact": bool(selector_report["anchor_exact"]),
            }
        )
    return report


def lattice_oracle_case(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    visible_limit: int = 8,
    morphology_limit: int = 64,
    surface_limit: int = 64,
    suture_weight: float = 2.0,
    morphology_weight: float = 1.0,
    surface_weight: float = 1.0,
    local_prior_calibration: bool = False,
    apply_local_prior_calibration: bool = False,
    prior_weight_grid: list[float] | None = None,
    local_prior_calibration_cases: int = 12,
    local_prior_calibration_context_chars: int = 12,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    target = example.target[example.hole_start : example.hole_end]
    summaries: list[dict[str, Any]] = []
    exact_sources: list[str] = []
    local_calibration_report: dict[str, Any] | None = None
    local_calibration_suggested_rank: int | None = None
    effective_suture_weight = float(suture_weight)
    effective_morphology_weight = float(morphology_weight)
    effective_surface_weight = float(surface_weight)
    if local_prior_calibration:
        local_calibration_report = calibrate_lattice_prior_weights_on_visible_context(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
            default_suture_weight=suture_weight,
            default_morphology_weight=morphology_weight,
            default_surface_weight=surface_weight,
            weight_grid=prior_weight_grid or [suture_weight, morphology_weight, surface_weight],
            calibration_cases=local_prior_calibration_cases,
            calibration_context_chars=local_prior_calibration_context_chars,
        )
        local_calibration_report["applied"] = bool(apply_local_prior_calibration)
        selected_weights = local_calibration_report["selected_weights"]
        local_calibration_suggested_rank, _ = exact_rank_for_prior_weights(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
            suture_weight=float(selected_weights["suture_weight"]),
            morphology_weight=float(selected_weights["morphology_weight"]),
            surface_weight=float(selected_weights["surface_weight"]),
        )
        if apply_local_prior_calibration:
            effective_suture_weight = float(selected_weights["suture_weight"])
            effective_morphology_weight = float(selected_weights["morphology_weight"])
            effective_surface_weight = float(selected_weights["surface_weight"])
    ranked_candidates = rank_lattice_candidates_by_prior(
        build_lattice_candidate_rows(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
        ),
        suture_weight=effective_suture_weight,
        morphology_weight=effective_morphology_weight,
        surface_weight=effective_surface_weight,
    )
    for candidate in ranked_candidates:
        pred = torch.tensor(candidate["ids"], dtype=torch.long)
        exact = bool(torch.equal(pred, target))
        sources = candidate["sources"]
        if exact:
            exact_sources.extend(str(source["source"]) for source in sources)
        summaries.append(
            {
                "predicted_hole": candidate["predicted_hole"],
                "sources": sources,
                "prior_score": candidate["prior_score"],
                "prior_source": candidate["prior_source"],
                "exact": exact,
                "prior_rank": candidate["prior_rank"],
            }
        )
    prior_selected = summaries[0] if summaries else None
    exact_prior_ranks = [int(row["prior_rank"]) for row in summaries if row["exact"]]
    best_exact_prior_rank = min(exact_prior_ranks) if exact_prior_ranks else None

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
        "prior_selected_hole": prior_selected["predicted_hole"] if prior_selected is not None else None,
        "prior_selected_exact": bool(prior_selected["exact"]) if prior_selected is not None else False,
        "prior_selected_score": prior_selected["prior_score"] if prior_selected is not None else None,
        "prior_selected_source": prior_selected["prior_source"] if prior_selected is not None else None,
        "effective_lattice_suture_weight": effective_suture_weight,
        "effective_lattice_morphology_weight": effective_morphology_weight,
        "effective_lattice_surface_weight": effective_surface_weight,
        "local_prior_calibration": local_calibration_report,
        "local_prior_calibration_suggested_prior_exact_rank": local_calibration_suggested_rank,
        "local_prior_calibration_suggested_prior_top4_exact": (
            local_calibration_suggested_rank is not None and local_calibration_suggested_rank < 4
        ),
        "prior_exact_rank": best_exact_prior_rank,
        "prior_exact_in_top4": best_exact_prior_rank is not None and best_exact_prior_rank < 4,
        "prior_exact_in_top8": best_exact_prior_rank is not None and best_exact_prior_rank < 8,
        "prior_selector": "max_structural_source_prior_without_model",
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
    result = {
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
    return result


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
    prior_rerank_top_k: int = 0,
    verifier_mode: str = "suture_loo",
    verifier_top_k: int = 0,
    selector_margin: float = 0.0,
    selector_margin_sweep: list[float] | None = None,
    local_prior_calibration: bool = False,
    apply_local_prior_calibration: bool = False,
    prior_weight_grid: list[float] | None = None,
    local_prior_calibration_cases: int = 12,
    local_prior_calibration_context_chars: int = 12,
) -> dict[str, Any]:
    example = parse_marked_infill(marked_text, tokenizer)
    target = example.target[example.hole_start : example.hole_end]
    local_calibration_report: dict[str, Any] | None = None
    local_calibration_suggested_rank: int | None = None
    effective_suture_weight = float(suture_weight)
    effective_morphology_weight = float(morphology_weight)
    effective_surface_weight = float(surface_weight)
    if local_prior_calibration:
        local_calibration_report = calibrate_lattice_prior_weights_on_visible_context(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
            default_suture_weight=suture_weight,
            default_morphology_weight=morphology_weight,
            default_surface_weight=surface_weight,
            weight_grid=prior_weight_grid or [suture_weight, morphology_weight, surface_weight],
            calibration_cases=local_prior_calibration_cases,
            calibration_context_chars=local_prior_calibration_context_chars,
        )
        local_calibration_report["applied"] = bool(apply_local_prior_calibration)
        selected_weights = local_calibration_report["selected_weights"]
        local_calibration_suggested_rank, _ = exact_rank_for_prior_weights(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
            suture_weight=float(selected_weights["suture_weight"]),
            morphology_weight=float(selected_weights["morphology_weight"]),
            surface_weight=float(selected_weights["surface_weight"]),
        )
        if apply_local_prior_calibration:
            effective_suture_weight = float(selected_weights["suture_weight"])
            effective_morphology_weight = float(selected_weights["morphology_weight"])
            effective_surface_weight = float(selected_weights["surface_weight"])
    ranked_candidates = rank_lattice_candidates_by_prior(
        build_lattice_candidate_rows(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=visible_limit,
            morphology_limit=morphology_limit,
            surface_limit=surface_limit,
        ),
        suture_weight=effective_suture_weight,
        morphology_weight=effective_morphology_weight,
        surface_weight=effective_surface_weight,
    )
    exact_prior_ranks = [
        int(candidate["prior_rank"])
        for candidate in ranked_candidates
        if torch.equal(torch.tensor(candidate["ids"], dtype=torch.long), target)
    ]
    prior_exact_rank = min(exact_prior_ranks) if exact_prior_ranks else None
    if prior_rerank_top_k > 0:
        candidates_to_score = ranked_candidates[:prior_rerank_top_k]
    else:
        candidates_to_score = ranked_candidates
    rerank_limit = int(prior_rerank_top_k) if prior_rerank_top_k > 0 else len(ranked_candidates)
    oracle_exact_in_scored_set = prior_exact_rank is not None and prior_exact_rank < rerank_limit
    scored_candidate_ids = {tuple(int(token_id) for token_id in candidate["ids"]) for candidate in candidates_to_score}

    scored_rows: list[dict[str, Any]] = []
    scored_options: list[tuple[float, torch.Tensor, dict[str, Any]]] = []
    for candidate in candidates_to_score:
        repaired = example.tokens.clone()
        repaired[example.hole_start : example.hole_end] = torch.tensor(candidate["ids"], dtype=torch.long)
        score, verifier_scores = score_lattice_verifier(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired,
            hole_start=example.hole_start,
            hole_end=example.hole_end,
            guide=guide,
            guidance=guidance,
            temperature=temperature,
            top_k=verifier_top_k,
            verifier_mode=verifier_mode,
        )
        pred = repaired[example.hole_start : example.hole_end]
        prior_score = float(candidate["prior_score"]) if candidate["prior_score"] is not None else float("-inf")
        combined_score = score + prior_score
        row = {
            "source": candidate["prior_source"],
            "predicted_hole": tokenizer.decode(pred),
            "sources": candidate["sources"],
            "prior_rank": int(candidate["prior_rank"]),
            "prior_score": candidate["prior_score"],
            "prior_source": candidate["prior_source"],
            "diffusion_score": json_score(score),
            "verifier_mode": verifier_mode,
            "verifier_scores": verifier_scores,
            "diffusion_score_is_finite": math.isfinite(score),
            "combined_score": json_score(combined_score),
            "byte_accuracy": float((pred == target).float().mean().item()) if target.numel() else 0.0,
            "exact": bool(torch.equal(pred, target)),
        }
        scored_rows.append(row)
        scored_options.append((combined_score, repaired, row))
    best_score, best_repaired, best_row, selector_report = select_lattice_row_with_margin(
        scored_options,
        selector_margin=selector_margin,
    )
    sweep_report = selector_margin_sweep_report(
        scored_options,
        selector_margins=selector_margin_sweep or [],
        oracle_candidate_exact=prior_exact_rank is not None,
        oracle_candidate_exact_in_scored_set=oracle_exact_in_scored_set,
    )
    pred = best_repaired[example.hole_start : example.hole_end]
    result = {
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
        "selector": "diffusion_score_plus_structural_prior_topk",
        **selector_report,
        "lattice_verifier_mode": verifier_mode,
        "lattice_verifier_top_k": int(verifier_top_k),
        "lattice_suture_weight": effective_suture_weight,
        "lattice_morphology_weight": effective_morphology_weight,
        "lattice_surface_weight": effective_surface_weight,
        "configured_lattice_suture_weight": float(suture_weight),
        "configured_lattice_morphology_weight": float(morphology_weight),
        "configured_lattice_surface_weight": float(surface_weight),
        "local_prior_calibration": local_calibration_report,
        "local_prior_calibration_suggested_prior_exact_rank": local_calibration_suggested_rank,
        "local_prior_calibration_suggested_prior_top4_exact": (
            local_calibration_suggested_rank is not None and local_calibration_suggested_rank < 4
        ),
        "candidate_count": len(ranked_candidates),
        "scored_candidate_count": len(candidates_to_score),
        "prior_rerank_top_k": int(prior_rerank_top_k),
        "prior_exact_rank": prior_exact_rank,
        "prior_exact_in_rerank_set": oracle_exact_in_scored_set,
        "prior_selected_hole": ranked_candidates[0]["predicted_hole"] if ranked_candidates else None,
        "prior_selected_exact": bool(prior_exact_rank == 0),
        "candidate_summaries": scored_rows,
        "unscored_prior_candidate_summaries": [
            {
                "predicted_hole": candidate["predicted_hole"],
                "sources": candidate["sources"],
                "prior_rank": int(candidate["prior_rank"]),
                "prior_score": candidate["prior_score"],
                "prior_source": candidate["prior_source"],
                "exact": bool(torch.equal(torch.tensor(candidate["ids"], dtype=torch.long), target)),
            }
            for candidate in ranked_candidates
            if tuple(int(token_id) for token_id in candidate["ids"]) not in scored_candidate_ids
        ],
        "oracle_candidate_exact": prior_exact_rank is not None,
        "oracle_candidate_exact_in_scored_set": oracle_exact_in_scored_set,
        "selector_margin_sweep": sweep_report,
    }
    result["outcome_category"] = classify_retrieval_lattice_outcome(result)
    return result


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


def summarize_selector_margin_sweep(rows: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        for sweep_row in row.get("selector_margin_sweep", []):
            key = format_selector_margin(float(sweep_row["selector_margin"]))
            buckets.setdefault(key, []).append(sweep_row)
    summary: dict[str, Any] = {}
    for key, sweep_rows in sorted(buckets.items(), key=lambda item: float(item[0])):
        outcome_categories = Counter(str(row.get("outcome_category", "unknown")) for row in sweep_rows)
        selector_effects = Counter(str(row.get("selector_effect", "unknown")) for row in sweep_rows)
        summary[key] = {
            "cases": len(sweep_rows),
            "exact_match_rate": sum(1.0 for row in sweep_rows if row.get("exact")) / len(sweep_rows),
            "byte_accuracy": sum(float(row.get("byte_accuracy", 0.0)) for row in sweep_rows) / len(sweep_rows),
            "selector_margin_applied_rate": sum(1.0 for row in sweep_rows if row.get("selector_margin_applied"))
            / len(sweep_rows),
            "outcome_categories": dict(sorted(outcome_categories.items())),
            "selector_effects": dict(sorted(selector_effects.items())),
        }
    return summary


def summarize_retrieval_lattice(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = summarize_infill(rows)
    if not rows:
        summary.update(
            {
                "oracle_candidate_exact_rate": 0.0,
                "oracle_candidate_exact_in_scored_set_rate": 0.0,
                "prior_selected_exact_rate": 0.0,
                "raw_best_exact_rate": 0.0,
                "anchor_exact_rate": 0.0,
                "selector_margin_applied_rate": 0.0,
                "avg_scored_candidate_count": 0.0,
                "avg_prior_exact_rank": None,
                "outcome_categories": {},
                "selector_effects": {},
                "margin_rescued_exact_rate": 0.0,
                "raw_verifier_overrode_exact_anchor_rate": 0.0,
                "avg_anchor_margin_gap": 0.0,
                "avg_exact_anchor_margin_gap": None,
                "max_exact_anchor_margin_gap": None,
                "selector_margin_sweep": {},
            }
        )
        return summary

    prior_rank_rows = [row for row in rows if row.get("prior_exact_rank") is not None]
    exact_anchor_rows = [row for row in rows if row.get("anchor_exact")]
    outcome_categories = Counter(str(row.get("outcome_category", "unknown")) for row in rows)
    selector_effects = Counter(str(row.get("selector_effect", "unknown")) for row in rows)
    summary.update(
        {
            "oracle_candidate_exact_rate": sum(1.0 for row in rows if row.get("oracle_candidate_exact")) / len(rows),
            "oracle_candidate_exact_in_scored_set_rate": sum(
                1.0 for row in rows if row.get("oracle_candidate_exact_in_scored_set")
            )
            / len(rows),
            "prior_selected_exact_rate": sum(1.0 for row in rows if row.get("prior_selected_exact")) / len(rows),
            "raw_best_exact_rate": sum(1.0 for row in rows if row.get("raw_best_exact")) / len(rows),
            "anchor_exact_rate": sum(1.0 for row in rows if row.get("anchor_exact")) / len(rows),
            "selector_margin_applied_rate": sum(1.0 for row in rows if row.get("selector_margin_applied"))
            / len(rows),
            "avg_scored_candidate_count": sum(float(row.get("scored_candidate_count", 0)) for row in rows)
            / len(rows),
            "avg_prior_exact_rank": (
                sum(float(row["prior_exact_rank"]) for row in prior_rank_rows) / len(prior_rank_rows)
                if prior_rank_rows
                else None
            ),
            "outcome_categories": dict(sorted(outcome_categories.items())),
            "selector_effects": dict(sorted(selector_effects.items())),
            "margin_rescued_exact_rate": sum(
                1.0 for row in rows if row.get("selector_effect") == "margin_rescued_exact_anchor"
            )
            / len(rows),
            "raw_verifier_overrode_exact_anchor_rate": sum(
                1.0 for row in rows if row.get("selector_effect") == "raw_verifier_overrode_exact_anchor"
            )
            / len(rows),
            "avg_anchor_margin_gap": sum(float(row.get("anchor_margin_gap", 0.0)) for row in rows) / len(rows),
            "avg_exact_anchor_margin_gap": (
                sum(float(row.get("anchor_margin_gap", 0.0)) for row in exact_anchor_rows) / len(exact_anchor_rows)
                if exact_anchor_rows
                else None
            ),
            "max_exact_anchor_margin_gap": (
                max(float(row.get("anchor_margin_gap", 0.0)) for row in exact_anchor_rows)
                if exact_anchor_rows
                else None
            ),
            "selector_margin_sweep": summarize_selector_margin_sweep(rows),
        }
    )
    return summary


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
            "prior_selected_exact_rate": 0.0,
            "prior_top4_exact_rate": 0.0,
            "prior_top8_exact_rate": 0.0,
            "avg_prior_exact_rank": None,
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
        "prior_selected_exact_rate": sum(1.0 for row in rows if row["prior_selected_exact"]) / len(rows),
        "prior_top4_exact_rate": sum(1.0 for row in rows if row["prior_exact_in_top4"]) / len(rows),
        "prior_top8_exact_rate": sum(1.0 for row in rows if row["prior_exact_in_top8"]) / len(rows),
        "avg_prior_exact_rank": (
            sum(float(row["prior_exact_rank"]) for row in rows if row["prior_exact_rank"] is not None)
            / sum(1 for row in rows if row["prior_exact_rank"] is not None)
        )
        if any(row["prior_exact_rank"] is not None for row in rows)
        else None,
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
    prior_weight_grid = parse_prior_weight_grid(args.lattice_prior_weight_grid)
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
            suture_weight=args.lattice_suture_weight,
            morphology_weight=args.lattice_morphology_weight,
            surface_weight=args.lattice_surface_weight,
            local_prior_calibration=args.lattice_local_prior_calibration,
            apply_local_prior_calibration=args.lattice_apply_local_prior_calibration,
            prior_weight_grid=prior_weight_grid,
            local_prior_calibration_cases=args.lattice_local_prior_calibration_cases,
            local_prior_calibration_context_chars=args.lattice_local_prior_calibration_context_chars,
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
            "lattice_suture_weight": float(args.lattice_suture_weight),
            "lattice_morphology_weight": float(args.lattice_morphology_weight),
            "lattice_surface_weight": float(args.lattice_surface_weight),
            "lattice_local_prior_calibration": bool(args.lattice_local_prior_calibration),
            "lattice_apply_local_prior_calibration": bool(args.lattice_apply_local_prior_calibration),
            "lattice_prior_weight_grid": prior_weight_grid,
            "lattice_local_prior_calibration_cases": int(args.lattice_local_prior_calibration_cases),
            "lattice_local_prior_calibration_context_chars": int(args.lattice_local_prior_calibration_context_chars),
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
    selector_margin_sweep = parse_selector_margin_sweep(args.lattice_selector_margin_sweep)
    prior_weight_grid = parse_prior_weight_grid(args.lattice_prior_weight_grid)
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
            prior_rerank_top_k=args.lattice_prior_rerank_top_k,
            verifier_mode=args.lattice_verifier_mode,
            verifier_top_k=args.lattice_verifier_top_k,
            selector_margin=args.lattice_selector_margin,
            selector_margin_sweep=selector_margin_sweep,
            local_prior_calibration=args.lattice_local_prior_calibration,
            apply_local_prior_calibration=args.lattice_apply_local_prior_calibration,
            prior_weight_grid=prior_weight_grid,
            local_prior_calibration_cases=args.lattice_local_prior_calibration_cases,
            local_prior_calibration_context_chars=args.lattice_local_prior_calibration_context_chars,
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
    retrieval_lattice_summary = summarize_retrieval_lattice(retrieval_lattice_rows)
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
            "lattice_prior_rerank_top_k": int(args.lattice_prior_rerank_top_k),
            "lattice_verifier_mode": args.lattice_verifier_mode,
            "lattice_verifier_top_k": int(args.lattice_verifier_top_k),
            "lattice_selector_margin": float(args.lattice_selector_margin),
            "lattice_selector_margin_sweep": parse_selector_margin_sweep(args.lattice_selector_margin_sweep),
            "lattice_local_prior_calibration": bool(args.lattice_local_prior_calibration),
            "lattice_apply_local_prior_calibration": bool(args.lattice_apply_local_prior_calibration),
            "lattice_prior_weight_grid": prior_weight_grid,
            "lattice_local_prior_calibration_cases": int(args.lattice_local_prior_calibration_cases),
            "lattice_local_prior_calibration_context_chars": int(args.lattice_local_prior_calibration_context_chars),
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
    parser.add_argument("--lattice-prior-rerank-top-k", type=int, default=0)
    parser.add_argument("--lattice-verifier-mode", choices=["suture_loo", "full_hole", "dual"], default="suture_loo")
    parser.add_argument("--lattice-verifier-top-k", type=int, default=0)
    parser.add_argument("--lattice-selector-margin", type=float, default=0.0)
    parser.add_argument("--lattice-selector-margin-sweep", default="0,1,2,3,5")
    parser.add_argument("--lattice-local-prior-calibration", action="store_true")
    parser.add_argument("--lattice-apply-local-prior-calibration", action="store_true")
    parser.add_argument("--lattice-prior-weight-grid", default="0.5,1,2,4")
    parser.add_argument("--lattice-local-prior-calibration-cases", type=int, default=12)
    parser.add_argument("--lattice-local-prior-calibration-context-chars", type=int, default=12)
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
