from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

from ..bench import (
    build_lattice_candidate_rows,
    make_marked_cases,
    parse_positive_int_grid,
    rank_lattice_candidates_by_prior,
    sha256_text,
    split_text,
)
from ..data import load_text
from ..infill import parse_marked_infill
from ..lattice.in_document_echo import echo_overlap_audit, in_document_echo_candidates
from ..ngram import BigramGuide
from ..tokenizer import ByteTokenizer


DEFAULT_CONFIG: dict[str, Any] = {
    "data": "data/tinyshakespeare.txt",
    "cases": 8,
    "span_chars": 4,
    "context_chars": 36,
    "val_fraction": 0.08,
    "seed": 42,
    "max_candidates_per_example": 128,
    "anchor_window_bytes": 64,
    "anchor_sizes": [32, 24, 16, 12, 8, 6, 4],
    "lattice_visible_candidates": 8,
    "lattice_morphology_candidates": 64,
    "lattice_surface_candidates": 64,
    "lattice_bi_anchor_candidates": 8,
    "shuffle_trials": 32,
    "require_unseen_hole": False,
}


def _parse_scalar(raw: str) -> Any:
    value = raw.strip()
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if "," in value:
        return [int(chunk.strip()) for chunk in value.split(",") if chunk.strip()]
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def load_config(path: str | Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".json":
        config.update(json.loads(text))
        return config
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, raw = stripped.split(":", 1)
        config[key.strip()] = _parse_scalar(raw)
    return config


def current_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return None


def current_git_dirty() -> bool | None:
    try:
        return bool(subprocess.check_output(["git", "status", "--short"], text=True).strip())
    except Exception:
        return None


def _byte_offset(text: str, char_offset: int) -> int:
    return len(text[:char_offset].encode("utf-8", errors="replace"))


def _make_document_cases(
    text: str,
    *,
    train_text: str,
    cases: int,
    span_chars: int,
    context_chars: int,
    seed: int,
    require_unseen_hole: bool,
) -> list[dict[str, Any]]:
    clean = text.replace("[[", "").replace("]]", "")
    if len(clean) < (context_chars * 2) + span_chars + 8:
        raise ValueError("text is too small for in-document echo case construction")
    rng = random.Random(seed)
    out: list[dict[str, Any]] = []
    attempts = 0
    while len(out) < cases and attempts < cases * 400:
        attempts += 1
        start = rng.randint(context_chars, len(clean) - context_chars - span_chars - 1)
        hole = clean[start : start + span_chars]
        if "\n" in hole or not hole.strip():
            continue
        if require_unseen_hole and hole in train_text:
            continue
        before = clean[start - context_chars : start]
        after = clean[start + span_chars : start + span_chars + context_chars]
        byte_start = _byte_offset(clean, start)
        byte_end = _byte_offset(clean, start + span_chars)
        out.append(
            {
                "marked_text": f"{before}[[{hole}]]{after}",
                "target_char_start": int(start),
                "target_byte_start": int(byte_start),
                "target_byte_end": int(byte_end),
                "target_byte_len": int(byte_end - byte_start),
            }
        )
    if len(out) < cases:
        raise ValueError(f"only built {len(out)} in-document echo cases from requested {cases}")
    return out


def _target_key(marked_text: str, tokenizer: ByteTokenizer) -> tuple[int, ...]:
    example = parse_marked_infill(marked_text, tokenizer)
    return tuple(int(token_id) for token_id in example.target[example.hole_start : example.hole_end].tolist())


def _rank_for_key(rows: list[dict[str, Any]], key: tuple[int, ...]) -> int | None:
    for rank, row in enumerate(rows):
        if tuple(int(token_id) for token_id in row["ids"]) == key:
            return rank
    return None


def _p95(values: list[int]) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))
    return int(ordered[index])


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row.get(key)) / len(rows)


def _shuffle_falsification(
    *,
    case_rows: list[dict[str, Any]],
    target_keys: list[tuple[int, ...]],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if not case_rows or not target_keys or trials <= 0:
        return {
            "trials": int(trials),
            "gold_in_echo_lattice_at_128": 0.0,
            "top4_exact_combined": 0.0,
        }
    rng = random.Random(seed)
    echo_rates: list[float] = []
    top4_rates: list[float] = []
    for _ in range(trials):
        shuffled = list(target_keys)
        rng.shuffle(shuffled)
        echo_hits = 0
        top4_hits = 0
        for row, target in zip(case_rows, shuffled, strict=False):
            echo_hits += int(target in {tuple(item) for item in row["_echo_keys_at_128"]})
            top4_hits += int(target in {tuple(item) for item in row["_combined_top4_keys"]})
        echo_rates.append(echo_hits / len(case_rows))
        top4_rates.append(top4_hits / len(case_rows))
    return {
        "trials": int(trials),
        "gold_in_echo_lattice_at_128": sum(echo_rates) / len(echo_rates),
        "top4_exact_combined": sum(top4_rates) / len(top4_rates),
    }


def build_receipt(config: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    tokenizer = ByteTokenizer()
    text = load_text(str(cfg["data"]))
    train_text, val_text = split_text(text, val_fraction=float(cfg["val_fraction"]))
    guide = BigramGuide.from_text(train_text, tokenizer)
    anchor_sizes = [int(size) for size in cfg["anchor_sizes"]]
    cases = _make_document_cases(
        val_text,
        train_text=train_text,
        cases=int(cfg["cases"]),
        span_chars=int(cfg["span_chars"]),
        context_chars=int(cfg["context_chars"]),
        seed=int(cfg["seed"]),
        require_unseen_hole=bool(cfg["require_unseen_hole"]),
    )
    max_candidates = int(cfg["max_candidates_per_example"])
    case_rows: list[dict[str, Any]] = []
    target_keys: list[tuple[int, ...]] = []

    for case_id, case in enumerate(cases):
        marked_text = str(case["marked_text"])
        target_key = _target_key(marked_text, tokenizer)
        target_keys.append(target_key)
        prior = rank_lattice_candidates_by_prior(
            build_lattice_candidate_rows(
                tokenizer=tokenizer,
                marked_text=marked_text,
                guide=guide,
                train_text=train_text,
                visible_limit=int(cfg["lattice_visible_candidates"]),
                morphology_limit=int(cfg["lattice_morphology_candidates"]),
                surface_limit=int(cfg["lattice_surface_candidates"]),
                bi_anchor_limit=int(cfg["lattice_bi_anchor_candidates"]),
                bi_anchor_sizes=anchor_sizes,
            )
        )[:max_candidates]
        echo = in_document_echo_candidates(
            tokenizer=tokenizer,
            document_text=val_text,
            span_start=int(case["target_byte_start"]),
            span_end=int(case["target_byte_end"]),
            context_bytes=int(cfg["context_chars"]),
            anchor_sizes=anchor_sizes,
            anchor_window_bytes=int(cfg["anchor_window_bytes"]),
            limit=max_candidates,
        )
        seen: set[tuple[int, ...]] = set()
        combined: list[dict[str, Any]] = []
        for row in prior:
            key = tuple(int(token_id) for token_id in row["ids"])
            if key not in seen:
                combined.append(row)
                seen.add(key)
        for row in echo:
            key = tuple(int(token_id) for token_id in row["ids"])
            if key not in seen:
                combined.append(row)
                seen.add(key)
        combined = combined[:max_candidates]
        echo_rank = _rank_for_key(echo, target_key)
        prior_rank = _rank_for_key(prior, target_key)
        combined_rank = _rank_for_key(combined, target_key)
        hit_modes: list[str] = []
        if echo_rank is not None:
            for source in echo[echo_rank].get("sources", []):
                mode = str(source.get("anchor_mode", ""))
                if mode and mode not in hit_modes:
                    hit_modes.append(mode)
        audit = echo_overlap_audit(echo)
        case_rows.append(
            {
                "case_id": int(case_id),
                "target_byte_start": int(case["target_byte_start"]),
                "target_byte_end": int(case["target_byte_end"]),
                "target_byte_len": int(case["target_byte_len"]),
                "target_sha256": sha256_text(parse_marked_infill(marked_text, tokenizer).hole),
                "prior_candidate_count": len(prior),
                "echo_candidate_count": len(echo),
                "combined_candidate_count": len(combined),
                "gold_in_prior_lattice_at_128": prior_rank is not None,
                "gold_in_echo_lattice_at_128": echo_rank is not None,
                "gold_in_combined_lattice_at_128": combined_rank is not None,
                "top4_exact_prior": prior_rank is not None and prior_rank < 4,
                "top4_exact_combined_label_free_order": combined_rank is not None and combined_rank < 4,
                "gold_rank_echo": echo_rank,
                "gold_rank_prior": prior_rank,
                "gold_rank_combined": combined_rank,
                "echo_hit_modes": hit_modes,
                "echo_overlap_audit": audit,
                "_echo_keys_at_128": [tuple(int(token_id) for token_id in row["ids"]) for row in echo[:max_candidates]],
                "_combined_top4_keys": [
                    tuple(int(token_id) for token_id in row["ids"]) for row in combined[:4]
                ],
            }
        )

    candidate_counts = [int(row["echo_candidate_count"]) for row in case_rows]
    echo_ranks = [int(row["gold_rank_echo"]) for row in case_rows if row["gold_rank_echo"] is not None]
    prior_rate = _rate(case_rows, "gold_in_prior_lattice_at_128")
    combined_rate = _rate(case_rows, "gold_in_combined_lattice_at_128")
    echo_rate = _rate(case_rows, "gold_in_echo_lattice_at_128")
    shuffle = _shuffle_falsification(
        case_rows=case_rows,
        target_keys=target_keys,
        trials=int(cfg["shuffle_trials"]),
        seed=int(cfg["seed"]) + 991,
    )
    total_audit = {
        "target_window_overlap_hits": sum(
            int(row["echo_overlap_audit"]["target_window_overlap_hits"]) for row in case_rows
        ),
        "target_anchor_window_overlap_hits": sum(
            int(row["echo_overlap_audit"]["target_anchor_window_overlap_hits"]) for row in case_rows
        ),
        "same_offset_hits": sum(int(row["echo_overlap_audit"]["same_offset_hits"]) for row in case_rows),
        "sentinel_source_window_hits": sum(
            int(row["echo_overlap_audit"]["sentinel_source_window_hits"]) for row in case_rows
        ),
    }
    hit_modes = [mode for row in case_rows for mode in row["echo_hit_modes"]]
    receipt = {
        "proof_name": "in_document_echo_oracle_smoke",
        "kind": "helixdiff_in_document_echo_oracle_receipt",
        "model_load": False,
        "git": {"commit": current_commit(), "dirty": current_git_dirty()},
        "data": {
            "path": str(cfg["data"]),
            "sha256": sha256_text(text),
            "train_sha256": sha256_text(train_text),
            "val_sha256": sha256_text(val_text),
            "val_fraction": float(cfg["val_fraction"]),
        },
        "case_selection": {
            "cases": int(cfg["cases"]),
            "seed": int(cfg["seed"]),
            "span_chars": int(cfg["span_chars"]),
            "context_chars": int(cfg["context_chars"]),
            "require_unseen_hole": bool(cfg["require_unseen_hole"]),
            "selection_uses_echo_hit": False,
        },
        "max_candidates_per_example": max_candidates,
        "redaction": {
            "target_span_redacted_before_index": True,
            "sentinel_present_in_candidate_windows": False,
            "target_bytes_available_to_echo_index": False,
            "masked_bytes_seen_by_features": False,
        },
        "source_policy": {
            "uses_eval_visible_document": True,
            "claim_mode": "transductive_visible_context_repair",
            **total_audit,
        },
        "echo_lattice": {
            "candidate_count_median": median(candidate_counts) if candidate_counts else 0,
            "candidate_count_p95": _p95(candidate_counts),
            "exact_bi_anchor_hits": hit_modes.count("exact_bi_anchor_echo"),
            "left_echo_hits": hit_modes.count("left_echo"),
            "right_echo_hits": hit_modes.count("right_echo"),
            "morphology_echo_hits": hit_modes.count("morphology_echo"),
            "gold_in_echo_lattice_at_128": echo_rate,
            "gold_rank_median": median(echo_ranks) if echo_ranks else None,
        },
        "combined_lattice": {
            "union_order": "baseline_prior_first_then_redacted_echo_fill",
            "gold_in_prior_lattice_at_128": prior_rate,
            "gold_in_combined_lattice_at_128": combined_rate,
            "containment_lift": combined_rate - prior_rate,
            "top4_exact_prior": _rate(case_rows, "top4_exact_prior"),
            "top4_exact_combined_label_free_order": _rate(case_rows, "top4_exact_combined_label_free_order"),
        },
        "shuffle_falsification": shuffle,
        "cases": [
            {key: value for key, value in row.items() if not key.startswith("_")} for row in case_rows
        ],
        "claim_boundary": (
            "model-free redacted in-document echo lattice for fixed-span visible-context document repair; "
            "not global LM SOTA, not variable-length infilling, and not selected model accuracy"
        ),
    }
    checks = {
        "model_not_loaded": receipt["model_load"] is False,
        "target_span_redacted_before_index": receipt["redaction"]["target_span_redacted_before_index"] is True,
        "target_bytes_not_available_to_echo_index": receipt["redaction"]["target_bytes_available_to_echo_index"] is False,
        "masked_bytes_not_seen_by_features": receipt["redaction"]["masked_bytes_seen_by_features"] is False,
        "no_target_window_overlap": receipt["source_policy"]["target_window_overlap_hits"] == 0,
        "no_target_anchor_window_overlap": receipt["source_policy"]["target_anchor_window_overlap_hits"] == 0,
        "no_same_offset_hits": receipt["source_policy"]["same_offset_hits"] == 0,
        "no_sentinel_source_window_hits": receipt["source_policy"]["sentinel_source_window_hits"] == 0,
        "candidate_count_p95_within_budget": receipt["echo_lattice"]["candidate_count_p95"] <= max_candidates,
        "containment_lift_positive": receipt["combined_lattice"]["gold_in_combined_lattice_at_128"]
        > receipt["combined_lattice"]["gold_in_prior_lattice_at_128"],
        "shuffle_echo_below_real": float(shuffle["gold_in_echo_lattice_at_128"]) < echo_rate,
    }
    receipt["checks"] = checks
    receipt["verdict"] = "pass" if all(checks.values()) else "fail"
    receipt["kill_criterion"] = (
        "kill if 8 cheap cases add echo candidates but produce zero containment lift at K=128 "
        "while candidate_count_p95 increases"
    )
    return receipt


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit the HelixDiff redacted in-document echo lattice smoke receipt.")
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--cases", type=int)
    parser.add_argument("--span-chars", type=int)
    parser.add_argument("--context-chars", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max-candidates-per-example", type=int)
    parser.add_argument("--anchor-sizes")
    parser.add_argument("--require-unseen-hole", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.cases is not None:
        config["cases"] = args.cases
    if args.span_chars is not None:
        config["span_chars"] = args.span_chars
    if args.context_chars is not None:
        config["context_chars"] = args.context_chars
    if args.seed is not None:
        config["seed"] = args.seed
    if args.max_candidates_per_example is not None:
        config["max_candidates_per_example"] = args.max_candidates_per_example
    if args.anchor_sizes:
        config["anchor_sizes"] = parse_positive_int_grid(args.anchor_sizes)
    if args.require_unseen_hole:
        config["require_unseen_hole"] = True
    receipt = build_receipt(config)
    text = json.dumps(receipt, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text if args.json or not args.out else f"wrote {args.out}")
    if receipt["verdict"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
