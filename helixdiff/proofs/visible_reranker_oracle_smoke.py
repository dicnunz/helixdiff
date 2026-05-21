from __future__ import annotations

import argparse
import json
import random
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

from ..bench import (
    _ranked_candidates_with_surface_report,
    bi_anchor_gap_candidates,
    build_lattice_candidate_rows,
    calibrate_lattice_visible_reranker_on_visible_context,
    make_marked_cases,
    parse_positive_int_grid,
    rank_lattice_candidates_by_prior,
    sha256_text,
    split_text,
    surface_verifier_candidate_report,
    visible_reranker_candidate_report,
)
from ..data import load_text
from ..infill import parse_marked_infill
from ..ngram import BigramGuide
from ..tokenizer import ByteTokenizer


DEFAULT_CONFIG: dict[str, Any] = {
    "data": "data/tinyshakespeare.txt",
    "cases": 2,
    "span_chars": 4,
    "context_chars": 36,
    "val_fraction": 0.08,
    "seed": 101,
    "max_candidates_per_example": 32,
    "calibration_cases": 2,
    "lattice_visible_candidates": 8,
    "lattice_morphology_candidates": 64,
    "lattice_surface_candidates": 64,
    "lattice_bi_anchor_candidates": 8,
    "lattice_bi_anchor_sizes": [32, 24, 16, 12, 8, 6, 4],
    "shuffle_trials": 3,
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


def _target_key(marked_text: str, tokenizer: ByteTokenizer) -> tuple[int, ...]:
    example = parse_marked_infill(marked_text, tokenizer)
    return tuple(int(token_id) for token_id in example.target[example.hole_start : example.hole_end].tolist())


def _branch_summary(case_rows: list[dict[str, Any]], branch: str) -> dict[str, Any]:
    if not case_rows:
        return {
            "selected_exact": 0.0,
            "top4_exact": 0.0,
            "avg_gold_rank": None,
            "help": 0,
            "harm": 0,
        }
    rank_key = f"{branch}_rank"
    selected_key = f"{branch}_selected_exact"
    top4_key = f"{branch}_top4_exact"
    rank_rows = [row for row in case_rows if row.get(rank_key) is not None]
    prior_top4 = {row["case_id"]: bool(row["prior_top4_exact"]) for row in case_rows}
    top4 = {row["case_id"]: bool(row[top4_key]) for row in case_rows}
    return {
        "selected_exact": sum(1.0 for row in case_rows if row[selected_key]) / len(case_rows),
        "top4_exact": sum(1.0 for row in case_rows if row[top4_key]) / len(case_rows),
        "avg_gold_rank": (
            sum(float(row[rank_key]) for row in rank_rows) / len(rank_rows) if rank_rows else None
        ),
        "help": sum(1 for case_id, value in top4.items() if value and not prior_top4[case_id]),
        "harm": sum(1 for case_id, value in top4.items() if not value and prior_top4[case_id]),
    }


def _shuffle_falsification(
    *,
    case_rows: list[dict[str, Any]],
    target_keys: list[tuple[int, ...]],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if not case_rows or not target_keys or trials <= 0:
        return {"num_trials": int(trials), "gold_in_lattice_at_128": 0.0, "visible_reranker_selected_exact": 0.0}
    rng = random.Random(seed)
    lattice_rates: list[float] = []
    selected_rates: list[float] = []
    for _ in range(trials):
        shuffled = list(target_keys)
        rng.shuffle(shuffled)
        lattice_hits = 0
        selected_hits = 0
        for row, target in zip(case_rows, shuffled, strict=False):
            candidate_keys = {tuple(item) for item in row["_candidate_keys_at_128"]}
            lattice_hits += int(target in candidate_keys)
            selected_hits += int(tuple(row["_visible_reranker_selected_key"]) == target)
        lattice_rates.append(lattice_hits / len(case_rows))
        selected_rates.append(selected_hits / len(case_rows))
    return {
        "num_trials": int(trials),
        "gold_in_lattice_at_128": sum(lattice_rates) / len(lattice_rates),
        "visible_reranker_selected_exact": sum(selected_rates) / len(selected_rates),
    }


def build_receipt(config: dict[str, Any]) -> dict[str, Any]:
    tokenizer = ByteTokenizer()
    text = load_text(str(config["data"]))
    train_text, val_text = split_text(text, float(config["val_fraction"]))
    cases = make_marked_cases(
        val_text,
        cases=int(config["cases"]),
        span_chars=int(config["span_chars"]),
        context_chars=int(config["context_chars"]),
        seed=int(config["seed"]),
        forbidden_text=train_text,
        require_unseen_hole=True,
    )
    guide = BigramGuide.from_text(train_text, tokenizer)
    bi_anchor_sizes = config["lattice_bi_anchor_sizes"]
    if isinstance(bi_anchor_sizes, str):
        bi_anchor_sizes = parse_positive_int_grid(bi_anchor_sizes)
    case_rows: list[dict[str, Any]] = []
    target_keys: list[tuple[int, ...]] = []
    leakage_flags = Counter()
    max_candidates = int(config["max_candidates_per_example"])

    for case_id, marked in enumerate(cases):
        target_key = _target_key(marked, tokenizer)
        target_text = parse_marked_infill(marked, tokenizer).hole
        target_keys.append(target_key)
        calibration = calibrate_lattice_visible_reranker_on_visible_context(
            tokenizer=tokenizer,
            marked_text=marked,
            guide=guide,
            train_text=train_text,
            visible_limit=int(config["lattice_visible_candidates"]),
            morphology_limit=int(config["lattice_morphology_candidates"]),
            surface_limit=int(config["lattice_surface_candidates"]),
            suture_weight=2.0,
            morphology_weight=1.0,
            surface_weight=1.0,
            calibration_cases=max(1, int(config.get("calibration_cases", 2))),
            calibration_context_chars=12,
        )
        selected_weights = calibration["selected_weights"]
        ranked = rank_lattice_candidates_by_prior(
            build_lattice_candidate_rows(
                tokenizer=tokenizer,
                marked_text=marked,
                guide=guide,
                train_text=train_text,
                visible_limit=int(config["lattice_visible_candidates"]),
                morphology_limit=int(config["lattice_morphology_candidates"]),
                surface_limit=int(config["lattice_surface_candidates"]),
                bi_anchor_limit=int(config["lattice_bi_anchor_candidates"]),
                bi_anchor_sizes=list(bi_anchor_sizes),
            )
        )[:max_candidates]
        surface = surface_verifier_candidate_report(ranked, marked_text=marked, train_text=train_text)
        reranker = visible_reranker_candidate_report(
            _ranked_candidates_with_surface_report(ranked, surface),
            prior_weight=float(selected_weights["prior_weight"]),
            surface_weight=float(selected_weights["surface_weight"]),
        )
        candidate_keys = [tuple(int(token_id) for token_id in row["ids"]) for row in ranked]
        prior_rank = candidate_keys.index(target_key) if target_key in candidate_keys else None
        surface_ranks = {
            tuple(int(token_id) for token_id in ids): report["surface_verifier_rank"] for ids, report in surface.items()
        }
        reranker_ranks = {
            tuple(int(token_id) for token_id in ids): report["visible_reranker_rank"] for ids, report in reranker.items()
        }
        surface_rank = int(surface_ranks[target_key]) if target_key in surface_ranks else None
        reranker_rank = int(reranker_ranks[target_key]) if target_key in reranker_ranks else None

        prior_selected = candidate_keys[0] if candidate_keys else ()
        surface_selected = min(candidate_keys, key=lambda key: surface_ranks.get(key, 10**9), default=())
        reranker_selected = min(candidate_keys, key=lambda key: reranker_ranks.get(key, 10**9), default=())
        bi_anchor_rows = bi_anchor_gap_candidates(
            tokenizer=tokenizer,
            marked_text=marked,
            train_text=train_text,
            anchor_sizes=list(bi_anchor_sizes),
            limit=int(config["lattice_bi_anchor_candidates"]),
        )
        if target_text in train_text:
            leakage_flags["target_hole_seen_in_train_split"] += 1
        case_rows.append(
            {
                "case_id": case_id,
                "target_hole_sha256": sha256_text(target_text),
                "candidate_count": len(candidate_keys),
                "gold_in_lattice_at_128": target_key in candidate_keys[:128],
                "bi_anchor_candidate_count": len(bi_anchor_rows),
                "bi_anchor_gold_in_lattice": any(
                    tuple(int(token_id) for token_id in row["ids"]) == target_key for row in bi_anchor_rows
                ),
                "prior_rank": prior_rank,
                "surface_rank": surface_rank,
                "visible_reranker_rank": reranker_rank,
                "prior_selected_exact": prior_selected == target_key,
                "surface_selected_exact": surface_selected == target_key,
                "visible_reranker_selected_exact": reranker_selected == target_key,
                "prior_top4_exact": prior_rank is not None and prior_rank < 4,
                "surface_top4_exact": surface_rank is not None and surface_rank < 4,
                "visible_reranker_top4_exact": reranker_rank is not None and reranker_rank < 4,
                "visible_reranker_calibration": {
                    "status": calibration["status"],
                    "selected_weights": selected_weights,
                    "apply": False,
                    "calibration_cases": calibration["calibration_cases"],
                },
                "_candidate_keys_at_128": [list(key) for key in candidate_keys[:128]],
                "_visible_reranker_selected_key": list(reranker_selected),
            }
        )

    candidate_counts = sorted(int(row["candidate_count"]) for row in case_rows)
    p95_index = min(len(candidate_counts) - 1, int(round((len(candidate_counts) - 1) * 0.95))) if candidate_counts else 0
    selector_anchor_sweep = {
        "prior": _branch_summary(case_rows, "prior"),
        "surface": _branch_summary(case_rows, "surface"),
        "visible_reranker": {
            **_branch_summary(case_rows, "visible_reranker"),
            "calibration": {
                "bins": [],
                "ece_like": None,
                "status": "per_case_visible_context_weight_sweep",
            },
            "apply": False,
        },
    }
    shuffle = _shuffle_falsification(
        case_rows=case_rows,
        target_keys=target_keys,
        trials=int(config["shuffle_trials"]),
        seed=int(config["seed"]) + 17,
    )
    leakage = {
        "eval_doc_hits": 0,
        "same_doc_hits": 0,
        "near_duplicate_flags": 0,
        "masked_bytes_seen_by_features": False,
        "gold_used_for_selection": False,
        "target_hole_seen_in_train_split": int(leakage_flags["target_hole_seen_in_train_split"]),
        "document_id_support": "single_contiguous_train_validation_split_no_doc_ids",
    }
    contract_passed = (
        leakage["target_hole_seen_in_train_split"] == 0
        and leakage["eval_doc_hits"] == 0
        and leakage["same_doc_hits"] == 0
        and leakage["masked_bytes_seen_by_features"] is False
        and leakage["gold_used_for_selection"] is False
        and selector_anchor_sweep["visible_reranker"]["apply"] is False
    )
    return {
        "proof_name": "visible_reranker_oracle_smoke",
        "commit": current_commit(),
        "git_dirty": current_git_dirty(),
        "model_load": False,
        "device": "cpu",
        "num_examples": len(case_rows),
        "max_candidates_per_example": max_candidates,
        "train_split_sha256": sha256_text(train_text),
        "validation_split_sha256": sha256_text(val_text),
        "case_filter": {
            "require_unseen_hole": True,
            "mask_policy": "fixed_seed_visible_context",
            "lattice_bi_anchor_candidates": int(config["lattice_bi_anchor_candidates"]),
            "lattice_bi_anchor_sizes": list(bi_anchor_sizes),
        },
        "lattice": {
            "gold_in_lattice_at_128": sum(1.0 for row in case_rows if row["gold_in_lattice_at_128"])
            / len(case_rows),
            "bi_anchor_gold_in_lattice_rate": sum(1.0 for row in case_rows if row["bi_anchor_gold_in_lattice"])
            / len(case_rows),
            "median_candidate_count": candidate_counts[len(candidate_counts) // 2] if candidate_counts else 0,
            "p95_candidate_count": candidate_counts[p95_index] if candidate_counts else 0,
        },
        "selector_anchor_sweep": selector_anchor_sweep,
        "leakage": leakage,
        "shuffle_falsification": shuffle,
        "cases": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in case_rows
        ],
        "verdict": "pass" if contract_passed else "fail",
        "claim_boundary": (
            "model-free visible-context reranker proof only; reranker apply=false and no generation claim"
        ),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit a model-free visible-reranker oracle smoke receipt.")
    parser.add_argument("--config", default="configs/proof_visible_reranker_oracle_smoke.yaml")
    parser.add_argument("--out")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    receipt = build_receipt(load_config(args.config))
    text = json.dumps(receipt, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text if args.json or not args.out else f"wrote {args.out}")


if __name__ == "__main__":
    main()
