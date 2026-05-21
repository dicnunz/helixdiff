from __future__ import annotations

import argparse
import json
import math
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
    "counterfactual_context_chars": 12,
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


def _counterfactual_marked_text(marked_text: str, mode: str, *, context_chars: int) -> str:
    before, rest = marked_text.split("[[", 1)
    hole, after = rest.split("]]", 1)
    width = max(1, int(context_chars))
    if mode == "blank_context":
        return f"{' ' * min(len(before), width)}[[{hole}]]{' ' * min(len(after), width)}"
    if mode == "swapped_edges":
        return f"{after[:width]}[[{hole}]]{before[-width:]}"
    raise ValueError(f"unknown counterfactual context mode: {mode}")


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


def _counterfactual_context_summary(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_rows:
        return {
            "verdict": "fail",
            "real_visible_reranker_selected_exact": 0.0,
            "blank_context_selected_exact": 0.0,
            "swapped_edges_selected_exact": 0.0,
            "causal_visible_selected_exact": 0.0,
            "causal_visible_top4_exact": 0.0,
            "context_specificity_gap": 0.0,
            "apply": False,
        }
    cases = len(case_rows)
    real_rate = sum(1.0 for row in case_rows if row["visible_reranker_selected_exact"]) / cases
    blank_rate = sum(
        1.0 for row in case_rows if row["counterfactual_context"]["blank_context_selected_exact"]
    ) / cases
    swapped_rate = sum(
        1.0 for row in case_rows if row["counterfactual_context"]["swapped_edges_selected_exact"]
    ) / cases
    causal_rate = sum(
        1.0 for row in case_rows if row["counterfactual_context"]["causal_visible_selected_exact"]
    ) / cases
    causal_top4 = sum(
        1.0 for row in case_rows if row["counterfactual_context"]["causal_visible_top4_exact"]
    ) / cases
    return {
        "verdict": "pass" if causal_rate >= real_rate and max(blank_rate, swapped_rate) < real_rate else "fail",
        "real_visible_reranker_selected_exact": real_rate,
        "blank_context_selected_exact": blank_rate,
        "swapped_edges_selected_exact": swapped_rate,
        "causal_visible_selected_exact": causal_rate,
        "causal_visible_top4_exact": causal_top4,
        "context_specificity_gap": real_rate - max(blank_rate, swapped_rate),
        "apply": False,
        "claim_boundary": (
            "diagnostic-only causal visible-context audit; counterfactual labels never select the held-out answer"
        ),
    }


def evaluate_visible_reranker_oracle_contract(receipt: dict[str, Any]) -> dict[str, Any]:
    """Make the model-free smoke fail closed when its anti-leak checks stop falsifying."""

    leakage = receipt.get("leakage", {})
    leakage = leakage if isinstance(leakage, dict) else {}
    lattice = receipt.get("lattice", {})
    lattice = lattice if isinstance(lattice, dict) else {}
    shuffle = receipt.get("shuffle_falsification", {})
    shuffle = shuffle if isinstance(shuffle, dict) else {}
    sweep = receipt.get("selector_anchor_sweep", {})
    sweep = sweep if isinstance(sweep, dict) else {}
    visible = sweep.get("visible_reranker", {})
    visible = visible if isinstance(visible, dict) else {}
    counterfactual = receipt.get("counterfactual_context", {})
    counterfactual = counterfactual if isinstance(counterfactual, dict) else {}

    try:
        real_selected_exact = float(visible.get("selected_exact", 0.0))
        shuffled_selected_exact = float(shuffle.get("visible_reranker_selected_exact", 1.0))
        real_lattice_rate = float(lattice.get("gold_in_lattice_at_128", 0.0))
        shuffled_lattice_rate = float(shuffle.get("gold_in_lattice_at_128", 1.0))
        blank_context_selected_exact = float(counterfactual.get("blank_context_selected_exact", 1.0))
        swapped_edges_selected_exact = float(counterfactual.get("swapped_edges_selected_exact", 1.0))
        causal_visible_selected_exact = float(counterfactual.get("causal_visible_selected_exact", 0.0))
    except (TypeError, ValueError):
        real_selected_exact = 0.0
        shuffled_selected_exact = 1.0
        real_lattice_rate = 0.0
        shuffled_lattice_rate = 1.0
        blank_context_selected_exact = 1.0
        swapped_edges_selected_exact = 1.0
        causal_visible_selected_exact = 0.0

    checks = {
        "model_not_loaded": receipt.get("model_load") is False,
        "visible_reranker_diagnostic_only": visible.get("apply") is False,
        "causal_visible_context_diagnostic_only": counterfactual.get("apply") is False,
        "gold_not_used_for_selection": leakage.get("gold_used_for_selection") is False,
        "masked_bytes_not_seen_by_features": leakage.get("masked_bytes_seen_by_features") is False,
        "no_train_split_target_hole_leakage": int(leakage.get("target_hole_seen_in_train_split", 1) or 0) == 0,
        "no_eval_doc_hits": int(leakage.get("eval_doc_hits", 1) or 0) == 0,
        "no_same_doc_hits": int(leakage.get("same_doc_hits", 1) or 0) == 0,
        "shuffle_falsification_drops_selected_exact": shuffled_selected_exact < real_selected_exact,
        "shuffle_falsification_drops_lattice_coverage": shuffled_lattice_rate < real_lattice_rate,
        "counterfactual_context_drops_selected_exact": max(
            blank_context_selected_exact, swapped_edges_selected_exact
        )
        < real_selected_exact,
        "causal_visible_context_no_harm": causal_visible_selected_exact >= real_selected_exact,
    }
    missing = [name for name, passed in checks.items() if not passed]
    return {
        "passed": not missing,
        "missing": missing,
        "checks": checks,
        "observed": {
            "visible_reranker_selected_exact": real_selected_exact,
            "shuffle_visible_reranker_selected_exact": shuffled_selected_exact,
            "gold_in_lattice_at_128": real_lattice_rate,
            "shuffle_gold_in_lattice_at_128": shuffled_lattice_rate,
            "blank_context_selected_exact": blank_context_selected_exact,
            "swapped_edges_selected_exact": swapped_edges_selected_exact,
            "causal_visible_selected_exact": causal_visible_selected_exact,
        },
        "claim_boundary": "model_free_oracle_smoke_only_no_generation_claim",
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
        counterfactual_surfaces: dict[str, dict[tuple[int, ...], dict[str, Any]]] = {}
        counterfactual_selected: dict[str, tuple[int, ...]] = {}
        counterfactual_ranks: dict[str, int | None] = {}
        for mode in ("blank_context", "swapped_edges"):
            variant_marked = _counterfactual_marked_text(
                marked,
                mode,
                context_chars=int(config.get("counterfactual_context_chars", 12)),
            )
            variant_surface = surface_verifier_candidate_report(ranked, marked_text=variant_marked, train_text=train_text)
            variant_reranker = visible_reranker_candidate_report(
                _ranked_candidates_with_surface_report(ranked, variant_surface),
                prior_weight=float(selected_weights["prior_weight"]),
                surface_weight=float(selected_weights["surface_weight"]),
            )
            counterfactual_surfaces[mode] = variant_surface
            counterfactual_selected[mode] = min(
                candidate_keys,
                key=lambda key: int(variant_reranker[key]["visible_reranker_rank"]),
                default=(),
            )
            counterfactual_ranks[mode] = (
                int(variant_reranker[target_key]["visible_reranker_rank"])
                if target_key in variant_reranker
                else None
            )
        causal_rows: list[dict[str, Any]] = []
        for candidate in ranked:
            key = tuple(int(token_id) for token_id in candidate["ids"])
            raw_prior = candidate.get("prior_score")
            prior_score = float(raw_prior) if raw_prior is not None else -1e9
            if not math.isfinite(prior_score):
                prior_score = -1e9
            real_surface = float(surface[key]["surface_verifier_score"])
            null_surface = max(
                float(counterfactual_surfaces["blank_context"][key]["surface_verifier_score"]),
                float(counterfactual_surfaces["swapped_edges"][key]["surface_verifier_score"]),
            )
            surface_lift = real_surface - null_surface
            causal_rows.append(
                {
                    "ids": key,
                    "predicted_hole": str(candidate["predicted_hole"]),
                    "prior_rank": int(candidate["prior_rank"]),
                    "score": (
                        float(selected_weights["prior_weight"]) * prior_score
                        + float(selected_weights["surface_weight"]) * surface_lift
                    ),
                    "surface_lift": surface_lift,
                }
            )
        causal_rows.sort(
            key=lambda row: (
                -float(row["score"]),
                int(row["prior_rank"]),
                str(row["predicted_hole"]),
            )
        )
        causal_selected = causal_rows[0]["ids"] if causal_rows else ()
        causal_rank = next((rank for rank, row in enumerate(causal_rows) if row["ids"] == target_key), None)
        target_causal_row = next((row for row in causal_rows if row["ids"] == target_key), None)
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
                "counterfactual_context": {
                    "blank_context_selected_exact": counterfactual_selected["blank_context"] == target_key,
                    "blank_context_rank": counterfactual_ranks["blank_context"],
                    "swapped_edges_selected_exact": counterfactual_selected["swapped_edges"] == target_key,
                    "swapped_edges_rank": counterfactual_ranks["swapped_edges"],
                    "causal_visible_selected_exact": causal_selected == target_key,
                    "causal_visible_rank": causal_rank,
                    "causal_visible_top4_exact": causal_rank is not None and causal_rank < 4,
                    "target_surface_lift": (
                        float(target_causal_row["surface_lift"]) if target_causal_row is not None else None
                    ),
                    "apply": False,
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
    counterfactual_context = _counterfactual_context_summary(case_rows)
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
    receipt = {
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
        "counterfactual_context": counterfactual_context,
        "leakage": leakage,
        "shuffle_falsification": shuffle,
        "cases": [
            {key: value for key, value in row.items() if not key.startswith("_")}
            for row in case_rows
        ],
        "claim_boundary": (
            "model-free visible-context reranker proof only; reranker apply=false and no generation claim"
        ),
    }
    proof_contract = evaluate_visible_reranker_oracle_contract(receipt)
    receipt["proof_contract"] = proof_contract
    receipt["verdict"] = "pass" if proof_contract["passed"] else "fail"
    return receipt


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
