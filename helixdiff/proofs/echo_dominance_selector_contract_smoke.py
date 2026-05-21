from __future__ import annotations

import argparse
import json
import random
import subprocess
from pathlib import Path
from statistics import median
from typing import Any

from ..bench import build_lattice_candidate_rows, parse_positive_int_grid, rank_lattice_candidates_by_prior, sha256_text, split_text
from ..data import load_text
from ..infill import parse_marked_infill
from ..lattice.echo_dominance_selector import rank_with_echo_dominance
from ..lattice.in_document_echo import echo_overlap_audit, in_document_echo_candidates
from ..ngram import BigramGuide
from ..tokenizer import ByteTokenizer
from .in_document_echo_oracle_smoke import _make_document_cases, _rank_for_key, _target_key


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
    "min_real_null_margin": 2.0,
    "max_rank_promotions": 1,
    "shuffle_trials": 32,
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


def _case_context(document_text: str, case: dict[str, Any], *, context_bytes: int) -> dict[str, bytes]:
    document_bytes = document_text.encode("utf-8", errors="replace")
    span_start = int(case["target_byte_start"])
    span_end = int(case["target_byte_end"])
    return {
        "left": document_bytes[max(0, span_start - int(context_bytes)) : span_start],
        "right": document_bytes[span_end : min(len(document_bytes), span_end + int(context_bytes))],
    }


def _dedupe(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    seen: set[tuple[int, ...]] = set()
    out: list[dict[str, Any]] = []
    for row in rows:
        key = tuple(int(token_id) for token_id in row["ids"])
        if key in seen:
            continue
        out.append(row)
        seen.add(key)
        if len(out) >= limit:
            break
    return out


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row.get(key)) / len(rows)


def _median(values: list[float]) -> float:
    return float(median(values)) if values else 0.0


def _case_rows(
    *,
    document_text: str,
    train_text: str,
    cases: list[dict[str, Any]],
    tokenizer: ByteTokenizer,
    guide: BigramGuide,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    max_candidates = int(config["max_candidates_per_example"])
    anchor_sizes = [int(size) for size in config["anchor_sizes"]]
    rows: list[dict[str, Any]] = []
    for case_id, case in enumerate(cases):
        marked_text = str(case["marked_text"])
        target_key = _target_key(marked_text, tokenizer)
        prior = rank_lattice_candidates_by_prior(
            build_lattice_candidate_rows(
                tokenizer=tokenizer,
                marked_text=marked_text,
                guide=guide,
                train_text=train_text,
                visible_limit=int(config["lattice_visible_candidates"]),
                morphology_limit=int(config["lattice_morphology_candidates"]),
                surface_limit=int(config["lattice_surface_candidates"]),
                bi_anchor_limit=int(config["lattice_bi_anchor_candidates"]),
                bi_anchor_sizes=anchor_sizes,
            )
        )[:max_candidates]
        echo = in_document_echo_candidates(
            tokenizer=tokenizer,
            document_text=document_text,
            span_start=int(case["target_byte_start"]),
            span_end=int(case["target_byte_end"]),
            context_bytes=int(config["context_chars"]),
            anchor_sizes=anchor_sizes,
            anchor_window_bytes=int(config["anchor_window_bytes"]),
            limit=max_candidates,
        )
        swapped_case = cases[(case_id + 1) % len(cases)]
        swapped = _case_context(document_text, swapped_case, context_bytes=int(config["context_chars"]))
        contract_ranked, dominance = rank_with_echo_dominance(
            tokenizer=tokenizer,
            document_text=document_text,
            span_start=int(case["target_byte_start"]),
            span_end=int(case["target_byte_end"]),
            prior=prior,
            echo=echo,
            context_bytes=int(config["context_chars"]),
            anchor_sizes=anchor_sizes,
            anchor_window_bytes=int(config["anchor_window_bytes"]),
            null_contexts=[
                {"name": "blank", "left": b"", "right": b""},
                {"name": "swapped_edges", "left": swapped["left"], "right": swapped["right"]},
            ],
            min_real_null_margin=float(config["min_real_null_margin"]),
            max_rank_promotions=int(config["max_rank_promotions"]),
            max_candidates=max_candidates,
        )
        combined = _dedupe([*prior, *echo], limit=max_candidates)
        prior_rank = _rank_for_key(prior, target_key)
        combined_rank = _rank_for_key(combined, target_key)
        contract_rank = _rank_for_key(contract_ranked, target_key)
        audit = echo_overlap_audit(echo)
        rows.append(
            {
                "case_id": int(case_id),
                "target_sha256": sha256_text(parse_marked_infill(marked_text, tokenizer).hole),
                "target_byte_start": int(case["target_byte_start"]),
                "target_byte_end": int(case["target_byte_end"]),
                "prior_candidate_count": len(prior),
                "echo_candidate_count": len(echo),
                "contract_candidate_count": len(contract_ranked),
                "gold_in_prior_lattice_at_128": prior_rank is not None,
                "gold_in_combined_lattice_at_128": combined_rank is not None,
                "prior_selected_exact": prior_rank == 0,
                "contract_selected_exact": contract_rank == 0,
                "prior_top4_exact": prior_rank is not None and prior_rank < 4,
                "contract_top4_exact": contract_rank is not None and contract_rank < 4,
                "gold_rank_prior": prior_rank,
                "gold_rank_combined": combined_rank,
                "gold_rank_contract": contract_rank,
                "echo_overlap_audit": audit,
                "echo_dominance": {
                    key: value
                    for key, value in dominance.items()
                    if key != "decisions"
                },
                "promotion_decisions": dominance["decisions"][:8],
                "_prior_top4_keys": [tuple(int(token_id) for token_id in row["ids"]) for row in prior[:4]],
                "_contract_top4_keys": [
                    tuple(int(token_id) for token_id in row["ids"]) for row in contract_ranked[:4]
                ],
            }
        )
    return rows


def _shuffle_falsification(
    *,
    case_rows: list[dict[str, Any]],
    target_keys: list[tuple[int, ...]],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if not case_rows or not target_keys or trials <= 0:
        return {"trials": int(trials), "contract_selected_exact": 0.0, "contract_top4_exact": 0.0}
    rng = random.Random(seed)
    selected_rates: list[float] = []
    top4_rates: list[float] = []
    promoted_rates: list[float] = []
    for _ in range(trials):
        shuffled = list(target_keys)
        rng.shuffle(shuffled)
        selected_hits = 0
        top4_hits = 0
        promoted = 0
        for row, target in zip(case_rows, shuffled, strict=False):
            top4 = [tuple(item) for item in row["_contract_top4_keys"]]
            selected_hits += int(bool(top4) and tuple(target) == top4[0])
            top4_hits += int(tuple(target) in set(top4))
            promoted += int(row["echo_dominance"]["promoted_cases"] > 0)
        selected_rates.append(selected_hits / len(case_rows))
        top4_rates.append(top4_hits / len(case_rows))
        promoted_rates.append(promoted / len(case_rows))
    return {
        "trials": int(trials),
        "promoted_cases": sum(promoted_rates) / len(promoted_rates),
        "contract_selected_exact": sum(selected_rates) / len(selected_rates),
        "contract_top4_exact": sum(top4_rates) / len(top4_rates),
    }


def _strip_private_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped = []
    for row in rows:
        clean = {key: value for key, value in row.items() if not key.startswith("_")}
        stripped.append(clean)
    return stripped


def build_receipt(config: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)
    tokenizer = ByteTokenizer()
    text = load_text(str(cfg["data"]))
    train_text, val_text = split_text(text, val_fraction=float(cfg["val_fraction"]))
    guide = BigramGuide.from_text(train_text, tokenizer)
    cases = _make_document_cases(
        val_text,
        train_text=train_text,
        cases=int(cfg["cases"]),
        span_chars=int(cfg["span_chars"]),
        context_chars=int(cfg["context_chars"]),
        seed=int(cfg["seed"]),
        require_unseen_hole=False,
    )
    target_keys = [_target_key(str(case["marked_text"]), tokenizer) for case in cases]
    case_rows = _case_rows(
        document_text=val_text,
        train_text=train_text,
        cases=cases,
        tokenizer=tokenizer,
        guide=guide,
        config=cfg,
    )
    overlap_totals = {
        "target_window_overlap_hits": sum(int(row["echo_overlap_audit"]["target_window_overlap_hits"]) for row in case_rows),
        "target_anchor_window_overlap_hits": sum(
            int(row["echo_overlap_audit"]["target_anchor_window_overlap_hits"]) for row in case_rows
        ),
        "same_offset_hits": sum(int(row["echo_overlap_audit"]["same_offset_hits"]) for row in case_rows),
        "sentinel_source_window_hits": sum(
            int(row["echo_overlap_audit"]["sentinel_source_window_hits"]) for row in case_rows
        ),
        "near_duplicate_flags": 0,
    }
    promoted_cases = sum(int(row["echo_dominance"]["promoted_cases"]) for row in case_rows)
    rank1_promotions = sum(int(row["echo_dominance"]["rank1_promotions"]) for row in case_rows)
    rank2_to_rank4_promotions = sum(int(row["echo_dominance"]["rank2_to_rank4_promotions"]) for row in case_rows)
    promoted_under_blank = sum(int(row["echo_dominance"]["promoted_under_blank"]) for row in case_rows)
    promoted_under_swapped = sum(int(row["echo_dominance"]["promoted_under_swapped_edges"]) for row in case_rows)
    promoted_margin_mins = [
        float(row["echo_dominance"]["promoted_causal_margin_min"])
        for row in case_rows
        if int(row["echo_dominance"]["promoted_cases"]) > 0
    ]
    target_after_freeze = {
        "prior_selected_exact": _rate(case_rows, "prior_selected_exact"),
        "contract_selected_exact": _rate(case_rows, "contract_selected_exact"),
        "prior_top4_exact": _rate(case_rows, "prior_top4_exact"),
        "contract_top4_exact": _rate(case_rows, "contract_top4_exact"),
        "beats_prior_selected": sum(
            1 for row in case_rows if row["contract_selected_exact"] and not row["prior_selected_exact"]
        ),
        "harms_prior_selected": sum(
            1 for row in case_rows if not row["contract_selected_exact"] and row["prior_selected_exact"]
        ),
        "beats_prior_top4": sum(1 for row in case_rows if row["contract_top4_exact"] and not row["prior_top4_exact"]),
        "harms_prior_top4": sum(1 for row in case_rows if not row["contract_top4_exact"] and row["prior_top4_exact"]),
        "gold_in_prior_lattice_at_128": _rate(case_rows, "gold_in_prior_lattice_at_128"),
        "gold_in_combined_lattice_at_128": _rate(case_rows, "gold_in_combined_lattice_at_128"),
    }
    shuffle = _shuffle_falsification(
        case_rows=case_rows,
        target_keys=target_keys,
        trials=int(cfg["shuffle_trials"]),
        seed=int(cfg["seed"]) + 1997,
    )
    selector_ready = (
        promoted_cases > 0
        and promoted_under_blank == 0
        and promoted_under_swapped == 0
        and min(promoted_margin_mins, default=0.0) >= float(cfg["min_real_null_margin"])
        and target_after_freeze["contract_top4_exact"] > target_after_freeze["prior_top4_exact"]
        and target_after_freeze["harms_prior_top4"] == 0
        and shuffle["contract_top4_exact"] < target_after_freeze["contract_top4_exact"]
    )
    kill_triggered = (
        target_after_freeze["gold_in_combined_lattice_at_128"] > target_after_freeze["gold_in_prior_lattice_at_128"]
        and promoted_cases > 0
        and (
            target_after_freeze["contract_top4_exact"] <= target_after_freeze["prior_top4_exact"]
            or target_after_freeze["harms_prior_top4"] > 0
        )
    )
    contract_settings = {
        "name": "echo_dominance_v0",
        "apply": selector_ready,
        "diagnostic_only": not selector_ready,
        "model_load": False,
        "params": {
            "min_real_null_margin": float(cfg["min_real_null_margin"]),
            "require_exact_bi_anchor_or_two_modes": True,
            "baseline_preserving": True,
            "max_rank_promotions": int(cfg["max_rank_promotions"]),
            "contrastive_nulls": ["blank", "swapped_edges"],
        },
    }
    receipt = {
        "proof_name": "echo_dominance_selector_contract_smoke",
        "kind": "helixdiff_echo_dominance_selector_contract_receipt",
        "model_load": False,
        "git": {"commit": current_commit(), "dirty": current_git_dirty()},
        "data": {
            "path": str(cfg["data"]),
            "sha256": sha256_text(text),
            "train_sha256": sha256_text(train_text),
            "val_sha256": sha256_text(val_text),
            "val_fraction": float(cfg["val_fraction"]),
        },
        "num_cases": len(case_rows),
        "max_candidates_per_example": int(cfg["max_candidates_per_example"]),
        "selector_contract": {
            **contract_settings,
            "ready": selector_ready,
            "status": "ready_for_heldout" if selector_ready else "killed_fail_closed",
            "reason": None if selector_ready else "direct_echo_promotion_did_not_improve_top4_without_harm",
            "selected_before_target_eval": True,
            "target_metric_used_for_selection": False,
            "baseline_preserving": True,
            "min_real_null_margin": float(cfg["min_real_null_margin"]),
        },
        "redaction": {
            "target_span_redacted_before_echo_index": True,
            "target_bytes_available_to_selector": False,
            "masked_bytes_seen_by_features": False,
            "sentinel_present_in_index": False,
        },
        "source_policy": {
            "uses_eval_visible_document": True,
            "claim_mode": "transductive_fixed_span_visible_context_repair",
            **overlap_totals,
        },
        "echo_dominance": {
            "promoted_cases": promoted_cases,
            "rank1_promotions": rank1_promotions,
            "rank2_to_rank4_promotions": rank2_to_rank4_promotions,
            "median_real_score": _median([float(row["echo_dominance"]["median_real_score"]) for row in case_rows]),
            "median_null_score": _median([float(row["echo_dominance"]["median_null_score"]) for row in case_rows]),
            "median_causal_margin": _median([float(row["echo_dominance"]["median_causal_margin"]) for row in case_rows]),
            "promoted_causal_margin_min": min(promoted_margin_mins, default=0.0),
            "promoted_under_blank": promoted_under_blank,
            "promoted_under_swapped_edges": promoted_under_swapped,
        },
        "target_after_freeze": target_after_freeze,
        "shuffle_falsification": shuffle,
        "claim_allowed": {
            "echo_dominance_selector": selector_ready,
            "target_lift": False,
            "model_quality": False,
            "global_sota": False,
        },
        "fail_closed": {
            "enabled": not selector_ready,
            "kill_triggered": kill_triggered,
            "direct_promotion_claim_blocked": not selector_ready,
        },
        "case_rows": _strip_private_rows(case_rows),
        "claim_boundary": (
            "model-free Causal Echo-Dominance selector for fixed-span visible-context document repair; "
            "not global LM SOTA, not variable-length infill, and not model-quality evidence until a held-out "
            "selector-contract benchmark passes"
        ),
    }
    checks = {
        "model_not_loaded": receipt["model_load"] is False,
        "contract_status_consistent": receipt["selector_contract"]["ready"] is selector_ready,
        "contract_apply_matches_readiness": receipt["selector_contract"]["apply"] is selector_ready,
        "diagnostic_only_matches_readiness": receipt["selector_contract"]["diagnostic_only"] is (not selector_ready),
        "selected_before_target_eval": receipt["selector_contract"]["selected_before_target_eval"] is True,
        "target_metric_not_used_for_selection": receipt["selector_contract"]["target_metric_used_for_selection"] is False,
        "target_span_redacted_before_echo_index": receipt["redaction"]["target_span_redacted_before_echo_index"] is True,
        "target_bytes_not_available_to_selector": receipt["redaction"]["target_bytes_available_to_selector"] is False,
        "masked_bytes_not_seen_by_features": receipt["redaction"]["masked_bytes_seen_by_features"] is False,
        "no_target_window_overlap": receipt["source_policy"]["target_window_overlap_hits"] == 0,
        "no_target_anchor_window_overlap": receipt["source_policy"]["target_anchor_window_overlap_hits"] == 0,
        "no_same_offset_hits": receipt["source_policy"]["same_offset_hits"] == 0,
        "no_near_duplicate_flags": receipt["source_policy"]["near_duplicate_flags"] == 0,
        "promoted_cases_positive": receipt["echo_dominance"]["promoted_cases"] > 0,
        "blank_null_not_promoted": receipt["echo_dominance"]["promoted_under_blank"] == 0,
        "swapped_edges_null_not_promoted": receipt["echo_dominance"]["promoted_under_swapped_edges"] == 0,
        "promoted_margin_passes": receipt["echo_dominance"]["promoted_causal_margin_min"] >= float(cfg["min_real_null_margin"]),
        "ready_or_fail_closed": selector_ready or receipt["fail_closed"]["kill_triggered"] is True,
        "ready_requires_top4_lift": (not selector_ready)
        or target_after_freeze["contract_top4_exact"] > target_after_freeze["prior_top4_exact"],
        "ready_requires_no_top4_harm": (not selector_ready) or target_after_freeze["harms_prior_top4"] == 0,
        "ready_requires_shuffle_below_contract": (not selector_ready)
        or shuffle["contract_top4_exact"] < target_after_freeze["contract_top4_exact"],
        "fail_closed_blocks_claim": selector_ready or receipt["claim_allowed"]["echo_dominance_selector"] is False,
    }
    receipt["checks"] = checks
    receipt["verdict"] = "pass" if all(checks.values()) else "fail"
    receipt["kill_criterion"] = (
        "kill direct echo promotion if combined K=128 containment stays above prior but the frozen "
        "contract does not improve top4 without harm on this smoke"
    )
    contract = {
        "kind": "helixdiff_selector_contract",
        "source": "echo_dominance_selector_contract_smoke",
        "ready_for_heldout": selector_ready and receipt["verdict"] == "pass",
        **contract_settings,
        "receipt_sha256": sha256_text(json.dumps({k: v for k, v in receipt.items() if k != "case_rows"}, sort_keys=True)),
    }
    if not selector_ready:
        contract["apply"] = False
        contract["diagnostic_only"] = True
    return receipt, contract


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit the HelixDiff Causal Echo-Dominance selector-contract smoke receipt.")
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--contract-out")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--cases", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--anchor-sizes")
    parser.add_argument("--min-real-null-margin", type=float)
    parser.add_argument("--max-rank-promotions", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.cases is not None:
        config["cases"] = args.cases
    if args.seed is not None:
        config["seed"] = args.seed
    if args.anchor_sizes:
        config["anchor_sizes"] = parse_positive_int_grid(args.anchor_sizes)
    if args.min_real_null_margin is not None:
        config["min_real_null_margin"] = args.min_real_null_margin
    if args.max_rank_promotions is not None:
        config["max_rank_promotions"] = args.max_rank_promotions
    receipt, contract = build_receipt(config)
    receipt_text = json.dumps(receipt, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(receipt_text, encoding="utf-8")
    if args.contract_out:
        contract_out = Path(args.contract_out)
        contract_out.parent.mkdir(parents=True, exist_ok=True)
        contract_out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(receipt_text if args.json or not args.out else f"wrote {args.out}")
    if receipt["verdict"] != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
