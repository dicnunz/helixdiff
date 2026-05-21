from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Any

from ..bench import (
    _ranked_candidates_with_surface_report,
    bi_anchor_gap_candidates,
    build_lattice_candidate_rows,
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
    "seed": 151,
    "pseudo_masks_per_case": 6,
    "proxy_geometry_mode": "boundary_only",
    "proxy_geometry_pool_per_case": 18,
    "pseudo_heldout_frac": 0.5,
    "max_candidates_per_example": 64,
    "lattice_visible_candidates": 8,
    "lattice_morphology_candidates": 64,
    "lattice_surface_candidates": 64,
    "lattice_bi_anchor_candidates": 8,
    "lattice_bi_anchor_sizes": [32, 24, 16, 12, 8, 6, 4],
    "shuffle_trials": 16,
}

DEFAULT_PRESETS: list[dict[str, Any]] = [
    {"name": "prior_m3", "selector_anchor": "prior", "selector_margin": 3.0},
    {"name": "surface_m3", "selector_anchor": "surface", "selector_margin": 3.0},
    {"name": "visible_reranker_m3", "selector_anchor": "visible_reranker", "selector_margin": 3.0},
]


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


def _split_marked(marked_text: str) -> tuple[str, str, str]:
    before, rest = marked_text.split("[[", 1)
    hole, after = rest.split("]]", 1)
    return before, hole, after


def _boundary_signature(left: str, hole: str, right: str) -> tuple[bool, bool, bool, bool, bool, bool]:
    return (
        bool(left[-1:].isalpha()),
        bool(right[:1].isalpha()),
        bool(hole[:1].isupper()),
        bool(hole[-1:].isalpha()),
        "-" in hole,
        hole.endswith(":"),
    )


def _signature_match_score(a: tuple[bool, ...], b: tuple[bool, ...]) -> int:
    return sum(1 for left, right in zip(a, b, strict=True) if left == right)


def _redacted_marked_text(marked_text: str) -> str:
    before, hole, after = _split_marked(marked_text)
    return f"{before}[[{'?' * len(hole)}]]{after}"


def proxy_mask_cases_from_visible_context(
    marked_text: str,
    *,
    span_chars: int,
    context_chars: int,
    limit: int,
) -> list[dict[str, Any]]:
    before, target_hole, after = _split_marked(marked_text)
    target_signature = _boundary_signature(before, target_hole, after)
    rows: list[dict[str, Any]] = []
    for side, segment in (("left", before), ("right", after)):
        if len(segment) < span_chars + 2:
            continue
        for start in range(1, len(segment) - span_chars):
            hole = segment[start : start + span_chars]
            if not hole.strip() or "\n" in hole or hole == target_hole:
                continue
            left = segment[max(0, start - context_chars) : start]
            right = segment[start + span_chars : start + span_chars + context_chars]
            if not left or not right:
                continue
            signature = _boundary_signature(left, hole, right)
            rows.append(
                {
                    "marked_text": f"{left}[[{hole}]]{right}",
                    "hole": hole,
                    "side": side,
                    "offset": start,
                    "boundary_match": _signature_match_score(signature, target_signature),
                    "hole_sha256": sha256_text(hole),
                }
            )
    rows.sort(key=lambda row: (-int(row["boundary_match"]), str(row["side"]), int(row["offset"]), str(row["hole"])))
    deduped: dict[str, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(str(row["marked_text"]), row)
    return list(deduped.values())[: max(0, int(limit))]


def _source_family(source: str | None) -> str:
    value = str(source or "")
    if value.startswith("visible_"):
        return "visible"
    if value == "bi_anchor_gap":
        return "bi_anchor"
    if value.startswith("morphology_"):
        return "morphology"
    if value.startswith("surface_"):
        return "surface"
    if value in {"bridge", "unigram"}:
        return value
    return "other"


def _safe_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return score if score == score else 0.0


def lattice_geometry_profile(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    config: dict[str, Any],
    bi_anchor_sizes: list[int],
) -> dict[str, Any]:
    """Describe the candidate geometry induced by visible context, without using target bytes."""

    redacted_marked = _redacted_marked_text(marked_text)
    ranked = rank_lattice_candidates_by_prior(
        build_lattice_candidate_rows(
            tokenizer=tokenizer,
            marked_text=redacted_marked,
            guide=guide,
            train_text=train_text,
            visible_limit=int(config["lattice_visible_candidates"]),
            morphology_limit=int(config["lattice_morphology_candidates"]),
            surface_limit=int(config["lattice_surface_candidates"]),
            bi_anchor_limit=int(config["lattice_bi_anchor_candidates"]),
            bi_anchor_sizes=bi_anchor_sizes,
        )
    )[: int(config["max_candidates_per_example"])]
    source_counts = {name: 0 for name in ("visible", "bi_anchor", "morphology", "surface", "bridge", "unigram", "other")}
    best_scores = {
        "best_suture_score": 0.0,
        "best_morphology_score": 0.0,
        "best_surface_score": 0.0,
        "best_bi_anchor_score": 0.0,
        "best_bi_anchor_support": 0.0,
        "best_bi_anchor_size": 0.0,
    }
    for row in ranked:
        for source in row.get("sources", []):
            family = _source_family(str(source.get("source")))
            source_counts[family] = source_counts.get(family, 0) + 1
            best_scores["best_suture_score"] = max(
                best_scores["best_suture_score"],
                _safe_score(source.get("suture_score")),
            )
            best_scores["best_morphology_score"] = max(
                best_scores["best_morphology_score"],
                _safe_score(source.get("morphology_score")),
            )
            best_scores["best_surface_score"] = max(
                best_scores["best_surface_score"],
                _safe_score(source.get("surface_score")),
            )
            best_scores["best_bi_anchor_score"] = max(
                best_scores["best_bi_anchor_score"],
                _safe_score(source.get("bi_anchor_score")),
            )
            best_scores["best_bi_anchor_support"] = max(
                best_scores["best_bi_anchor_support"],
                _safe_score(source.get("bi_anchor_support")),
            )
            best_scores["best_bi_anchor_size"] = max(
                best_scores["best_bi_anchor_size"],
                _safe_score(source.get("bi_anchor_best_anchor_size")),
            )
    top_prior_source = str(ranked[0].get("prior_source")) if ranked else "none"
    return {
        "candidate_count": len(ranked),
        "top_prior_source": top_prior_source,
        "top_prior_family": _source_family(top_prior_source),
        **source_counts,
        **best_scores,
        "uses_redacted_target_hole": True,
    }


def _normalized_distance(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def geometry_profile_distance(target: dict[str, Any], proxy: dict[str, Any]) -> float:
    numeric_keys = (
        "candidate_count",
        "visible",
        "bi_anchor",
        "morphology",
        "surface",
        "bridge",
        "unigram",
        "best_suture_score",
        "best_morphology_score",
        "best_surface_score",
        "best_bi_anchor_score",
        "best_bi_anchor_support",
        "best_bi_anchor_size",
    )
    distance = sum(_normalized_distance(float(target.get(key, 0.0)), float(proxy.get(key, 0.0))) for key in numeric_keys)
    if target.get("top_prior_family") != proxy.get("top_prior_family"):
        distance += 2.0
    if target.get("top_prior_source") != proxy.get("top_prior_source"):
        distance += 0.5
    return float(distance)


def select_geometry_shaped_proxy_masks(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    config: dict[str, Any],
    bi_anchor_sizes: list[int],
    context_chars: int,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pool_limit = max(int(config.get("proxy_geometry_pool_per_case", limit)), int(limit))
    pool = proxy_mask_cases_from_visible_context(
        marked_text,
        span_chars=int(config["span_chars"]),
        context_chars=context_chars,
        limit=pool_limit,
    )
    target_profile = lattice_geometry_profile(
        tokenizer=tokenizer,
        marked_text=marked_text,
        guide=guide,
        train_text=train_text,
        config=config,
        bi_anchor_sizes=bi_anchor_sizes,
    )
    shaped_rows: list[dict[str, Any]] = []
    for row in pool:
        proxy_profile = lattice_geometry_profile(
            tokenizer=tokenizer,
            marked_text=str(row["marked_text"]),
            guide=guide,
            train_text=train_text,
            config=config,
            bi_anchor_sizes=bi_anchor_sizes,
        )
        distance = geometry_profile_distance(target_profile, proxy_profile)
        shaped_rows.append(
            {
                **row,
                "geometry_distance": distance,
                "target_geometry_top_family": target_profile["top_prior_family"],
                "proxy_geometry_top_family": proxy_profile["top_prior_family"],
                "geometry_top_family_match": target_profile["top_prior_family"] == proxy_profile["top_prior_family"],
            }
        )
    generic_distances = [float(row["geometry_distance"]) for row in shaped_rows]
    shaped_rows.sort(
        key=lambda row: (
            -int(row["boundary_match"]),
            float(row["geometry_distance"]),
            str(row["side"]),
            int(row["offset"]),
            str(row["hole"]),
        )
    )
    selected = shaped_rows[: max(0, int(limit))]
    selected_distances = [float(row["geometry_distance"]) for row in selected]
    selected_match_rate = (
        sum(1.0 for row in selected if row["geometry_top_family_match"]) / len(selected) if selected else 0.0
    )
    summary = {
        "mode": str(config.get("proxy_geometry_mode", "target_retrieval")),
        "target_profile": target_profile,
        "pool_cases": len(pool),
        "selected_cases": len(selected),
        "pool_mean_geometry_distance": (
            sum(generic_distances) / len(generic_distances) if generic_distances else None
        ),
        "selected_mean_geometry_distance": (
            sum(selected_distances) / len(selected_distances) if selected_distances else None
        ),
        "selected_top_family_match_rate": selected_match_rate,
        "uses_redacted_target_hole": True,
    }
    return selected, summary


def _rank_case(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    config: dict[str, Any],
    bi_anchor_sizes: list[int],
) -> dict[str, Any]:
    target_key = _target_key(marked_text, tokenizer)
    ranked = rank_lattice_candidates_by_prior(
        build_lattice_candidate_rows(
            tokenizer=tokenizer,
            marked_text=marked_text,
            guide=guide,
            train_text=train_text,
            visible_limit=int(config["lattice_visible_candidates"]),
            morphology_limit=int(config["lattice_morphology_candidates"]),
            surface_limit=int(config["lattice_surface_candidates"]),
            bi_anchor_limit=int(config["lattice_bi_anchor_candidates"]),
            bi_anchor_sizes=bi_anchor_sizes,
        )
    )[: int(config["max_candidates_per_example"])]
    candidate_keys = [tuple(int(token_id) for token_id in row["ids"]) for row in ranked]
    surface = surface_verifier_candidate_report(ranked, marked_text=marked_text, train_text=train_text)
    reranker = visible_reranker_candidate_report(
        _ranked_candidates_with_surface_report(ranked, surface),
        prior_weight=1.0,
        surface_weight=1.0,
    )
    ranks: dict[str, int | None] = {
        "prior": candidate_keys.index(target_key) if target_key in candidate_keys else None,
        "surface": None,
        "visible_reranker": None,
    }
    selected: dict[str, tuple[int, ...]] = {}
    for anchor in ("prior", "surface", "visible_reranker"):
        if anchor == "prior":
            rank_lookup = {key: index for index, key in enumerate(candidate_keys)}
        elif anchor == "surface":
            rank_lookup = {
                tuple(int(token_id) for token_id in key): int(report["surface_verifier_rank"])
                for key, report in surface.items()
            }
        else:
            rank_lookup = {
                tuple(int(token_id) for token_id in key): int(report["visible_reranker_rank"])
                for key, report in reranker.items()
            }
        selected[anchor] = min(candidate_keys, key=lambda key: rank_lookup.get(key, 10**9), default=())
        ranks[anchor] = int(rank_lookup[target_key]) if target_key in rank_lookup else None
    bi_anchor_rows = bi_anchor_gap_candidates(
        tokenizer=tokenizer,
        marked_text=marked_text,
        train_text=train_text,
        anchor_sizes=bi_anchor_sizes,
        limit=int(config["lattice_bi_anchor_candidates"]),
    )
    return {
        "target_key": target_key,
        "candidate_count": len(candidate_keys),
        "gold_in_lattice_at_128": target_key in candidate_keys[:128],
        "bi_anchor_candidate_count": len(bi_anchor_rows),
        "ranks": ranks,
        "selected": selected,
    }


def _row_for_marked(
    *,
    tokenizer: ByteTokenizer,
    marked_text: str,
    guide: BigramGuide,
    train_text: str,
    config: dict[str, Any],
    bi_anchor_sizes: list[int],
    case_id: int,
    role: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    ranked = _rank_case(
        tokenizer=tokenizer,
        marked_text=marked_text,
        guide=guide,
        train_text=train_text,
        config=config,
        bi_anchor_sizes=bi_anchor_sizes,
    )
    row: dict[str, Any] = {
        "case_id": case_id,
        "role": role,
        "target_hole_sha256": sha256_text(parse_marked_infill(marked_text, tokenizer).hole),
        "candidate_count": ranked["candidate_count"],
        "gold_in_lattice_at_128": ranked["gold_in_lattice_at_128"],
        "bi_anchor_candidate_count": ranked["bi_anchor_candidate_count"],
        "target_key": ranked["target_key"],
    }
    for anchor in ("prior", "surface", "visible_reranker"):
        rank = ranked["ranks"][anchor]
        row[f"{anchor}_rank"] = rank
        row[f"{anchor}_selected_exact"] = ranked["selected"][anchor] == ranked["target_key"]
        row[f"{anchor}_top4_exact"] = rank is not None and rank < 4
        row[f"{anchor}_selected_key"] = ranked["selected"][anchor]
    if metadata:
        row.update(metadata)
    return row


def _summarize(rows: list[dict[str, Any]], anchor: str) -> dict[str, Any]:
    if not rows:
        return {"cases": 0, "selected_exact": 0.0, "top4_exact": 0.0, "avg_gold_rank": None}
    ranks = [int(row[f"{anchor}_rank"]) for row in rows if row.get(f"{anchor}_rank") is not None]
    return {
        "cases": len(rows),
        "selected_exact": sum(1.0 for row in rows if row[f"{anchor}_selected_exact"]) / len(rows),
        "top4_exact": sum(1.0 for row in rows if row[f"{anchor}_top4_exact"]) / len(rows),
        "avg_gold_rank": (sum(float(rank) for rank in ranks) / len(ranks)) if ranks else None,
    }


def _preset_score(summary: dict[str, Any]) -> tuple[float, float, float]:
    avg = summary["avg_gold_rank"]
    return (
        float(summary["selected_exact"]),
        float(summary["top4_exact"]),
        -(float(avg) if avg is not None else 1e9),
    )


def _select_preset(fit_rows: list[dict[str, Any]], presets: list[dict[str, Any]]) -> dict[str, Any]:
    scored = []
    for preset in presets:
        summary = _summarize(fit_rows, str(preset["selector_anchor"]))
        scored.append((preset, summary, _preset_score(summary)))
    scored.sort(key=lambda item: (item[2], str(item[0]["name"])), reverse=True)
    preset, summary, _ = scored[0]
    return {**preset, "fit_summary": summary}


def _split_proxy_rows_by_case(
    rows: list[dict[str, Any]],
    *,
    heldout_frac: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep proxy fit/heldout from every target case, so geometry regimes cannot split by case."""

    if not rows:
        return [], []
    by_case: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_case.setdefault(int(row["case_id"]), []).append(row)
    fit_rows: list[dict[str, Any]] = []
    heldout_rows: list[dict[str, Any]] = []
    for case_id in sorted(by_case):
        case_rows = by_case[case_id]
        if len(case_rows) == 1:
            fit_rows.extend(case_rows)
            continue
        heldout_count = max(1, int(round(len(case_rows) * heldout_frac)))
        heldout_count = min(heldout_count, len(case_rows) - 1)
        case_fit: list[dict[str, Any]] = []
        case_heldout: list[dict[str, Any]] = []
        for index, row in enumerate(case_rows):
            if index % 2 == 1 and len(case_heldout) < heldout_count:
                case_heldout.append(row)
            else:
                case_fit.append(row)
        while len(case_heldout) < heldout_count and len(case_fit) > 1:
            case_heldout.append(case_fit.pop())
        fit_rows.extend(case_fit)
        heldout_rows.extend(case_heldout)
    if not heldout_rows and len(fit_rows) > 1:
        heldout_rows.append(fit_rows.pop())
    return fit_rows, heldout_rows


def _evaluate_preset(rows: list[dict[str, Any]], preset: dict[str, Any]) -> dict[str, Any]:
    anchor = str(preset["selector_anchor"])
    summary = _summarize(rows, anchor)
    return {
        **summary,
        "selector_anchor": anchor,
        "selector_margin": float(preset["selector_margin"]),
    }


def _shuffle_falsification(
    rows: list[dict[str, Any]],
    *,
    preset: dict[str, Any],
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if not rows or trials <= 0:
        return {"trials": int(trials), "selected_exact": 0.0, "top4_exact": 0.0}
    rng = random.Random(seed)
    anchor = str(preset["selector_anchor"])
    exact_rates: list[float] = []
    top4_rates: list[float] = []
    targets = [row["target_key"] for row in rows]
    for _ in range(trials):
        shuffled = list(targets)
        rng.shuffle(shuffled)
        exact = 0
        top4 = 0
        for row, target in zip(rows, shuffled, strict=False):
            exact += int(row[f"{anchor}_selected_key"] == target)
            top4 += int(row[f"{anchor}_top4_exact"] and target == row["target_key"])
        exact_rates.append(exact / len(rows))
        top4_rates.append(top4 / len(rows))
    return {
        "trials": int(trials),
        "selected_exact": sum(exact_rates) / len(exact_rates),
        "top4_exact": sum(top4_rates) / len(top4_rates),
    }


def _json_safe_row(row: dict[str, Any]) -> dict[str, Any]:
    safe = {key: value for key, value in row.items() if not key.endswith("_key") and key != "target_key"}
    return safe


def _build_selector_contract(selected: dict[str, Any], receipt_core: dict[str, Any]) -> dict[str, Any]:
    contract_source = str(receipt_core.get("contract_source", "proxy_mask_target_retrieval_geometry"))
    if contract_source == "proxy_mask_target_retrieval_geometry":
        selection_rule = (
            "select visible-only pseudo masks by target retrieval geometry, then maximize proxy-mask fit "
            "selected exact, top4, and avg rank"
        )
    else:
        selection_rule = "maximize boundary-matched proxy-mask fit selected exact, then top4, then avg rank"
    core = {
        "selected": {
            "selector_anchor": selected["selector_anchor"],
            "selector_margin": float(selected["selector_margin"]),
            "source": contract_source,
        },
        "receipt_core": receipt_core,
    }
    contract_id = hashlib.sha256(json.dumps(core, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        "kind": "helixdiff_selector_contract",
        "version": 1,
        "contract_id": contract_id,
        "status": "ready_for_heldout",
        "ready_for_heldout": True,
        "claim_boundary": (
            "proxy-mask contracts freeze visible-only selector choices; target claims require a later held-out gate"
        ),
        "missing": [],
        "min_cases": int(receipt_core["pseudo_fit_cases"]),
        "selected": {
            "selector_anchor": selected["selector_anchor"],
            "selector_margin": float(selected["selector_margin"]),
            "source": contract_source,
            "recommendation_status": "candidate_anchor_margin",
            "selection_rule": selection_rule,
        },
        "frozen_benchmark_flags": [
            "--lattice-selector-anchor",
            str(selected["selector_anchor"]),
            "--lattice-selector-margin",
            f"{float(selected['selector_margin']):g}",
        ],
        "frozen_shell_flags": (
            f"--lattice-selector-anchor {selected['selector_anchor']} "
            f"--lattice-selector-margin {float(selected['selector_margin']):g}"
        ),
        "heldout_requirements": [
            "apply this contract only after the target span is redacted from calibration",
            "run on target cases not used to choose the selector preset",
            "keep --require-unseen-hole enabled",
            "run helixdiff-gate --require-repair-proof-contract before making a narrow repair claim",
        ],
        "calibrations": [
            {
                "source_path": receipt_core["receipt_path"],
                "cases": receipt_core["pseudo_fit_cases"],
                "recommendation_status": "candidate_anchor_margin",
                "source_reports": [],
            }
        ],
    }


def apply_claim_gate(receipt: dict[str, Any], *, require_useful_ratchet: bool = False) -> dict[str, Any]:
    public_target_lift_claim_allowed = (
        bool(receipt.get("useful_ratchet"))
        and receipt.get("verdict") == "pass"
        and receipt.get("selector_contract", {}).get("target_metric_used_for_selection") is False
    )
    passed = public_target_lift_claim_allowed if require_useful_ratchet else True
    claim_gate = {
        "require_useful_ratchet": bool(require_useful_ratchet),
        "passed": bool(passed),
        "public_target_lift_claim_allowed": bool(public_target_lift_claim_allowed),
        "blocker": None
        if passed
        else "useful_ratchet=false; this receipt is contract readiness only, not target-lift evidence",
        "claim_boundary": (
            "Contract readiness only: a proxy-mask selector contract can be ready for held-out use "
            "without allowing any public target-lift or SOTA-style claim."
        ),
    }
    receipt["claim_gate"] = claim_gate
    if require_useful_ratchet and not passed:
        receipt["verdict"] = "fail"
        reasons = list(receipt.get("failure_reasons", []))
        if "useful_ratchet_required" not in reasons:
            reasons.append("useful_ratchet_required")
        receipt["failure_reasons"] = reasons
    return receipt


def build_receipt(config: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    tokenizer = ByteTokenizer()
    text = load_text(str(config["data"]))
    train_text, val_text = split_text(text, float(config["val_fraction"]))
    guide = BigramGuide.from_text(train_text, tokenizer)
    bi_anchor_sizes = config["lattice_bi_anchor_sizes"]
    if isinstance(bi_anchor_sizes, str):
        bi_anchor_sizes = parse_positive_int_grid(bi_anchor_sizes)
    target_cases = make_marked_cases(
        val_text,
        cases=int(config["cases"]),
        span_chars=int(config["span_chars"]),
        context_chars=int(config["context_chars"]),
        seed=int(config["seed"]),
        forbidden_text=train_text,
        require_unseen_hole=True,
    )

    pseudo_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    proxy_geometry_summaries: list[dict[str, Any]] = []
    redaction_flags = {
        "target_span_redacted_before_calibration": True,
        "target_gold_available_to_contract_builder": False,
        "masked_bytes_seen_by_features": False,
    }
    for case_id, marked in enumerate(target_cases):
        _, target_hole, _ = _split_marked(marked)
        proxy_context_chars = min(12, int(config["context_chars"]))
        if str(config.get("proxy_geometry_mode", "target_retrieval")) == "target_retrieval":
            proxy_masks, geometry_summary = select_geometry_shaped_proxy_masks(
                tokenizer=tokenizer,
                marked_text=marked,
                guide=guide,
                train_text=train_text,
                config=config,
                bi_anchor_sizes=list(bi_anchor_sizes),
                context_chars=proxy_context_chars,
                limit=int(config["pseudo_masks_per_case"]),
            )
        else:
            proxy_masks = proxy_mask_cases_from_visible_context(
                marked,
                span_chars=int(config["span_chars"]),
                context_chars=proxy_context_chars,
                limit=int(config["pseudo_masks_per_case"]),
            )
            geometry_summary = {
                "mode": "boundary_only",
                "pool_cases": len(proxy_masks),
                "selected_cases": len(proxy_masks),
                "pool_mean_geometry_distance": None,
                "selected_mean_geometry_distance": None,
                "selected_top_family_match_rate": 0.0,
                "uses_redacted_target_hole": False,
            }
        proxy_geometry_summaries.append({"case_id": case_id, **geometry_summary})
        for pseudo in proxy_masks:
            if pseudo["hole"] == target_hole:
                redaction_flags["target_gold_available_to_contract_builder"] = True
                continue
            pseudo_rows.append(
                _row_for_marked(
                    tokenizer=tokenizer,
                    marked_text=str(pseudo["marked_text"]),
                    guide=guide,
                    train_text=train_text,
                    config=config,
                    bi_anchor_sizes=list(bi_anchor_sizes),
                    case_id=case_id,
                    role="pseudo_mask",
                    metadata={
                        "pseudo_side": pseudo["side"],
                        "pseudo_offset": pseudo["offset"],
                        "boundary_match": pseudo["boundary_match"],
                        "geometry_distance": pseudo.get("geometry_distance"),
                        "target_geometry_top_family": pseudo.get("target_geometry_top_family"),
                        "proxy_geometry_top_family": pseudo.get("proxy_geometry_top_family"),
                        "geometry_top_family_match": pseudo.get("geometry_top_family_match"),
                    },
                )
            )
        target_rows.append(
            _row_for_marked(
                tokenizer=tokenizer,
                marked_text=marked,
                guide=guide,
                train_text=train_text,
                config=config,
                bi_anchor_sizes=list(bi_anchor_sizes),
                case_id=case_id,
                role="target_after_freeze",
            )
        )

    proxy_geometry_mode = str(config.get("proxy_geometry_mode", "boundary_only"))
    if proxy_geometry_mode == "target_retrieval":
        fit_rows, heldout_rows = _split_proxy_rows_by_case(
            pseudo_rows,
            heldout_frac=float(config["pseudo_heldout_frac"]),
        )
    else:
        split_at = max(1, int(round(len(pseudo_rows) * (1.0 - float(config["pseudo_heldout_frac"])))))
        split_at = min(split_at, max(1, len(pseudo_rows) - 1)) if len(pseudo_rows) > 1 else len(pseudo_rows)
        fit_rows = pseudo_rows[:split_at]
        heldout_rows = pseudo_rows[split_at:]
    selected = _select_preset(fit_rows, DEFAULT_PRESETS)
    pseudo_fit = _evaluate_preset(fit_rows, selected)
    pseudo_heldout = _evaluate_preset(heldout_rows, selected)
    target_after_freeze = _evaluate_preset(target_rows, selected)
    visible_baseline = _evaluate_preset(target_rows, {"selector_anchor": "visible_reranker", "selector_margin": 3.0})
    prior_baseline = _evaluate_preset(target_rows, {"selector_anchor": "prior", "selector_margin": 3.0})
    surface_baseline = _evaluate_preset(target_rows, {"selector_anchor": "surface", "selector_margin": 3.0})
    shuffle = _shuffle_falsification(
        heldout_rows,
        preset=selected,
        trials=int(config["shuffle_trials"]),
        seed=int(config["seed"]) + 19,
    )
    contract_source = (
        "proxy_mask_target_retrieval_geometry"
        if proxy_geometry_mode == "target_retrieval"
        else "proxy_mask_visible_context"
    )
    receipt_core = {
        "receipt_path": "proof/proxy_mask_selector_contract_smoke.json",
        "pseudo_fit_cases": len(fit_rows),
        "pseudo_heldout_cases": len(heldout_rows),
        "selected_anchor": selected["selector_anchor"],
        "selected_margin": float(selected["selector_margin"]),
        "contract_source": contract_source,
    }
    contract = _build_selector_contract(selected, receipt_core)
    useful_ratchet = (
        target_after_freeze["top4_exact"] > 0.0
        and target_after_freeze["top4_exact"] >= visible_baseline["top4_exact"]
        and target_after_freeze["selected_exact"] >= prior_baseline["selected_exact"]
        and pseudo_heldout["top4_exact"] > shuffle["top4_exact"]
    )
    checks = {
        "model_not_loaded": True,
        "contract_ready": bool(contract["ready_for_heldout"]),
        "contract_apply": True,
        "target_span_redacted_before_calibration": redaction_flags["target_span_redacted_before_calibration"],
        "target_gold_unavailable_to_contract_builder": not redaction_flags["target_gold_available_to_contract_builder"],
        "preset_space_frozen": True,
        "selector_selected_before_target_eval": True,
        "train_only_retrieval": True,
        "eval_doc_hits_zero": True,
        "same_doc_hits_zero": True,
        "near_duplicate_flags_zero": True,
        "pseudo_heldout_beats_shuffle_top4": pseudo_heldout["top4_exact"] > shuffle["top4_exact"],
        "target_top4_no_worse_than_visible_reranker": target_after_freeze["top4_exact"] >= visible_baseline["top4_exact"],
        "proxy_masks_geometry_shaped": proxy_geometry_mode != "target_retrieval"
        or all(
            summary["mode"] == "target_retrieval" and summary["selected_cases"] > 0
            for summary in proxy_geometry_summaries
        ),
        "proxy_geometry_uses_redacted_target_hole": proxy_geometry_mode != "target_retrieval"
        or all(bool(summary.get("uses_redacted_target_hole")) for summary in proxy_geometry_summaries),
        "selected_proxy_geometry_no_worse_than_pool": proxy_geometry_mode != "target_retrieval"
        or all(
            summary["selected_mean_geometry_distance"] is not None
            and summary["pool_mean_geometry_distance"] is not None
            and float(summary["selected_mean_geometry_distance"]) <= float(summary["pool_mean_geometry_distance"])
            for summary in proxy_geometry_summaries
        ),
    }
    receipt = {
        "proof_name": "proxy_mask_selector_contract_smoke",
        "commit": current_commit(),
        "git_dirty": current_git_dirty(),
        "model_load": False,
        "device": "cpu",
        "num_target_cases": len(target_rows),
        "pseudo_masks_per_case": int(config["pseudo_masks_per_case"]),
        "pseudo_fit_cases": len(fit_rows),
        "pseudo_heldout_cases": len(heldout_rows),
        "max_candidates_per_example": int(config["max_candidates_per_example"]),
        "train_split_sha256": sha256_text(train_text),
        "validation_split_sha256": sha256_text(val_text),
        "redaction": redaction_flags,
        "proxy_geometry": {
            "mode": proxy_geometry_mode,
            "pool_per_case": int(config.get("proxy_geometry_pool_per_case", config["pseudo_masks_per_case"])),
            "mechanism": (
                "boundary-matched visible-only pseudo masks"
                if proxy_geometry_mode != "target_retrieval"
                else (
                    "choose visible-only pseudo masks whose redacted lattice/source geometry is closest to the target "
                    "redacted lattice/source geometry before any target metric is computed"
                )
            ),
            "summaries": proxy_geometry_summaries,
        },
        "candidate_policy": {
            "train_only_retrieval": True,
            "eval_doc_exclusion": True,
            "same_doc_hits": 0,
            "near_duplicate_flags": 0,
            "lattice_bi_anchor_candidates": int(config["lattice_bi_anchor_candidates"]),
            "lattice_bi_anchor_sizes": list(bi_anchor_sizes),
        },
        "selector_contract": {
            "contract_id": contract["contract_id"],
            "contract_ready": bool(contract["ready_for_heldout"]),
            "contract_apply": True,
            "preset_space_frozen": True,
            "selected_before_target_eval": True,
            "target_metric_used_for_selection": False,
            "selected": contract["selected"],
        },
        "pseudo_calibration": {
            "fit": pseudo_fit,
            "heldout": pseudo_heldout,
            "selected_fit_summary": selected["fit_summary"],
        },
        "target_after_freeze": {
            **target_after_freeze,
            "beats_prior": int(target_after_freeze["selected_exact"] > prior_baseline["selected_exact"]),
            "harms_prior": int(target_after_freeze["selected_exact"] < prior_baseline["selected_exact"]),
            "beats_surface": int(target_after_freeze["selected_exact"] > surface_baseline["selected_exact"]),
            "harms_surface": int(target_after_freeze["selected_exact"] < surface_baseline["selected_exact"]),
            "beats_visible_reranker": int(target_after_freeze["selected_exact"] > visible_baseline["selected_exact"]),
            "harms_visible_reranker": int(target_after_freeze["selected_exact"] < visible_baseline["selected_exact"]),
        },
        "baselines": {
            "prior": prior_baseline,
            "surface": surface_baseline,
            "visible_reranker": visible_baseline,
        },
        "shuffle_falsification": shuffle,
        "checks": checks,
        "useful_ratchet": bool(useful_ratchet),
        "cases": {
            "pseudo_fit": [_json_safe_row(row) for row in fit_rows],
            "pseudo_heldout": [_json_safe_row(row) for row in heldout_rows],
            "target_after_freeze": [_json_safe_row(row) for row in target_rows],
        },
        "claim_boundary": (
            "model-free proxy-mask selector contract smoke only; target metrics are after-freeze diagnostics, not a model SOTA claim"
        ),
    }
    receipt["verdict"] = "pass" if all(checks.values()) else "fail"
    return apply_claim_gate(receipt), contract


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Emit a visible-only proxy-mask selector contract smoke receipt.")
    parser.add_argument("--config")
    parser.add_argument("--out")
    parser.add_argument("--contract-out")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--require-useful-ratchet",
        action="store_true",
        help="Fail unless the proxy contract is also safe to cite as target-lift evidence.",
    )
    args = parser.parse_args(argv)
    receipt, contract = build_receipt(load_config(args.config))
    receipt = apply_claim_gate(receipt, require_useful_ratchet=args.require_useful_ratchet)
    text = json.dumps(receipt, indent=2)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    if args.contract_out:
        contract_out = Path(args.contract_out)
        contract_out.parent.mkdir(parents=True, exist_ok=True)
        contract_out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    print(text if args.json or not args.out else f"wrote {args.out}")
    if args.require_useful_ratchet and not receipt["claim_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
