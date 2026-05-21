from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPAIR_PROOF_CONTRACT_BOUNDARY = "repair_lattice_claim_requires_predeclared_heldout_proof_contract"


def _masked_loss(report: dict[str, Any]) -> float:
    return float(report["masked_eval"]["loss"])


def _masked_accuracy(report: dict[str, Any]) -> float:
    return float(report["masked_eval"]["masked_accuracy"])


def _byte_accuracy(report: dict[str, Any], variant: str) -> float:
    return float(report["infill"][variant]["summary"]["byte_accuracy"])


def _exact_match_rate(report: dict[str, Any], variant: str) -> float:
    return float(report["infill"][variant]["summary"].get("exact_match_rate", 0.0))


def _case_count(report: dict[str, Any], variant: str) -> int:
    return int(report["infill"][variant]["summary"].get("cases", 0))


def _has_variant(report: dict[str, Any], variant: str) -> bool:
    return isinstance(report.get("infill", {}).get(variant), dict)


def _variant_summary(report: dict[str, Any], variant: str) -> dict[str, Any]:
    summary = report.get("infill", {}).get(variant, {}).get("summary", {})
    return summary if isinstance(summary, dict) else {}


def _variant_cases(report: dict[str, Any], variant: str) -> list[dict[str, Any]]:
    cases = report.get("infill", {}).get(variant, {}).get("cases", [])
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def _frozen_context_ok(report: dict[str, Any], variant: str) -> bool:
    return bool(report["infill"][variant]["summary"].get("frozen_context_ok", False))


def _nonempty_dict(value: Any) -> bool:
    return isinstance(value, dict) and bool(value)


def _any_case_has(cases: list[dict[str, Any]], field: str) -> bool:
    return any(bool(case.get(field)) for case in cases)


def evaluate_repair_proof_contract(report: dict[str, Any]) -> dict[str, Any]:
    """Check whether a repair-lattice metric run carries enough proof to be claimable."""

    case_filter = report.get("case_filter", {})
    case_filter = case_filter if isinstance(case_filter, dict) else {}
    lattice_summary = _variant_summary(report, "retrieval_lattice")
    lattice_cases = _case_count(report, "retrieval_lattice") if _has_variant(report, "retrieval_lattice") else 0
    lattice_rows = _variant_cases(report, "retrieval_lattice")
    summary_case_count = int(lattice_summary.get("cases", 0)) if lattice_summary.get("cases") is not None else 0
    local_surface_cases = int(lattice_summary.get("local_surface_anchor_calibration_cases", 0) or 0)

    checks = {
        "checkpoint_sha256_recorded": bool(report.get("checkpoint_sha256")),
        "split_hashes_recorded": bool(report.get("train_split_sha256"))
        and bool(report.get("validation_split_sha256")),
        "unseen_holes_required": case_filter.get("require_unseen_hole") is True,
        "retrieval_lattice_available": _has_variant(report, "retrieval_lattice"),
        "nearest_visible_available": _has_variant(report, "nearest_visible_baseline"),
        "case_rows_match_summary": bool(lattice_rows)
        and len(lattice_rows) == lattice_cases
        and summary_case_count == lattice_cases,
        "oracle_scored_set_reported": "oracle_candidate_exact_in_scored_set_rate" in lattice_summary,
        "surface_verifier_diagnostics_reported": all(
            key in lattice_summary
            for key in (
                "surface_verifier_selected_exact_rate",
                "surface_verifier_top4_exact_rate",
                "surface_verifier_harm_count",
                "surface_verifier_help_count",
            )
        ),
        "selector_margin_sweep_reported": _nonempty_dict(lattice_summary.get("selector_margin_sweep"))
        or _any_case_has(lattice_rows, "selector_margin_sweep"),
        "selector_anchor_margin_sweep_reported": _nonempty_dict(
            lattice_summary.get("selector_anchor_margin_sweep")
        )
        or _any_case_has(lattice_rows, "selector_anchor_margin_sweep"),
        "local_surface_anchor_calibration_reported": lattice_cases > 0 and local_surface_cases >= lattice_cases,
        "local_surface_anchor_margin_sweep_reported": _nonempty_dict(
            lattice_summary.get("local_surface_anchor_margin_sweep")
        )
        or _any_case_has(lattice_rows, "local_surface_anchor_margin_sweep"),
    }
    missing = [name for name, passed in checks.items() if not passed]
    passed = not missing
    return {
        "passed": passed,
        "claim_boundary": (
            "predeclared_repair_lattice_claim_contract_met" if passed else REPAIR_PROOF_CONTRACT_BOUNDARY
        ),
        "missing": missing,
        "checks": checks,
        "current": {
            "checkpoint": report.get("checkpoint"),
            "cases": lattice_cases,
            "case_rows": len(lattice_rows),
            "configured_selector_margin": case_filter.get("lattice_selector_margin"),
            "configured_selector_anchor": case_filter.get("lattice_selector_anchor"),
            "configured_selector_anchor_sweep": case_filter.get("lattice_selector_anchor_sweep"),
            "configured_selector_margin_sweep": case_filter.get("lattice_selector_margin_sweep"),
            "configured_local_surface_anchor_calibration": case_filter.get(
                "lattice_local_surface_anchor_calibration"
            ),
        },
    }


def evaluate_repair_lattice_gate(
    report: dict[str, Any],
    *,
    min_cases: int = 4,
    min_lift: float = 0.0,
    require_proof_contract: bool = False,
) -> dict[str, Any]:
    proof_contract = evaluate_repair_proof_contract(report)
    if not _has_variant(report, "retrieval_lattice"):
        return {
            "passed": False,
            "claim_boundary": "no_repair_lattice_claim",
            "reason": "current report has no retrieval_lattice benchmark variant",
            "checks": {"retrieval_lattice_available": False},
            "thresholds": {"min_cases": min_cases, "min_lift": min_lift},
            "proof_contract_required": bool(require_proof_contract),
            "repair_proof_contract": proof_contract,
        }
    if not _has_variant(report, "nearest_visible_baseline"):
        return {
            "passed": False,
            "claim_boundary": "mechanism_only_claim_required_do_not_call_model_sota",
            "reason": "retrieval_lattice exists, but nearest_visible_baseline is missing",
            "checks": {"retrieval_lattice_available": True, "nearest_visible_available": False},
            "thresholds": {"min_cases": min_cases, "min_lift": min_lift},
            "proof_contract_required": bool(require_proof_contract),
            "repair_proof_contract": proof_contract,
        }

    lattice = _byte_accuracy(report, "retrieval_lattice")
    bridge = _byte_accuracy(report, "bridge_only_baseline")
    nearest = _byte_accuracy(report, "nearest_visible_baseline")
    lattice_exact = _exact_match_rate(report, "retrieval_lattice")
    nearest_exact = _exact_match_rate(report, "nearest_visible_baseline")
    cases = _case_count(report, "retrieval_lattice")
    checks = {
        "retrieval_lattice_available": True,
        "nearest_visible_available": True,
        "case_count_met": cases >= min_cases,
        "retrieval_lattice_beats_bridge_only": (lattice - bridge) > min_lift,
        "retrieval_lattice_beats_nearest_visible": (lattice - nearest) > min_lift,
        "retrieval_lattice_exact_beats_nearest_visible": (lattice_exact - nearest_exact) > min_lift,
        "frozen_context_preserved": _frozen_context_ok(report, "retrieval_lattice"),
    }
    if require_proof_contract:
        checks["repair_proof_contract_met"] = bool(proof_contract["passed"])
    passed = all(checks.values())
    return {
        "passed": passed,
        "claim_boundary": (
            "narrow_repair_lattice_claim_allowed_not_model_sota"
            if passed
            else "mechanism_only_claim_required_do_not_call_model_sota"
        ),
        "current": {
            "checkpoint": report.get("checkpoint"),
            "cases": cases,
            "bridge_only_byte_accuracy": bridge,
            "nearest_visible_byte_accuracy": nearest,
            "retrieval_lattice_byte_accuracy": lattice,
            "nearest_visible_exact_match_rate": nearest_exact,
            "retrieval_lattice_exact_match_rate": lattice_exact,
        },
        "deltas": {
            "retrieval_lattice_minus_bridge_only_byte_accuracy": lattice - bridge,
            "retrieval_lattice_minus_nearest_visible_byte_accuracy": lattice - nearest,
            "retrieval_lattice_minus_nearest_visible_exact_match_rate": lattice_exact - nearest_exact,
        },
        "thresholds": {"min_cases": min_cases, "min_lift": min_lift},
        "checks": checks,
        "proof_contract_required": bool(require_proof_contract),
        "repair_proof_contract": proof_contract,
    }


def evaluate_gate(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    min_masked_accuracy_gain: float = 0.015,
    min_infill_lift: float = 0.0,
    min_repair_cases: int = 4,
    min_repair_lift: float = 0.0,
    require_repair_proof_contract: bool = False,
) -> dict[str, Any]:
    baseline_acc = _masked_accuracy(baseline)
    current_acc = _masked_accuracy(current)
    bridge_only = _byte_accuracy(current, "bridge_only_baseline")
    unguided = _byte_accuracy(current, "unguided")
    guided = _byte_accuracy(current, "bridge_guided")
    masked_accuracy_gain = current_acc - baseline_acc
    masked_loss_delta = _masked_loss(current) - _masked_loss(baseline)
    checks = {
        "masked_ce_improved": masked_loss_delta < 0.0,
        "masked_accuracy_gain_met": masked_accuracy_gain >= min_masked_accuracy_gain,
        "model_only_beats_bridge_only": (unguided - bridge_only) > min_infill_lift,
        "bridge_guided_beats_bridge_only": (guided - bridge_only) > min_infill_lift,
        "frozen_context_preserved": _frozen_context_ok(current, "unguided")
        and _frozen_context_ok(current, "bridge_guided"),
    }
    model_passed = all(checks.values())
    repair_lattice = evaluate_repair_lattice_gate(
        current,
        min_cases=min_repair_cases,
        min_lift=min_repair_lift,
        require_proof_contract=require_repair_proof_contract,
    )
    passed = model_passed
    claim_boundary = "mechanism_only_claim_required_do_not_call_model_sota"
    if model_passed:
        claim_boundary = "mac_local_checkpoint_claim_allowed"
    elif repair_lattice["passed"]:
        claim_boundary = "narrow_repair_lattice_claim_allowed_not_model_sota"
    return {
        "passed": passed,
        "model_quality_passed": model_passed,
        "repair_lattice_passed": repair_lattice["passed"],
        "claim_boundary": claim_boundary,
        "baseline": {
            "checkpoint": baseline.get("checkpoint"),
            "masked_loss": _masked_loss(baseline),
            "masked_accuracy": baseline_acc,
        },
        "current": {
            "checkpoint": current.get("checkpoint"),
            "masked_loss": _masked_loss(current),
            "masked_accuracy": current_acc,
            "quality_label": current.get("quality_label"),
            "bridge_only_byte_accuracy": bridge_only,
            "unguided_byte_accuracy": unguided,
            "bridge_guided_byte_accuracy": guided,
        },
        "deltas": {
            "masked_loss_delta": masked_loss_delta,
            "masked_accuracy_gain": masked_accuracy_gain,
            "unguided_minus_bridge_only_byte_accuracy": unguided - bridge_only,
            "bridge_guided_minus_bridge_only_byte_accuracy": guided - bridge_only,
        },
        "thresholds": {
            "min_masked_accuracy_gain": min_masked_accuracy_gain,
            "min_infill_lift": min_infill_lift,
            "min_repair_cases": min_repair_cases,
            "min_repair_lift": min_repair_lift,
            "require_repair_proof_contract": bool(require_repair_proof_contract),
        },
        "checks": checks,
        "repair_lattice_gate": repair_lattice,
    }


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate HelixDiff model-quality claims from benchmark JSON.")
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--min-masked-accuracy-gain", type=float, default=0.015)
    parser.add_argument("--min-infill-lift", type=float, default=0.0)
    parser.add_argument("--min-repair-cases", type=int, default=4)
    parser.add_argument("--min-repair-lift", type=float, default=0.0)
    parser.add_argument(
        "--require-repair-proof-contract",
        action="store_true",
        help=(
            "Require the retrieval-lattice run to include the strict proof contract: hashes, unseen holes, "
            "nearest-visible baseline, selector sweeps, and local surface-anchor calibration diagnostics."
        ),
    )
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = evaluate_gate(
        current=load_report(args.current),
        baseline=load_report(args.baseline),
        min_masked_accuracy_gain=args.min_masked_accuracy_gain,
        min_infill_lift=args.min_infill_lift,
        min_repair_cases=args.min_repair_cases,
        min_repair_lift=args.min_repair_lift,
        require_repair_proof_contract=args.require_repair_proof_contract,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
