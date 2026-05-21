from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable


CONTRACT_BOUNDARY = (
    "selector contracts freeze calibration choices only; a claim still requires applying the contract "
    "to a separate held-out benchmark and passing the repair gate"
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_calibration_objects(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    raise ValueError("selector calibration must be a JSON object or a list of objects")


def load_calibration_objects(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    calibrations: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for calibration in iter_calibration_objects(payload):
            copied = dict(calibration)
            copied["_source_path"] = str(path)
            copied["_source_sha256"] = sha256_file(path)
            calibrations.append(copied)
    return calibrations


def load_selector_contract(path: str | Path) -> dict[str, Any]:
    contract = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("selector contract must be a JSON object")
    if contract.get("kind") != "helixdiff_selector_contract":
        raise ValueError("selector contract kind must be helixdiff_selector_contract")
    return contract


def selector_settings_from_contract(
    contract: dict[str, Any],
    *,
    require_ready: bool = False,
) -> dict[str, Any]:
    if require_ready and not contract.get("ready_for_heldout"):
        raise ValueError("selector contract is not ready_for_heldout")
    selected = contract.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("selector contract has no selected selector settings")
    anchor = selected.get("selector_anchor")
    if anchor not in {"prior", "surface", "visible_reranker"}:
        raise ValueError("selector contract selected.selector_anchor is invalid")
    try:
        margin = float(selected["selector_margin"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("selector contract selected.selector_margin is invalid") from exc
    return {
        "selector_anchor": str(anchor),
        "selector_margin": margin,
        "source": selected.get("source"),
        "contract_id": contract.get("contract_id"),
        "status": contract.get("status"),
        "ready_for_heldout": bool(contract.get("ready_for_heldout")),
    }


def _format_float(value: float) -> str:
    return f"{float(value):g}"


def _configured_anchor(calibration: dict[str, Any]) -> str:
    anchors = {
        str(report.get("configured_selector_anchor"))
        for report in calibration.get("reports", [])
        if isinstance(report, dict) and report.get("configured_selector_anchor") is not None
    }
    if len(anchors) == 1:
        return anchors.pop()
    return "prior"


def _candidate_from_anchor_recommendation(calibration: dict[str, Any]) -> dict[str, Any] | None:
    recommendation = calibration.get("anchor_recommendation", {})
    if not isinstance(recommendation, dict):
        return None
    anchor = recommendation.get("recommended_selector_anchor")
    margin = recommendation.get("recommended_margin")
    if recommendation.get("status") != "candidate_anchor_margin" or anchor is None or margin is None:
        return None
    return {
        "source": "anchor_recommendation",
        "selector_anchor": str(anchor),
        "selector_margin": float(margin),
        "recommendation_status": recommendation.get("status"),
        "exact_lift": recommendation.get("exact_lift_vs_prior_margin_0"),
        "byte_accuracy_lift": recommendation.get("byte_accuracy_lift_vs_prior_margin_0"),
        "selection_rule": recommendation.get("selection_rule"),
    }


def _candidate_from_margin_recommendation(calibration: dict[str, Any]) -> dict[str, Any] | None:
    recommendation = calibration.get("recommendation", {})
    if not isinstance(recommendation, dict):
        return None
    margin = recommendation.get("recommended_margin")
    if recommendation.get("status") != "candidate_margin" or margin is None:
        return None
    return {
        "source": "recommendation",
        "selector_anchor": _configured_anchor(calibration),
        "selector_margin": float(margin),
        "recommendation_status": recommendation.get("status"),
        "exact_lift": recommendation.get("exact_lift_vs_margin_0"),
        "byte_accuracy_lift": recommendation.get("byte_accuracy_lift_vs_margin_0"),
        "selection_rule": recommendation.get("selection_rule"),
    }


def _diagnostic_candidate(calibration: dict[str, Any]) -> dict[str, Any] | None:
    anchor_recommendation = calibration.get("anchor_recommendation", {})
    if isinstance(anchor_recommendation, dict):
        anchor = anchor_recommendation.get("diagnostic_best_selector_anchor")
        margin = anchor_recommendation.get("diagnostic_best_margin")
        if anchor is not None and margin is not None:
            return {
                "source": "diagnostic_anchor_recommendation",
                "selector_anchor": str(anchor),
                "selector_margin": float(margin),
                "recommendation_status": anchor_recommendation.get("status"),
                "selection_rule": anchor_recommendation.get("selection_rule"),
            }
    recommendation = calibration.get("recommendation", {})
    if isinstance(recommendation, dict) and recommendation.get("diagnostic_best_margin") is not None:
        return {
            "source": "diagnostic_recommendation",
            "selector_anchor": _configured_anchor(calibration),
            "selector_margin": float(recommendation["diagnostic_best_margin"]),
            "recommendation_status": recommendation.get("status"),
            "selection_rule": recommendation.get("selection_rule"),
        }
    return None


def _score_candidate(calibration: dict[str, Any], candidate: dict[str, Any]) -> tuple[int, float, float, float]:
    cases = int(calibration.get("cases", 0) or 0)
    exact_lift = candidate.get("exact_lift")
    byte_lift = candidate.get("byte_accuracy_lift")
    return (
        cases,
        float(exact_lift) if exact_lift is not None else 0.0,
        float(byte_lift) if byte_lift is not None else 0.0,
        -float(candidate["selector_margin"]),
    )


def _source_report_receipts(calibration: dict[str, Any]) -> list[dict[str, Any]]:
    receipts: list[dict[str, Any]] = []
    for report in calibration.get("reports", []):
        if not isinstance(report, dict):
            continue
        receipt = {
            "source_path": report.get("source_path"),
            "checkpoint": report.get("checkpoint"),
            "checkpoint_sha256": report.get("checkpoint_sha256"),
            "cases": report.get("cases"),
            "configured_selector_anchor": report.get("configured_selector_anchor"),
            "configured_selector_anchor_sweep": report.get("configured_selector_anchor_sweep"),
            "configured_selector_margin": report.get("configured_selector_margin"),
        }
        source_path = report.get("source_path")
        if isinstance(source_path, str) and Path(source_path).exists():
            receipt["source_sha256"] = sha256_file(source_path)
        receipts.append(receipt)
    return receipts


def build_selector_contract(
    calibrations: list[dict[str, Any]],
    *,
    min_cases: int = 4,
) -> dict[str, Any]:
    if not calibrations:
        raise ValueError("at least one selector calibration report is required")

    ready_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    diagnostic_candidates: list[tuple[dict[str, Any], dict[str, Any]]] = []
    calibration_summaries: list[dict[str, Any]] = []
    for calibration in calibrations:
        cases = int(calibration.get("cases", 0) or 0)
        anchor_candidate = _candidate_from_anchor_recommendation(calibration)
        margin_candidate = _candidate_from_margin_recommendation(calibration)
        if cases >= min_cases:
            for candidate in (anchor_candidate, margin_candidate):
                if candidate is not None:
                    ready_candidates.append((calibration, candidate))
        diagnostic = _diagnostic_candidate(calibration) or anchor_candidate or margin_candidate
        if diagnostic is not None:
            diagnostic_candidates.append((calibration, diagnostic))
        visible = calibration.get("visible_hole_reranker", {})
        calibration_summaries.append(
            {
                "source_path": calibration.get("_source_path"),
                "source_sha256": calibration.get("_source_sha256"),
                "cases": cases,
                "recommendation_status": calibration.get("recommendation", {}).get("status")
                if isinstance(calibration.get("recommendation"), dict)
                else None,
                "anchor_recommendation_status": calibration.get("anchor_recommendation", {}).get("status")
                if isinstance(calibration.get("anchor_recommendation"), dict)
                else None,
                "visible_hole_reranker_status": visible.get("status") if isinstance(visible, dict) else None,
                "visible_hole_reranker_bottleneck": visible.get("bottleneck") if isinstance(visible, dict) else None,
                "source_reports": _source_report_receipts(calibration),
            }
        )

    ready = bool(ready_candidates)
    pool = ready_candidates if ready else diagnostic_candidates
    if not pool:
        selected_calibration = max(calibrations, key=lambda item: int(item.get("cases", 0) or 0))
        selected_candidate: dict[str, Any] | None = None
    else:
        selected_calibration, selected_candidate = sorted(
            pool,
            key=lambda item: _score_candidate(item[0], item[1]),
            reverse=True,
        )[0]

    missing: list[str] = []
    if selected_candidate is None:
        missing.append("candidate_selector_recommendation")
    if int(selected_calibration.get("cases", 0) or 0) < min_cases:
        missing.append("min_cases")
    if not ready:
        missing.append("candidate_recommendation")

    selected = None
    frozen_flags: list[str] = []
    if selected_candidate is not None:
        selected = {
            "selector_anchor": selected_candidate["selector_anchor"],
            "selector_margin": selected_candidate["selector_margin"],
            "source": selected_candidate["source"],
            "recommendation_status": selected_candidate.get("recommendation_status"),
            "selection_rule": selected_candidate.get("selection_rule"),
        }
        frozen_flags = [
            "--lattice-selector-anchor",
            selected_candidate["selector_anchor"],
            "--lattice-selector-margin",
            _format_float(float(selected_candidate["selector_margin"])),
        ]

    contract_core = {
        "ready_for_heldout": ready,
        "selected": selected,
        "min_cases": min_cases,
        "calibrations": calibration_summaries,
    }
    contract_id = hashlib.sha256(json.dumps(contract_core, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return {
        "kind": "helixdiff_selector_contract",
        "version": 1,
        "contract_id": contract_id,
        "status": "ready_for_heldout" if ready else "diagnostic_only",
        "ready_for_heldout": ready,
        "claim_boundary": CONTRACT_BOUNDARY,
        "missing": missing,
        "min_cases": min_cases,
        "selected": selected,
        "frozen_benchmark_flags": frozen_flags,
        "frozen_shell_flags": " ".join(frozen_flags),
        "heldout_requirements": [
            "run on cases not used to build this selector contract",
            "keep --require-unseen-hole enabled",
            "report bridge-only and nearest-visible baselines",
            "run helixdiff-gate --require-repair-proof-contract before making a narrow repair claim",
        ],
        "calibrations": calibration_summaries,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Freeze HelixDiff selector calibration into a predeclared held-out benchmark contract."
    )
    parser.add_argument("calibrations", nargs="+", help="One or more helixdiff-calibrate-selector JSON reports.")
    parser.add_argument("--min-cases", type=int, default=4)
    parser.add_argument("--json-out")
    parser.add_argument("--require-ready", action="store_true")
    args = parser.parse_args(argv)
    contract = build_selector_contract(
        load_calibration_objects(args.calibrations),
        min_cases=args.min_cases,
    )
    print(json.dumps(contract, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(contract, indent=2), encoding="utf-8")
    if args.require_ready and not contract["ready_for_heldout"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
