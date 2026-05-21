from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


CLAIM_BOUNDARY = (
    "selector calibration is diagnostic unless the margin is chosen on a calibration split "
    "and then predeclared for a separate held-out benchmark"
)


def format_margin(margin: float) -> str:
    return f"{float(margin):g}"


def _mean(values: Iterable[float]) -> float | None:
    values = list(values)
    if not values:
        return None
    return sum(values) / len(values)


def _rate(rows: list[dict[str, Any]], predicate: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row.get(predicate)) / len(rows)


def _effect_rate(rows: list[dict[str, Any]], effect: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row.get("selector_effect") == effect) / len(rows)


def _outcome_rate(rows: list[dict[str, Any]], outcome: str) -> float:
    if not rows:
        return 0.0
    return sum(1.0 for row in rows if row.get("outcome_category") == outcome) / len(rows)


def iter_report_objects(payload: Any) -> Iterable[dict[str, Any]]:
    if isinstance(payload, dict):
        yield payload
        return
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                yield item
        return
    raise ValueError("benchmark report must be a JSON object or a list of objects")


def load_report_objects(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for report in iter_report_objects(payload):
            copied = dict(report)
            copied["_source_path"] = str(path)
            reports.append(copied)
    return reports


def retrieval_lattice_cases(report: dict[str, Any]) -> list[dict[str, Any]]:
    lattice = report.get("infill", {}).get("retrieval_lattice", {})
    cases = lattice.get("cases", [])
    if not isinstance(cases, list):
        return []
    return [case for case in cases if isinstance(case, dict)]


def summarize_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for report in reports:
        lattice = report.get("infill", {}).get("retrieval_lattice", {})
        case_filter = report.get("case_filter", {})
        rows.append(
            {
                "source_path": report.get("_source_path"),
                "checkpoint": report.get("checkpoint"),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
                "cases": len(retrieval_lattice_cases(report)),
                "configured_selector_margin": case_filter.get("lattice_selector_margin"),
                "summary_exact_match_rate": lattice.get("summary", {}).get("exact_match_rate"),
                "summary_byte_accuracy": lattice.get("summary", {}).get("byte_accuracy"),
            }
        )
    return rows


def collect_selector_margin_sweeps(cases: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for case in cases:
        for row in case.get("selector_margin_sweep", []):
            if not isinstance(row, dict) or "selector_margin" not in row:
                continue
            key = format_margin(float(row["selector_margin"]))
            buckets.setdefault(key, []).append(row)
    return buckets


def summarize_selector_margins(cases: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    buckets = collect_selector_margin_sweeps(cases)
    summaries: dict[str, dict[str, Any]] = {}
    for key, rows in sorted(buckets.items(), key=lambda item: float(item[0])):
        outcome_categories = Counter(str(row.get("outcome_category", "unknown")) for row in rows)
        selector_effects = Counter(str(row.get("selector_effect", "unknown")) for row in rows)
        summaries[key] = {
            "margin": float(key),
            "cases": len(rows),
            "exact_match_rate": _rate(rows, "exact"),
            "byte_accuracy": _mean(float(row.get("byte_accuracy", 0.0)) for row in rows) or 0.0,
            "selector_margin_applied_rate": _rate(rows, "selector_margin_applied"),
            "margin_rescued_exact_anchor_rate": _effect_rate(rows, "margin_rescued_exact_anchor"),
            "margin_blocked_exact_raw_rate": _effect_rate(rows, "margin_blocked_exact_raw"),
            "raw_verifier_overrode_exact_anchor_rate": _outcome_rate(rows, "raw_verifier_overrode_exact_anchor"),
            "scored_exact_not_selected_rate": _outcome_rate(rows, "scored_exact_not_selected"),
            "outcome_categories": dict(sorted(outcome_categories.items())),
            "selector_effects": dict(sorted(selector_effects.items())),
        }
    return summaries


def _rank_margin(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        float(summary["exact_match_rate"]),
        float(summary["byte_accuracy"]),
        -float(summary["margin_blocked_exact_raw_rate"]),
        -float(summary["margin"]),
    )


def recommend_selector_margin(
    margin_summaries: dict[str, dict[str, Any]],
    *,
    min_cases: int = 4,
    allow_blocked_exact_raw: bool = False,
) -> dict[str, Any]:
    if not margin_summaries:
        return {
            "status": "no_selector_margin_sweep",
            "recommended_margin": None,
            "diagnostic_best_margin": None,
            "reason": "no per-case selector_margin_sweep rows were found",
            "claim_boundary": CLAIM_BOUNDARY,
        }

    all_candidates = sorted(margin_summaries.values(), key=_rank_margin, reverse=True)
    diagnostic_best = all_candidates[0]
    eligible = [row for row in all_candidates if int(row["cases"]) >= min_cases]
    if not eligible:
        return {
            "status": "diagnostic_only_insufficient_cases",
            "recommended_margin": None,
            "diagnostic_best_margin": diagnostic_best["margin"],
            "reason": f"best observed margin has only {diagnostic_best['cases']} cases; require at least {min_cases}",
            "selection_rule": (
                "maximize exact_match_rate, then byte_accuracy, then avoid blocking exact raw verifier hits, "
                "then choose the lower margin"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    safe_candidates = (
        eligible
        if allow_blocked_exact_raw
        else [row for row in eligible if float(row["margin_blocked_exact_raw_rate"]) == 0.0]
    )
    if not safe_candidates:
        return {
            "status": "blocked_exact_raw_risk",
            "recommended_margin": None,
            "diagnostic_best_margin": diagnostic_best["margin"],
            "reason": "every eligible margin blocked at least one raw verifier exact hit",
            "selection_rule": (
                "maximize exact_match_rate, then byte_accuracy, then avoid blocking exact raw verifier hits, "
                "then choose the lower margin"
            ),
            "claim_boundary": CLAIM_BOUNDARY,
        }

    best = sorted(safe_candidates, key=_rank_margin, reverse=True)[0]
    zero = margin_summaries.get("0")
    exact_lift_vs_zero = None
    byte_lift_vs_zero = None
    if zero is not None:
        exact_lift_vs_zero = float(best["exact_match_rate"]) - float(zero["exact_match_rate"])
        byte_lift_vs_zero = float(best["byte_accuracy"]) - float(zero["byte_accuracy"])
    return {
        "status": "candidate_margin",
        "recommended_margin": best["margin"],
        "diagnostic_best_margin": diagnostic_best["margin"],
        "reason": "selected the lowest safe margin on the best exact/byte-accuracy frontier",
        "exact_lift_vs_margin_0": exact_lift_vs_zero,
        "byte_accuracy_lift_vs_margin_0": byte_lift_vs_zero,
        "selection_rule": (
            "maximize exact_match_rate, then byte_accuracy, then avoid blocking exact raw verifier hits, "
            "then choose the lower margin"
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def calibrate_selector_margins(
    reports: list[dict[str, Any]],
    *,
    min_cases: int = 4,
    allow_blocked_exact_raw: bool = False,
) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    for report in reports:
        cases.extend(retrieval_lattice_cases(report))
    margin_summaries = summarize_selector_margins(cases)
    exact_anchor_gaps = [
        float(case["anchor_margin_gap"])
        for case in cases
        if case.get("anchor_exact") and case.get("anchor_margin_gap") is not None
    ]
    all_anchor_gaps = [
        float(case["anchor_margin_gap"]) for case in cases if case.get("anchor_margin_gap") is not None
    ]
    return {
        "kind": "helixdiff_selector_margin_calibration",
        "reports": summarize_reports(reports),
        "cases": len(cases),
        "margins": margin_summaries,
        "anchor_gap_diagnostics": {
            "avg_anchor_margin_gap": _mean(all_anchor_gaps),
            "avg_exact_anchor_margin_gap": _mean(exact_anchor_gaps),
            "max_exact_anchor_margin_gap": max(exact_anchor_gaps) if exact_anchor_gaps else None,
            "exact_anchor_cases": len(exact_anchor_gaps),
        },
        "recommendation": recommend_selector_margin(
            margin_summaries,
            min_cases=min_cases,
            allow_blocked_exact_raw=allow_blocked_exact_raw,
        ),
        "claim_boundary": CLAIM_BOUNDARY,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate HelixDiff retrieval-lattice selector margins from benchmark JSON."
    )
    parser.add_argument("reports", nargs="+", help="One or more helixdiff-bench JSON reports.")
    parser.add_argument("--min-cases", type=int, default=4)
    parser.add_argument("--allow-blocked-exact-raw", action="store_true")
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = calibrate_selector_margins(
        load_report_objects(args.reports),
        min_cases=args.min_cases,
        allow_blocked_exact_raw=args.allow_blocked_exact_raw,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
