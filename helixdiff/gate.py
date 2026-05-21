from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _masked_loss(report: dict[str, Any]) -> float:
    return float(report["masked_eval"]["loss"])


def _masked_accuracy(report: dict[str, Any]) -> float:
    return float(report["masked_eval"]["masked_accuracy"])


def _byte_accuracy(report: dict[str, Any], variant: str) -> float:
    return float(report["infill"][variant]["summary"]["byte_accuracy"])


def _frozen_context_ok(report: dict[str, Any], variant: str) -> bool:
    return bool(report["infill"][variant]["summary"].get("frozen_context_ok", False))


def evaluate_gate(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    min_masked_accuracy_gain: float = 0.015,
    min_infill_lift: float = 0.0,
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
    passed = all(checks.values())
    return {
        "passed": passed,
        "claim_boundary": (
            "mac_local_checkpoint_claim_allowed"
            if passed
            else "mechanism_only_claim_required_do_not_call_model_sota"
        ),
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
        },
        "checks": checks,
    }


def load_report(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate HelixDiff model-quality claims from benchmark JSON.")
    parser.add_argument("--current", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--min-masked-accuracy-gain", type=float, default=0.015)
    parser.add_argument("--min-infill-lift", type=float, default=0.0)
    parser.add_argument("--json-out")
    args = parser.parse_args(argv)
    report = evaluate_gate(
        current=load_report(args.current),
        baseline=load_report(args.baseline),
        min_masked_accuracy_gain=args.min_masked_accuracy_gain,
        min_infill_lift=args.min_infill_lift,
    )
    print(json.dumps(report, indent=2))
    if args.json_out:
        out = Path(args.json_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
