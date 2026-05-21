from __future__ import annotations

import unittest

from helixdiff.gate import evaluate_gate, evaluate_repair_lattice_gate


def _variant(byte_accuracy: float, exact: float = 0.0) -> dict:
    return {"summary": {"byte_accuracy": byte_accuracy, "exact_match_rate": exact, "frozen_context_ok": True}}


def _report(
    loss: float,
    acc: float,
    bridge: float,
    unguided: float,
    guided: float,
    *,
    nearest: float | None = None,
    lattice: float | None = None,
    nearest_exact: float = 0.0,
    lattice_exact: float = 0.0,
    lattice_cases: int = 4,
) -> dict:
    report = {
        "checkpoint": "checkpoint.pt",
        "quality_label": "mechanism_checkpoint",
        "masked_eval": {"loss": loss, "masked_accuracy": acc},
        "infill": {
            "bridge_only_baseline": _variant(bridge),
            "unguided": _variant(unguided),
            "bridge_guided": _variant(guided),
        },
    }
    if nearest is not None:
        report["infill"]["nearest_visible_baseline"] = _variant(nearest, nearest_exact)
    if lattice is not None:
        report["infill"]["retrieval_lattice"] = _variant(lattice, lattice_exact)
        report["infill"]["retrieval_lattice"]["summary"]["cases"] = lattice_cases
    return report


class GateTests(unittest.TestCase):
    def test_gate_passes_only_when_model_and_guided_lift_clear_baseline(self) -> None:
        baseline = _report(loss=3.2, acc=0.14, bridge=0.1, unguided=0.1, guided=0.1)
        current = _report(loss=3.0, acc=0.16, bridge=0.2, unguided=0.25, guided=0.3)

        report = evaluate_gate(current=current, baseline=baseline)

        self.assertTrue(report["passed"])
        self.assertEqual(report["claim_boundary"], "mac_local_checkpoint_claim_allowed")

    def test_gate_fails_when_guidance_only_matches_bridge(self) -> None:
        baseline = _report(loss=3.2, acc=0.14, bridge=0.1, unguided=0.1, guided=0.1)
        current = _report(loss=3.0, acc=0.16, bridge=0.2, unguided=0.25, guided=0.2)

        report = evaluate_gate(current=current, baseline=baseline)

        self.assertFalse(report["passed"])
        self.assertEqual(report["claim_boundary"], "mechanism_only_claim_required_do_not_call_model_sota")

    def test_repair_lattice_gate_allows_narrow_claim_only_when_it_beats_nearest_visible(self) -> None:
        current = _report(
            loss=3.2,
            acc=0.14,
            bridge=0.1,
            unguided=0.1,
            guided=0.1,
            nearest=0.4,
            lattice=0.6,
            nearest_exact=0.25,
            lattice_exact=0.5,
            lattice_cases=8,
        )

        report = evaluate_repair_lattice_gate(current, min_cases=4)

        self.assertTrue(report["passed"])
        self.assertEqual(report["claim_boundary"], "narrow_repair_lattice_claim_allowed_not_model_sota")
        self.assertEqual(report["deltas"]["retrieval_lattice_minus_nearest_visible_byte_accuracy"], 0.19999999999999996)

    def test_main_gate_keeps_model_failure_separate_from_repair_lattice_win(self) -> None:
        baseline = _report(loss=3.2, acc=0.14, bridge=0.1, unguided=0.1, guided=0.1)
        current = _report(
            loss=3.1,
            acc=0.145,
            bridge=0.1,
            unguided=0.1,
            guided=0.1,
            nearest=0.4,
            lattice=0.6,
            nearest_exact=0.25,
            lattice_exact=0.5,
            lattice_cases=8,
        )

        report = evaluate_gate(current=current, baseline=baseline)

        self.assertFalse(report["passed"])
        self.assertFalse(report["model_quality_passed"])
        self.assertTrue(report["repair_lattice_passed"])
        self.assertEqual(report["claim_boundary"], "narrow_repair_lattice_claim_allowed_not_model_sota")

    def test_repair_lattice_gate_fails_when_it_only_matches_nearest_visible(self) -> None:
        current = _report(
            loss=3.2,
            acc=0.14,
            bridge=0.1,
            unguided=0.1,
            guided=0.1,
            nearest=0.5,
            lattice=0.5,
            nearest_exact=0.5,
            lattice_exact=0.5,
            lattice_cases=8,
        )

        report = evaluate_repair_lattice_gate(current, min_cases=4)

        self.assertFalse(report["passed"])
        self.assertFalse(report["checks"]["retrieval_lattice_beats_nearest_visible"])
        self.assertEqual(report["claim_boundary"], "mechanism_only_claim_required_do_not_call_model_sota")


if __name__ == "__main__":
    unittest.main()
