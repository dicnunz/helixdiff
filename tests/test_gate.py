from __future__ import annotations

import unittest

from helixdiff.gate import evaluate_gate


def _report(loss: float, acc: float, bridge: float, unguided: float, guided: float) -> dict:
    return {
        "checkpoint": "checkpoint.pt",
        "quality_label": "mechanism_checkpoint",
        "masked_eval": {"loss": loss, "masked_accuracy": acc},
        "infill": {
            "bridge_only_baseline": {"summary": {"byte_accuracy": bridge, "frozen_context_ok": True}},
            "unguided": {"summary": {"byte_accuracy": unguided, "frozen_context_ok": True}},
            "bridge_guided": {"summary": {"byte_accuracy": guided, "frozen_context_ok": True}},
        },
    }


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


if __name__ == "__main__":
    unittest.main()
