from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from helixdiff.contract import build_selector_contract, load_calibration_objects


def _calibration(*, cases: int = 4, status: str = "candidate_anchor_margin") -> dict:
    anchor_recommendation = {
        "status": status,
        "recommended_selector_anchor": "surface" if status == "candidate_anchor_margin" else None,
        "recommended_margin": 3.0 if status == "candidate_anchor_margin" else None,
        "diagnostic_best_selector_anchor": "surface",
        "diagnostic_best_margin": 3.0,
        "exact_lift_vs_prior_margin_0": 0.5,
        "byte_accuracy_lift_vs_prior_margin_0": 0.25,
        "selection_rule": "test rule",
    }
    return {
        "kind": "helixdiff_selector_margin_calibration",
        "cases": cases,
        "reports": [
            {
                "source_path": "proof/bench.json",
                "checkpoint": "checkpoint.pt",
                "checkpoint_sha256": "abc",
                "cases": cases,
                "configured_selector_anchor": "prior",
                "configured_selector_anchor_sweep": ["prior", "surface"],
                "configured_selector_margin": 0.0,
            }
        ],
        "recommendation": {
            "status": "candidate_margin" if status == "candidate_anchor_margin" else status,
            "recommended_margin": 3.0 if status == "candidate_anchor_margin" else None,
            "diagnostic_best_margin": 3.0,
            "exact_lift_vs_margin_0": 0.5,
            "byte_accuracy_lift_vs_margin_0": 0.25,
        },
        "anchor_recommendation": anchor_recommendation,
        "visible_hole_reranker": {
            "status": "candidate_visible_hole_reranker",
            "bottleneck": "visible_hole_reranker",
        },
    }


class SelectorContractTests(unittest.TestCase):
    def test_contract_freezes_ready_anchor_margin_recommendation(self) -> None:
        contract = build_selector_contract([_calibration()], min_cases=4)

        self.assertTrue(contract["ready_for_heldout"])
        self.assertEqual(contract["status"], "ready_for_heldout")
        self.assertEqual(contract["selected"]["selector_anchor"], "surface")
        self.assertEqual(contract["selected"]["selector_margin"], 3.0)
        self.assertEqual(
            contract["frozen_benchmark_flags"],
            ["--lattice-selector-anchor", "surface", "--lattice-selector-margin", "3"],
        )
        self.assertIn("separate held-out", contract["claim_boundary"])
        self.assertEqual(contract["calibrations"][0]["visible_hole_reranker_bottleneck"], "visible_hole_reranker")

    def test_contract_remains_diagnostic_when_case_count_is_too_small(self) -> None:
        contract = build_selector_contract([_calibration(cases=1)], min_cases=4)

        self.assertFalse(contract["ready_for_heldout"])
        self.assertEqual(contract["status"], "diagnostic_only")
        self.assertIn("min_cases", contract["missing"])
        self.assertEqual(contract["selected"]["selector_anchor"], "surface")

    def test_contract_can_load_json_and_records_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "calibration.json"
            path.write_text(json.dumps(_calibration()), encoding="utf-8")

            loaded = load_calibration_objects([path])
            contract = build_selector_contract(loaded, min_cases=4)

        self.assertTrue(contract["ready_for_heldout"])
        self.assertEqual(contract["calibrations"][0]["source_path"], str(path))
        self.assertEqual(len(contract["calibrations"][0]["source_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
