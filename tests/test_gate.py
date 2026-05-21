from __future__ import annotations

import unittest

from helixdiff.gate import evaluate_gate, evaluate_repair_lattice_gate, evaluate_repair_proof_contract


def _variant(byte_accuracy: float, exact: float = 0.0) -> dict:
    return {"summary": {"byte_accuracy": byte_accuracy, "exact_match_rate": exact, "frozen_context_ok": True}}


def _strict_lattice_variant(byte_accuracy: float, exact: float, cases: int = 4) -> dict:
    case_rows = []
    for _ in range(cases):
        case_rows.append(
            {
                "selector_margin_sweep": [{"selector_margin": 0.0, "exact": True, "byte_accuracy": 1.0}],
                "selector_anchor_margin_sweep": [
                    {"selector_anchor": "prior", "selector_margin": 0.0, "exact": True, "byte_accuracy": 1.0},
                    {
                        "selector_anchor": "visible_reranker",
                        "selector_margin": 0.0,
                        "exact": True,
                        "byte_accuracy": 1.0,
                    },
                ],
                "local_surface_anchor_calibration": {"selected_selector_anchor": "prior", "applied": False},
                "visible_reranker_calibration": {
                    "status": "selected_visible_context_reranker_weights",
                    "applied": False,
                    "selected_summary": {
                        "visible_reranker_top1_exact_rate": 1.0,
                        "visible_reranker_top4_exact_rate": 1.0,
                    },
                },
            }
        )
    return {
        "summary": {
            "byte_accuracy": byte_accuracy,
            "exact_match_rate": exact,
            "frozen_context_ok": True,
            "cases": cases,
            "oracle_candidate_exact_in_scored_set_rate": 1.0,
            "surface_verifier_selected_exact_rate": 1.0,
            "surface_verifier_top4_exact_rate": 1.0,
            "surface_verifier_harm_count": 0,
            "surface_verifier_help_count": 0,
            "visible_reranker_selected_exact_rate": 1.0,
            "visible_reranker_top4_exact_rate": 1.0,
            "visible_reranker_harm_count": 0,
            "visible_reranker_help_count": 0,
            "visible_reranker_calibration_cases": cases,
            "bi_anchor_oracle_exact_rate": 1.0,
            "selector_margin_sweep": {"0": {"cases": cases}},
            "selector_anchor_margin_sweep": {
                "prior": {"0": {"cases": cases}},
                "visible_reranker": {"0": {"cases": cases}},
            },
            "local_surface_anchor_calibration_cases": cases,
            "local_surface_anchor_margin_sweep": {"0": {"cases": cases}},
        },
        "cases": case_rows,
    }


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
        "checkpoint_sha256": "abc",
        "train_split_sha256": "train",
        "validation_split_sha256": "validation",
        "case_filter": {"require_unseen_hole": True},
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

    def test_repair_proof_contract_reports_missing_diagnostics(self) -> None:
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

        contract = evaluate_repair_proof_contract(current)

        self.assertFalse(contract["passed"])
        self.assertIn("selector_anchor_margin_sweep_reported", contract["missing"])
        self.assertIn("local_surface_anchor_calibration_reported", contract["missing"])

    def test_required_repair_proof_contract_blocks_metric_only_lattice_win(self) -> None:
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

        report = evaluate_repair_lattice_gate(current, min_cases=4, require_proof_contract=True)

        self.assertFalse(report["passed"])
        self.assertTrue(report["checks"]["retrieval_lattice_beats_nearest_visible"])
        self.assertFalse(report["checks"]["repair_proof_contract_met"])
        self.assertEqual(
            report["repair_proof_contract"]["claim_boundary"],
            "repair_lattice_claim_requires_predeclared_heldout_proof_contract",
        )

    def test_required_repair_proof_contract_allows_predeclared_lattice_win(self) -> None:
        current = _report(
            loss=3.2,
            acc=0.14,
            bridge=0.1,
            unguided=0.1,
            guided=0.1,
            nearest=0.4,
            lattice=None,
            nearest_exact=0.25,
            lattice_exact=0.5,
        )
        current["infill"]["retrieval_lattice"] = _strict_lattice_variant(0.6, 0.5, cases=8)
        current["case_filter"].update(
            {
                "lattice_prior_rerank_top_k": 4,
                "lattice_verifier_mode": "dual",
                "lattice_verifier_top_k": 0,
                "lattice_selector_margin": 3.0,
                "lattice_selector_anchor": "surface",
                "lattice_selector_anchor_sweep": ["prior", "surface", "visible_reranker"],
                "lattice_selector_margin_sweep": [0, 1, 2, 3, 5],
                "lattice_bi_anchor_candidates": 64,
                "lattice_bi_anchor_sizes": [32, 24, 16, 12, 8, 6, 4],
                "lattice_local_surface_anchor_calibration": True,
                "lattice_apply_local_surface_anchor_calibration": False,
                "lattice_visible_reranker_calibration": True,
                "lattice_apply_visible_reranker_calibration": False,
            }
        )

        report = evaluate_repair_lattice_gate(current, min_cases=4, require_proof_contract=True)

        self.assertTrue(report["passed"])
        self.assertTrue(report["repair_proof_contract"]["passed"])
        self.assertEqual(report["claim_boundary"], "narrow_repair_lattice_claim_allowed_not_model_sota")

    def test_required_repair_proof_contract_blocks_non_predeclared_recipe(self) -> None:
        current = _report(
            loss=3.2,
            acc=0.14,
            bridge=0.1,
            unguided=0.1,
            guided=0.1,
            nearest=0.4,
            lattice=None,
            nearest_exact=0.25,
            lattice_exact=0.5,
        )
        current["infill"]["retrieval_lattice"] = _strict_lattice_variant(0.6, 0.5, cases=8)
        current["case_filter"].update(
            {
                "lattice_prior_rerank_top_k": 8,
                "lattice_verifier_mode": "dual",
                "lattice_verifier_top_k": 0,
                "lattice_selector_margin": 3.0,
                "lattice_selector_anchor": "surface",
                "lattice_selector_anchor_sweep": ["prior", "surface", "visible_reranker"],
                "lattice_selector_margin_sweep": [0.0, 1.0, 2.0, 3.0, 5.0],
                "lattice_bi_anchor_candidates": 64,
                "lattice_bi_anchor_sizes": [32, 24, 16, 12, 8, 6, 4],
                "lattice_local_surface_anchor_calibration": True,
                "lattice_apply_local_surface_anchor_calibration": False,
                "lattice_visible_reranker_calibration": True,
                "lattice_apply_visible_reranker_calibration": False,
            }
        )

        report = evaluate_repair_lattice_gate(current, min_cases=4, require_proof_contract=True)

        self.assertFalse(report["passed"])
        contract = report["repair_proof_contract"]
        self.assertFalse(contract["checks"]["predeclared_strict_recipe_matched"])
        self.assertIn("prior_rerank_top_k_is_4", contract["strict_repair_recipe"]["missing"])

    def test_repair_proof_contract_blocks_diagnostic_selector_contract(self) -> None:
        current = _report(
            loss=3.2,
            acc=0.14,
            bridge=0.1,
            unguided=0.1,
            guided=0.1,
            nearest=0.4,
            lattice=None,
            nearest_exact=0.25,
            lattice_exact=0.5,
        )
        current["infill"]["retrieval_lattice"] = _strict_lattice_variant(0.6, 0.5, cases=8)
        current["case_filter"].update(
            {
                "lattice_prior_rerank_top_k": 4,
                "lattice_verifier_mode": "dual",
                "lattice_verifier_top_k": 0,
                "lattice_selector_margin": 3.0,
                "lattice_selector_anchor": "surface",
                "lattice_selector_anchor_sweep": ["prior", "surface", "visible_reranker"],
                "lattice_selector_margin_sweep": [0.0, 1.0, 2.0, 3.0, 5.0],
                "lattice_bi_anchor_candidates": 64,
                "lattice_bi_anchor_sizes": [32, 24, 16, 12, 8, 6, 4],
                "lattice_local_surface_anchor_calibration": True,
                "lattice_apply_local_surface_anchor_calibration": False,
                "lattice_visible_reranker_calibration": True,
                "lattice_apply_visible_reranker_calibration": False,
                "lattice_selector_contract": {
                    "contract_id": "abc123",
                    "sha256": "def456",
                    "status": "diagnostic_only",
                    "ready_for_heldout": False,
                },
            }
        )

        report = evaluate_repair_lattice_gate(current, min_cases=4, require_proof_contract=True)

        self.assertFalse(report["passed"])
        self.assertFalse(report["repair_proof_contract"]["checks"]["selector_contract_ready_if_present"])
        self.assertIn("selector_contract_ready_if_present", report["repair_proof_contract"]["missing"])


if __name__ == "__main__":
    unittest.main()
