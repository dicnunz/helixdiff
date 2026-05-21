from __future__ import annotations

import copy
import unittest

from helixdiff.contract import selector_settings_from_contract
from helixdiff.proofs.proxy_mask_selector_contract_smoke import apply_claim_gate, build_receipt, load_config


class ProxyMaskSelectorContractSmokeTests(unittest.TestCase):
    def test_receipt_builds_ready_visible_only_selector_contract(self) -> None:
        config = load_config(None)
        config.update(
            {
                "data": "data/tinyshakespeare.txt",
                "cases": 2,
                "span_chars": 4,
                "context_chars": 36,
                "pseudo_masks_per_case": 6,
                "max_candidates_per_example": 64,
                "shuffle_trials": 16,
            }
        )
        receipt, contract = build_receipt(config)

        self.assertEqual(receipt["verdict"], "pass")
        self.assertFalse(receipt["model_load"])
        self.assertTrue(receipt["redaction"]["target_span_redacted_before_calibration"])
        self.assertFalse(receipt["redaction"]["target_gold_available_to_contract_builder"])
        self.assertTrue(receipt["selector_contract"]["contract_ready"])
        self.assertTrue(receipt["selector_contract"]["contract_apply"])
        self.assertTrue(receipt["selector_contract"]["selected_before_target_eval"])
        self.assertFalse(receipt["selector_contract"]["target_metric_used_for_selection"])
        self.assertEqual(receipt["selector_contract"]["selected"]["source"], "proxy_mask_visible_context")
        self.assertGreater(receipt["pseudo_calibration"]["heldout"]["top4_exact"], receipt["shuffle_falsification"]["top4_exact"])
        self.assertTrue(receipt["checks"]["pseudo_heldout_beats_shuffle_top4"])
        self.assertFalse(receipt["claim_gate"]["require_useful_ratchet"])
        self.assertFalse(receipt["claim_gate"]["public_target_lift_claim_allowed"])
        self.assertIn("Contract readiness only", receipt["claim_gate"]["claim_boundary"])
        self.assertTrue(contract["ready_for_heldout"])
        settings = selector_settings_from_contract(contract, require_ready=True)
        self.assertIn(settings["selector_anchor"], {"prior", "surface", "visible_reranker"})
        self.assertEqual(settings["selector_margin"], 3.0)

    def test_useful_ratchet_gate_blocks_public_target_lift_claims(self) -> None:
        config = load_config(None)
        config.update(
            {
                "data": "data/tinyshakespeare.txt",
                "cases": 2,
                "span_chars": 4,
                "context_chars": 36,
                "pseudo_masks_per_case": 6,
                "max_candidates_per_example": 64,
                "shuffle_trials": 16,
            }
        )
        receipt, _ = build_receipt(config)

        gated = apply_claim_gate(copy.deepcopy(receipt), require_useful_ratchet=True)

        self.assertEqual(gated["verdict"], "fail")
        self.assertFalse(gated["claim_gate"]["passed"])
        self.assertFalse(gated["claim_gate"]["public_target_lift_claim_allowed"])
        self.assertIn("useful_ratchet_required", gated["failure_reasons"])
        self.assertIn("useful_ratchet=false", gated["claim_gate"]["blocker"])

    def test_target_retrieval_geometry_proxy_mode_fails_closed_without_proxy_heldout_lift(self) -> None:
        config = load_config(None)
        config.update(
            {
                "data": "data/tinyshakespeare.txt",
                "cases": 2,
                "span_chars": 4,
                "context_chars": 36,
                "pseudo_masks_per_case": 6,
                "max_candidates_per_example": 64,
                "shuffle_trials": 16,
                "proxy_geometry_mode": "target_retrieval",
                "proxy_geometry_pool_per_case": 18,
            }
        )
        receipt, contract = build_receipt(config)

        self.assertEqual(receipt["proxy_geometry"]["mode"], "target_retrieval")
        self.assertTrue(receipt["checks"]["proxy_masks_geometry_shaped"])
        self.assertTrue(receipt["checks"]["proxy_geometry_uses_redacted_target_hole"])
        self.assertTrue(receipt["checks"]["selected_proxy_geometry_no_worse_than_pool"])
        self.assertFalse(receipt["checks"]["pseudo_heldout_beats_shuffle_top4"])
        self.assertEqual(receipt["verdict"], "fail")
        self.assertFalse(receipt["claim_gate"]["public_target_lift_claim_allowed"])
        self.assertEqual(contract["selected"]["source"], "proxy_mask_target_retrieval_geometry")

    def test_target_shadow_proxy_mode_matches_fingerprint_but_fails_closed(self) -> None:
        config = load_config(None)
        config.update(
            {
                "data": "data/tinyshakespeare.txt",
                "cases": 2,
                "span_chars": 4,
                "context_chars": 36,
                "pseudo_masks_per_case": 6,
                "max_candidates_per_example": 64,
                "shuffle_trials": 16,
                "proxy_geometry_mode": "target_shadow",
                "proxy_shadow_pool_per_case": 18,
                "proxy_shadow_keep_per_case": 6,
                "proxy_shadow_anchor_guard_chars": 8,
            }
        )
        receipt, contract = build_receipt(config)

        self.assertEqual(receipt["target_shadow"]["shadow_match"], "target_lattice_fingerprint")
        self.assertTrue(receipt["checks"]["target_shadow_uses_fingerprint"])
        self.assertTrue(receipt["checks"]["target_shadow_no_exact_candidate_bytes"])
        self.assertTrue(receipt["checks"]["pseudo_masks_do_not_overlap_target_anchor_window"])
        self.assertFalse(receipt["checks"]["target_shadow_heldout_gate"])
        self.assertEqual(receipt["verdict"], "fail")
        self.assertFalse(receipt["selector_contract"]["contract_ready"])
        self.assertFalse(contract["ready_for_heldout"])
        for summary in receipt["target_shadow"]["summaries"]:
            self.assertLessEqual(
                summary["selected_mean_fingerprint_distance"],
                summary["pool_mean_fingerprint_distance"],
            )
            self.assertIn("pairwise_edit_mean", summary["fingerprint_fields"])
            self.assertFalse(summary["used_exact_target_candidate_bytes_for_matching"])


if __name__ == "__main__":
    unittest.main()
