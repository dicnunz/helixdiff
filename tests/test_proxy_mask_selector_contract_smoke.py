from __future__ import annotations

import unittest

from helixdiff.contract import selector_settings_from_contract
from helixdiff.proofs.proxy_mask_selector_contract_smoke import build_receipt, load_config


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
        self.assertGreater(receipt["pseudo_calibration"]["heldout"]["top4_exact"], receipt["shuffle_falsification"]["top4_exact"])
        self.assertTrue(receipt["checks"]["pseudo_heldout_beats_shuffle_top4"])
        self.assertTrue(contract["ready_for_heldout"])
        settings = selector_settings_from_contract(contract, require_ready=True)
        self.assertIn(settings["selector_anchor"], {"prior", "surface", "visible_reranker"})
        self.assertEqual(settings["selector_margin"], 3.0)


if __name__ == "__main__":
    unittest.main()
