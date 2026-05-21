from __future__ import annotations

import copy
import unittest

from helixdiff.proofs.visible_reranker_oracle_smoke import (
    build_receipt,
    evaluate_visible_reranker_oracle_contract,
    load_config,
)


class VisibleRerankerOracleSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config = load_config(None)
        config.update(
            {
                "data": "data/tinyshakespeare.txt",
                "cases": 2,
                "span_chars": 4,
                "context_chars": 36,
                "shuffle_trials": 3,
                "max_candidates_per_example": 32,
            }
        )
        cls.receipt = build_receipt(config)

    def test_receipt_contains_diagnostic_only_visible_reranker_contract(self) -> None:
        receipt = self.receipt
        self.assertFalse(receipt["model_load"])
        self.assertIn("git_dirty", receipt)
        self.assertEqual(receipt["verdict"], "pass")
        self.assertIn("prior", receipt["selector_anchor_sweep"])
        self.assertIn("surface", receipt["selector_anchor_sweep"])
        self.assertIn("visible_reranker", receipt["selector_anchor_sweep"])
        self.assertFalse(receipt["selector_anchor_sweep"]["visible_reranker"]["apply"])
        self.assertIn("calibration", receipt["selector_anchor_sweep"]["visible_reranker"])
        self.assertEqual(receipt["leakage"]["eval_doc_hits"], 0)
        self.assertEqual(receipt["leakage"]["same_doc_hits"], 0)
        self.assertFalse(receipt["leakage"]["masked_bytes_seen_by_features"])
        self.assertFalse(receipt["leakage"]["gold_used_for_selection"])
        self.assertIn("shuffle_falsification", receipt)
        self.assertEqual(receipt["shuffle_falsification"]["num_trials"], 3)
        self.assertIn("bi_anchor_gold_in_lattice_rate", receipt["lattice"])
        self.assertTrue(receipt["proof_contract"]["passed"])
        self.assertFalse(receipt["proof_contract"]["missing"])
        self.assertTrue(receipt["proof_contract"]["checks"]["shuffle_falsification_drops_selected_exact"])
        self.assertTrue(receipt["proof_contract"]["checks"]["shuffle_falsification_drops_lattice_coverage"])

    def test_contract_fails_when_shuffle_does_not_falsify(self) -> None:
        receipt = copy.deepcopy(self.receipt)
        receipt["shuffle_falsification"]["visible_reranker_selected_exact"] = receipt["selector_anchor_sweep"][
            "visible_reranker"
        ]["selected_exact"]

        contract = evaluate_visible_reranker_oracle_contract(receipt)

        self.assertFalse(contract["passed"])
        self.assertIn("shuffle_falsification_drops_selected_exact", contract["missing"])


if __name__ == "__main__":
    unittest.main()
