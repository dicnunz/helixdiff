from __future__ import annotations

import unittest

from helixdiff.proofs.visible_reranker_oracle_smoke import build_receipt, load_config


class VisibleRerankerOracleSmokeTests(unittest.TestCase):
    def test_receipt_contains_diagnostic_only_visible_reranker_contract(self) -> None:
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
        receipt = build_receipt(config)

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


if __name__ == "__main__":
    unittest.main()
