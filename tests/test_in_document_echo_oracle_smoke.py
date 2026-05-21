from __future__ import annotations

import unittest

from helixdiff.proofs.in_document_echo_oracle_smoke import build_receipt


class InDocumentEchoOracleSmokeTests(unittest.TestCase):
    def test_seeded_receipt_passes_with_redacted_containment_lift(self) -> None:
        receipt = build_receipt({"cases": 8, "shuffle_trials": 4, "seed": 42})

        self.assertFalse(receipt["model_load"])
        self.assertTrue(receipt["redaction"]["target_span_redacted_before_index"])
        self.assertFalse(receipt["redaction"]["target_bytes_available_to_echo_index"])
        self.assertFalse(receipt["redaction"]["masked_bytes_seen_by_features"])
        self.assertEqual(receipt["source_policy"]["target_window_overlap_hits"], 0)
        self.assertEqual(receipt["source_policy"]["target_anchor_window_overlap_hits"], 0)
        self.assertEqual(receipt["source_policy"]["same_offset_hits"], 0)
        self.assertLessEqual(
            receipt["echo_lattice"]["candidate_count_p95"],
            receipt["max_candidates_per_example"],
        )
        self.assertEqual(receipt["verdict"], "pass")
        self.assertGreater(
            receipt["combined_lattice"]["gold_in_combined_lattice_at_128"],
            receipt["combined_lattice"]["gold_in_prior_lattice_at_128"],
        )
        self.assertLess(
            receipt["shuffle_falsification"]["gold_in_echo_lattice_at_128"],
            receipt["echo_lattice"]["gold_in_echo_lattice_at_128"],
        )


if __name__ == "__main__":
    unittest.main()
