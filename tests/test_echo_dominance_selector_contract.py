from __future__ import annotations

import unittest

from helixdiff.lattice.echo_dominance_selector import rank_with_echo_dominance
from helixdiff.lattice.in_document_echo import in_document_echo_candidates
from helixdiff.proofs.echo_dominance_selector_contract_smoke import build_receipt
from helixdiff.tokenizer import ByteTokenizer


class EchoDominanceSelectorContractTests(unittest.TestCase):
    def test_exact_echo_candidate_can_dominate_blank_and_swapped_nulls(self) -> None:
        tokenizer = ByteTokenizer()
        document = (
            "alpha beta ZETA delta. "
            "unrelated filler keeps the second echo outside the forbidden window. "
            "alpha beta ZETA delta."
        )
        start = document.index("ZETA")
        span_start = len(document[:start].encode("utf-8"))
        span_end = len(document[: start + len("ZETA")].encode("utf-8"))
        prior = [
            {
                "ids": tokenizer.encode("NOPE", add_bos=False, add_eos=False),
                "predicted_hole": "NOPE",
                "source": "unit_test_prior",
                "rank": 0,
            }
        ]
        echo = in_document_echo_candidates(
            tokenizer=tokenizer,
            document_text=document,
            span_start=span_start,
            span_end=span_end,
            context_bytes=16,
            anchor_window_bytes=24,
            anchor_sizes=[11, 8, 6, 4],
            limit=16,
        )

        ranked, summary = rank_with_echo_dominance(
            tokenizer=tokenizer,
            document_text=document,
            span_start=span_start,
            span_end=span_end,
            prior=prior,
            echo=echo,
            context_bytes=16,
            anchor_sizes=[11, 8, 6, 4],
            anchor_window_bytes=24,
            null_contexts=[
                {"name": "blank", "left": b"", "right": b""},
                {"name": "swapped_edges", "left": b"gamma ", "right": b" omega"},
            ],
            min_real_null_margin=2.0,
            max_rank_promotions=1,
            max_candidates=16,
        )

        self.assertEqual(ranked[0]["predicted_hole"], "ZETA")
        self.assertEqual(summary["promoted_cases"], 1)
        self.assertEqual(summary["promoted_under_blank"], 0)
        self.assertEqual(summary["promoted_under_swapped_edges"], 0)
        self.assertGreaterEqual(summary["promoted_causal_margin_min"], 2.0)

    def test_receipt_kills_direct_promotion_claim_when_top4_does_not_lift(self) -> None:
        receipt, contract = build_receipt({"cases": 8, "shuffle_trials": 4})

        self.assertEqual(receipt["verdict"], "pass")
        self.assertFalse(receipt["model_load"])
        self.assertTrue(receipt["redaction"]["target_span_redacted_before_echo_index"])
        self.assertFalse(receipt["redaction"]["target_bytes_available_to_selector"])
        self.assertFalse(receipt["redaction"]["masked_bytes_seen_by_features"])
        self.assertEqual(receipt["source_policy"]["target_window_overlap_hits"], 0)
        self.assertEqual(receipt["source_policy"]["target_anchor_window_overlap_hits"], 0)
        self.assertEqual(receipt["source_policy"]["same_offset_hits"], 0)
        self.assertFalse(receipt["selector_contract"]["ready"])
        self.assertFalse(receipt["selector_contract"]["apply"])
        self.assertTrue(receipt["selector_contract"]["diagnostic_only"])
        self.assertEqual(receipt["selector_contract"]["status"], "killed_fail_closed")
        self.assertTrue(receipt["fail_closed"]["kill_triggered"])
        self.assertFalse(receipt["claim_allowed"]["echo_dominance_selector"])
        self.assertGreater(
            receipt["target_after_freeze"]["gold_in_combined_lattice_at_128"],
            receipt["target_after_freeze"]["gold_in_prior_lattice_at_128"],
        )
        self.assertEqual(
            receipt["target_after_freeze"]["contract_top4_exact"],
            receipt["target_after_freeze"]["prior_top4_exact"],
        )
        self.assertFalse(contract["ready_for_heldout"])
        self.assertFalse(contract["apply"])
        self.assertTrue(contract["diagnostic_only"])


if __name__ == "__main__":
    unittest.main()
