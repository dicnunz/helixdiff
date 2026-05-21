from __future__ import annotations

import unittest

from helixdiff.lattice.in_document_echo import in_document_echo_candidates, redacted_document_bytes
from helixdiff.tokenizer import ByteTokenizer


class InDocumentEchoLatticeTests(unittest.TestCase):
    def test_echo_candidates_use_redacted_same_document_without_anchor_overlap(self) -> None:
        tokenizer = ByteTokenizer()
        document = (
            "alpha beta ZETA delta. "
            "unrelated filler keeps the second echo outside the forbidden window. "
            "alpha beta ZETA delta."
        )
        start = document.index("ZETA")
        span_start = len(document[:start].encode("utf-8"))
        span_end = len(document[: start + len("ZETA")].encode("utf-8"))

        redacted = redacted_document_bytes(document, span_start, span_end)
        self.assertNotIn(b"ZETA", redacted[span_start:span_end])

        rows = in_document_echo_candidates(
            tokenizer=tokenizer,
            document_text=document,
            span_start=span_start,
            span_end=span_end,
            context_bytes=16,
            anchor_window_bytes=24,
            anchor_sizes=[11, 8, 6, 4],
            limit=16,
        )
        predicted = {row["predicted_hole"]: row for row in rows}

        self.assertIn("ZETA", predicted)
        row = predicted["ZETA"]
        self.assertEqual(row["source"], "in_document_echo")
        self.assertFalse(row["overlaps_target"])
        self.assertFalse(row["overlaps_anchor_window"])
        self.assertFalse(row["sentinel_in_source_window"])
        self.assertIn("exact_bi_anchor_echo", {source["anchor_mode"] for source in row["sources"]})

    def test_anchor_window_forbids_nearby_copies(self) -> None:
        tokenizer = ByteTokenizer()
        document = "alpha beta ZETA delta. alpha beta ZETA delta."
        start = document.index("ZETA")
        span_start = len(document[:start].encode("utf-8"))
        span_end = len(document[: start + len("ZETA")].encode("utf-8"))

        rows = in_document_echo_candidates(
            tokenizer=tokenizer,
            document_text=document,
            span_start=span_start,
            span_end=span_end,
            context_bytes=16,
            anchor_window_bytes=len(document.encode("utf-8")),
            anchor_sizes=[11, 8, 6, 4],
            limit=16,
        )

        self.assertNotIn("ZETA", {row["predicted_hole"] for row in rows})


if __name__ == "__main__":
    unittest.main()

