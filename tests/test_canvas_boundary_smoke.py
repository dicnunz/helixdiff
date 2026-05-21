from __future__ import annotations

import copy
import unittest

from helixdiff.proofs.canvas_boundary_smoke import build_receipt


class CanvasBoundarySmokeTests(unittest.TestCase):
    def test_canvas_boundary_allows_only_fixed_span_claim(self) -> None:
        receipt = build_receipt()

        self.assertEqual(receipt["verdict"], "pass")
        self.assertTrue(receipt["checks"]["prompt_infilling_sources_present"])
        self.assertTrue(receipt["checks"]["release_standard_discloses_fixed_canvas"])
        self.assertTrue(receipt["checks"]["proof_recipe_is_fixed_span"])
        self.assertTrue(receipt["checks"]["readme_discloses_fixed_canvas_boundary"])
        self.assertFalse(receipt["variable_length_gate_exists"])
        self.assertTrue(receipt["claim_allowed"]["fixed_span_visible_context_repair_plan"])
        self.assertFalse(receipt["claim_allowed"]["flexible_length_or_code_infilling"])

    def test_variable_length_claim_requires_future_gate(self) -> None:
        receipt = build_receipt()
        future = copy.deepcopy(receipt)
        future["variable_length_gate_exists"] = True
        future["checks"]["variable_length_gate_absent"] = False

        self.assertEqual(receipt["claim_allowed"]["flexible_length_or_code_infilling"], False)
        self.assertFalse(all(future["checks"].values()))


if __name__ == "__main__":
    unittest.main()
