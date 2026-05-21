from __future__ import annotations

import unittest

from helixdiff.gate import STRICT_REPAIR_RECIPE
from helixdiff.recipe import build_strict_repair_commands


class RecipeTests(unittest.TestCase):
    def test_strict_repair_recipe_prints_matching_benchmark_and_gate_commands(self) -> None:
        recipe = build_strict_repair_commands()
        benchmark = recipe["commands"]["benchmark"]
        gate = recipe["commands"]["gate"]

        self.assertEqual(recipe["strict_repair_recipe"], STRICT_REPAIR_RECIPE)
        self.assertIn("--lattice-prior-rerank-top-k", benchmark)
        self.assertIn("4", benchmark)
        self.assertIn("--lattice-verifier-mode", benchmark)
        self.assertIn("dual", benchmark)
        self.assertIn("--lattice-selector-anchor", benchmark)
        self.assertIn("surface", benchmark)
        self.assertIn("--lattice-selector-anchor-sweep", benchmark)
        self.assertIn("prior,surface", benchmark)
        self.assertIn("--lattice-selector-margin-sweep", benchmark)
        self.assertIn("0,1,2,3,5", benchmark)
        self.assertIn("--lattice-local-surface-anchor-calibration", benchmark)
        self.assertIn("--require-repair-proof-contract", gate)
        self.assertTrue(recipe["heavy_slot_required"])


if __name__ == "__main__":
    unittest.main()
