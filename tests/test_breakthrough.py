from __future__ import annotations

import unittest

from helixdiff.breakthrough import CLAIM_BOUNDARY, build_breakthrough_plan


class BreakthroughPlanTests(unittest.TestCase):
    def test_plan_preserves_honest_boundary_and_source_backing(self) -> None:
        plan = build_breakthrough_plan()
        source_ids = {source["id"] for source in plan["sources"]}
        lane_names = {lane["name"] for lane in plan["lanes"]}

        self.assertIn("not global language-model SOTA", CLAIM_BOUNDARY)
        self.assertIn("arxiv:2406.07524", source_ids)
        self.assertIn("arxiv:2503.09573", source_ids)
        self.assertIn("arxiv:2310.16834", source_ids)
        self.assertIn("arxiv:2502.09992", source_ids)
        self.assertIn("arxiv:2510.18114", source_ids)
        self.assertIn("strict_repair_lattice_proof", lane_names)
        self.assertIn("visible_hole_reranker", lane_names)
        self.assertFalse(plan["chatgpt_teammate_status"]["usable_this_run"])
        self.assertIn("Transport closed", plan["chatgpt_teammate_status"]["blocker"])

    def test_top_lane_reuses_predeclared_recipe_and_gate(self) -> None:
        plan = build_breakthrough_plan()
        top_lane = plan["lanes"][0]
        proof_commands = "\n".join(top_lane["proof_commands"])

        self.assertEqual(top_lane["name"], plan["current_best_move"])
        self.assertTrue(top_lane["heavy_slot_required"])
        self.assertIn("--lattice-prior-rerank-top-k 4", proof_commands)
        self.assertIn("--lattice-verifier-mode dual", proof_commands)
        self.assertIn("--lattice-selector-anchor surface", proof_commands)
        self.assertIn("--lattice-selector-anchor-sweep prior,surface,visible_reranker", proof_commands)
        self.assertIn("--lattice-visible-reranker-calibration", proof_commands)
        self.assertIn("--lattice-selector-margin-sweep 0,1,2,3,5", proof_commands)
        self.assertIn("--require-repair-proof-contract", proof_commands)


if __name__ == "__main__":
    unittest.main()
