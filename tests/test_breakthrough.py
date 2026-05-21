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
        self.assertIn("visible_reranker_oracle_smoke", lane_names)
        self.assertIn("gold_blind_bi_anchor_oracle", lane_names)
        self.assertIn("frozen_selector_heldout_trial", lane_names)
        self.assertIn("visible_hole_reranker", lane_names)
        self.assertTrue(plan["chatgpt_teammate_status"]["usable_this_run"])
        self.assertEqual(plan["chatgpt_teammate_status"]["model_mode"], "Extended Pro")
        self.assertIn("chatgpt.com/c/", plan["chatgpt_teammate_status"]["conversation_url"])
        self.assertIn("visible-reranker oracle smoke", plan["chatgpt_teammate_status"]["claim"])

    def test_top_lane_reuses_predeclared_recipe_and_gate(self) -> None:
        plan = build_breakthrough_plan()
        top_lane = plan["lanes"][0]
        proof_commands = "\n".join(top_lane["proof_commands"])

        self.assertEqual(top_lane["name"], plan["current_best_move"])
        self.assertFalse(top_lane["heavy_slot_required"])
        self.assertIn("helixdiff-visible-reranker-oracle-smoke", proof_commands)
        self.assertIn("--out proof/visible_reranker_oracle_smoke.json", proof_commands)

    def test_bi_anchor_lane_reuses_no_model_oracle(self) -> None:
        plan = build_breakthrough_plan()
        top_lane = [lane for lane in plan["lanes"] if lane["name"] == "gold_blind_bi_anchor_oracle"][0]
        proof_commands = "\n".join(top_lane["proof_commands"])

        self.assertFalse(top_lane["heavy_slot_required"])
        self.assertIn("--candidate-oracle-only", proof_commands)
        self.assertIn("--lattice-bi-anchor-candidates 64", proof_commands)
        self.assertIn("--lattice-bi-anchor-sizes 32,24,16,12,8,6,4", proof_commands)

    def test_strict_lane_reuses_predeclared_recipe_and_gate(self) -> None:
        plan = build_breakthrough_plan()
        strict_lane = [lane for lane in plan["lanes"] if lane["name"] == "strict_repair_lattice_proof"][0]
        proof_commands = "\n".join(strict_lane["proof_commands"])

        self.assertTrue(strict_lane["heavy_slot_required"])
        self.assertIn("--lattice-prior-rerank-top-k 4", proof_commands)
        self.assertIn("--lattice-verifier-mode dual", proof_commands)
        self.assertIn("--lattice-selector-anchor surface", proof_commands)
        self.assertIn("--lattice-selector-anchor-sweep prior,surface,visible_reranker", proof_commands)
        self.assertIn("--lattice-visible-reranker-calibration", proof_commands)
        self.assertIn("--lattice-bi-anchor-candidates 64", proof_commands)
        self.assertIn("--lattice-selector-margin-sweep 0,1,2,3,5", proof_commands)
        self.assertIn("--require-repair-proof-contract", proof_commands)

    def test_heldout_selector_contract_lane_requires_ready_contract(self) -> None:
        plan = build_breakthrough_plan()
        lane = [lane for lane in plan["lanes"] if lane["name"] == "frozen_selector_heldout_trial"][0]
        proof_commands = "\n".join(lane["proof_commands"])

        self.assertTrue(lane["heavy_slot_required"])
        self.assertIn("--lattice-selector-contract proof/selector_contract_strict_repair_8case.json", proof_commands)
        self.assertIn("--lattice-require-selector-contract-ready", proof_commands)
        self.assertIn("--json-out proof/bench_selector_contract_heldout_8case.json", proof_commands)


if __name__ == "__main__":
    unittest.main()
