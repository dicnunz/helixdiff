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
        self.assertIn("arxiv:2604.03677", source_ids)
        self.assertIn("arxiv:2602.01326", source_ids)
        self.assertIn("arxiv:2602.15014", source_ids)
        self.assertIn("strict_repair_lattice_proof", lane_names)
        self.assertIn("visible_reranker_oracle_smoke", lane_names)
        self.assertIn("proxy_mask_selector_contract_smoke", lane_names)
        self.assertIn("gold_blind_bi_anchor_oracle", lane_names)
        self.assertIn("frozen_selector_heldout_trial", lane_names)
        self.assertIn("visible_hole_reranker", lane_names)
        self.assertIn("prompt_canvas_curriculum", lane_names)
        self.assertIn("proxy_mask_selector_contract_smoke", lane_names)
        self.assertIn(
            "proxy-mask receipts cannot be cited as target-lift evidence unless --require-useful-ratchet passes",
            plan["release_standard"],
        )
        self.assertIn("fixed-length canvas limits disclosed until a variable-length repair gate exists", plan["release_standard"])
        self.assertTrue(plan["chatgpt_teammate_status"]["usable_this_run"])
        self.assertEqual(plan["chatgpt_teammate_status"]["model_mode"], "Extended Pro")
        self.assertEqual(plan["chatgpt_teammate_status"]["latest_response_status"], "answered")
        self.assertIsNone(plan["chatgpt_teammate_status"]["blocker"])
        self.assertIn("target-shadow proxy", plan["chatgpt_teammate_status"]["latest_recommendation"])
        self.assertIn("redacted target lattice fingerprint", plan["chatgpt_teammate_status"]["claim"])
        prior_note = plan["chatgpt_teammate_status"]["recorded_prior_note"]
        self.assertEqual(prior_note["model_mode"], "Extended Pro")
        self.assertIn("chatgpt.com/c/", prior_note["conversation_url"])
        self.assertIn("target-shadow proxy calibration", prior_note["contribution"])
        self.assertEqual(prior_note["verification_status"], "current_chrome_readback_reverified_url")

    def test_top_lane_reuses_predeclared_recipe_and_gate(self) -> None:
        plan = build_breakthrough_plan()
        top_lane = [lane for lane in plan["lanes"] if lane["name"] == "visible_reranker_oracle_smoke"][0]
        proof_commands = "\n".join(top_lane["proof_commands"])

        self.assertFalse(top_lane["heavy_slot_required"])
        self.assertIn("counterfactual context", top_lane["kill_condition"])
        self.assertIn("helixdiff-visible-reranker-oracle-smoke", proof_commands)
        self.assertIn("--out proof/visible_reranker_oracle_smoke.json", proof_commands)

    def test_current_best_move_is_proxy_mask_contract_smoke(self) -> None:
        plan = build_breakthrough_plan()
        lane = [lane for lane in plan["lanes"] if lane["name"] == plan["current_best_move"]][0]
        proof_commands = "\n".join(lane["proof_commands"])

        self.assertEqual(lane["name"], "proxy_mask_selector_contract_smoke")
        self.assertFalse(lane["heavy_slot_required"])
        self.assertIn("helixdiff-proxy-mask-selector-contract-smoke", proof_commands)
        self.assertIn("--contract-out proof/proxy_mask_selector_contract_smoke_contract.json", proof_commands)
        self.assertIn("claim_gate blocks target-lift wording", lane["pass_condition"])
        self.assertIn("--require-useful-ratchet fails", lane["kill_condition"])

    def test_proxy_mask_selector_contract_lane_is_visible_only(self) -> None:
        plan = build_breakthrough_plan()
        lane = [lane for lane in plan["lanes"] if lane["name"] == "proxy_mask_selector_contract_smoke"][0]
        proof_commands = "\n".join(lane["proof_commands"])

        self.assertFalse(lane["heavy_slot_required"])
        self.assertIn("helixdiff-proxy-mask-selector-contract-smoke", proof_commands)
        self.assertIn("--contract-out proof/proxy_mask_selector_contract_smoke_contract.json", proof_commands)
        self.assertIn("target_metric_used_for_selection=false", lane["pass_condition"])

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
