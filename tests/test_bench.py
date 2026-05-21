import unittest

import torch

from helixdiff.bench import (
    build_lattice_candidate_rows,
    classify_retrieval_lattice_outcome,
    classify_selector_effect,
    lattice_oracle_case,
    make_marked_cases,
    model_quality_label,
    morphology_candidates,
    nearest_visible_case,
    parse_selector_margin_sweep,
    rank_lattice_candidates_by_prior,
    score_lattice_verifier,
    select_lattice_row_with_margin,
    selector_margin_sweep_report,
    surface_splice_candidates,
    summarize_lattice_oracle,
    summarize_retrieval_lattice,
    summarize_selector_margin_sweep,
    sha256_text,
    split_text,
    visible_suture_candidates,
)
from helixdiff.ngram import BigramGuide
from helixdiff.tokenizer import ByteTokenizer


class _ProbeModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size
        self.mask_counts: list[list[int]] = []

    def forward(
        self,
        tokens: torch.Tensor,
        t: torch.Tensor,
        *,
        corruption_mode: torch.Tensor | int | None = None,
        mask_fraction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.mask_counts.append((tokens == ByteTokenizer().mask_token_id).sum(dim=1).tolist())
        logits = torch.zeros(tokens.shape[0], tokens.shape[1], self.vocab_size, device=tokens.device)
        logits[..., ByteTokenizer().byte_offset + ord("a")] = 4.0
        return logits


class BenchTest(unittest.TestCase):
    def test_split_text_keeps_validation_tail(self) -> None:
        train, val = split_text("abcdefghij", val_fraction=0.2)
        self.assertEqual(train, "abcdefgh")
        self.assertEqual(val, "ij")

    def test_sha256_text_is_stable(self) -> None:
        self.assertEqual(
            sha256_text("helix"),
            "54a85d2ae7b0a4d8005ab5cf466d4e582c6ea9aa5060b261241ec65a0ea58506",
        )

    def test_make_marked_cases_builds_single_hole(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz " * 20
        cases = make_marked_cases(text, cases=3, span_chars=4, context_chars=8, seed=1)
        self.assertEqual(len(cases), 3)
        for case in cases:
            self.assertEqual(case.count("[["), 1)
            self.assertEqual(case.count("]]"), 1)

    def test_make_marked_cases_can_require_unseen_holes(self) -> None:
        text = "abcdefgh uniquehole zyxwvuts " * 8
        cases = make_marked_cases(
            text,
            cases=1,
            span_chars=10,
            context_chars=8,
            seed=3,
            forbidden_text="abcdefgh zyxwvuts",
            require_unseen_hole=True,
        )
        self.assertEqual(len(cases), 1)
        self.assertNotIn("[[abcdefgh]]", cases[0])

    def test_quality_label_refuses_weak_model_as_strong(self) -> None:
        self.assertEqual(model_quality_label(0.14, 0.02, 0.1), "mechanism_checkpoint")
        self.assertEqual(model_quality_label(0.5, 0.3, 0.6), "strong_laptop_checkpoint")

    def test_nearest_visible_baseline_uses_visible_suture_match(self) -> None:
        row = nearest_visible_case(
            tokenizer=ByteTokenizer(),
            marked_text="alpha Tailor beta [[Tail]]or beta",
        )
        self.assertEqual(row["predicted_hole"], "Tail")
        self.assertTrue(row["exact"])
        self.assertTrue(row["frozen_context_unchanged"])
        self.assertEqual(row["nearest_visible_source"], "before")

    def test_visible_suture_candidates_do_not_join_across_hidden_gap(self) -> None:
        rows = visible_suture_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="aa b[[bc]]c dd",
            limit=4,
        )
        self.assertTrue(rows)
        self.assertNotEqual(rows[0]["predicted_hole"], "bc")

    def test_morphology_candidates_include_word_completion_from_train_text(self) -> None:
        rows = morphology_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="And Gabr[[iel']]s pumps were bright",
            train_text="Gabriel's horn sounded. Gabriel's name appears here.",
            limit=8,
        )
        self.assertIn("iel'", {row["predicted_hole"] for row in rows})

    def test_morphology_candidates_include_hyphen_prefix_from_train_prefixes(self) -> None:
        rows = morphology_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="then go a [[bat-]]fowling at night",
            train_text="bats batter battle. bats batter battle. cats cater cattle.",
            limit=16,
        )
        self.assertIn("bat-", {row["predicted_hole"] for row in rows})

    def test_morphology_candidates_include_name_possessive_suffix(self) -> None:
        rows = morphology_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="And Gabr[[iel']]s pumps were bright",
            train_text="Nathaniel's book and Michael's cloak are nearby.",
            limit=16,
        )
        self.assertIn("iel'", {row["predicted_hole"] for row in rows})

    def test_morphology_candidates_include_name_stem_prefix(self) -> None:
        rows = morphology_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="And [[Gabr]]iel's pumps were bright",
            train_text="Nathaniel's book and Michael's cloak are nearby.",
            limit=16,
        )
        self.assertIn("Gabr", {row["predicted_hole"] for row in rows})

    def test_morphology_candidates_include_speaker_label_completion(self) -> None:
        rows = morphology_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="with needle and thread.\n\nTai[[lor:]]\nBut did you not",
            train_text="The tailor mends. Another Tailor speaks.",
            limit=16,
        )
        self.assertIn("lor:", {row["predicted_hole"] for row in rows})

    def test_morphology_candidates_include_dash_bridge(self) -> None:
        rows = morphology_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="thy fortune slee[[p--d]]ie, rather",
            train_text="sleep sleeper die fie lie.",
            limit=32,
        )
        self.assertIn("p--d", {row["predicted_hole"] for row in rows})

    def test_morphology_candidates_include_hyphen_morpheme_bridge(self) -> None:
        rows = morphology_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="full of con[[y-ca]]tching!",
            train_text="cony rabbits and catching fish.",
            limit=32,
        )
        self.assertIn("y-ca", {row["predicted_hole"] for row in rows})

    def test_surface_splice_candidates_mine_possessive_prefix(self) -> None:
        rows = surface_splice_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="And [[Gabr]]iel's pumps were bright",
            train_text="Gabriel's trumpet and Michael's cloak are nearby.",
            limit=8,
        )
        self.assertIn("Gabr", {row["predicted_hole"] for row in rows})

    def test_surface_splice_candidates_mine_punctuation_inside_surface_unit(self) -> None:
        rows = surface_splice_candidates(
            tokenizer=ByteTokenizer(),
            marked_text="Thou let'st thy fortune slee[[p--d]]ie, rather;",
            train_text="sleep--die, cony-catching! Tailor:",
            limit=8,
        )
        self.assertIn("p--d", {row["predicted_hole"] for row in rows})

    def test_lattice_oracle_case_reports_morphology_hit(self) -> None:
        tokenizer = ByteTokenizer()
        train_text = "Nathaniel's book and Michael's cloak are nearby."
        row = lattice_oracle_case(
            tokenizer=tokenizer,
            marked_text="And Gabr[[iel']]s pumps were bright",
            guide=BigramGuide.from_text(train_text, tokenizer),
            train_text=train_text,
            visible_limit=4,
            morphology_limit=16,
        )
        self.assertTrue(row["oracle_candidate_exact"])
        self.assertTrue(row["morphology_oracle_exact"])
        self.assertIn("morphology_name_possessive_suffix", row["exact_candidate_sources"])
        self.assertTrue(row["prior_selected_exact"])

    def test_lattice_oracle_case_reports_surface_hit(self) -> None:
        tokenizer = ByteTokenizer()
        train_text = "sleep--die, cony-catching! Tailor:"
        row = lattice_oracle_case(
            tokenizer=tokenizer,
            marked_text="Thou let'st thy fortune slee[[p--d]]ie, rather;",
            guide=BigramGuide.from_text(train_text, tokenizer),
            train_text=train_text,
            visible_limit=4,
            morphology_limit=4,
            surface_limit=8,
        )
        self.assertTrue(row["oracle_candidate_exact"])
        self.assertTrue(row["surface_oracle_exact"])
        self.assertTrue(any(source.startswith("surface_") for source in row["exact_candidate_sources"]))
        self.assertTrue(row["prior_selected_exact"])
        self.assertEqual(row["candidate_summaries"][0]["prior_rank"], 0)
        self.assertEqual(row["prior_exact_rank"], 0)
        self.assertTrue(row["prior_exact_in_top4"])

    def test_prior_topk_can_isolate_reranker_candidates_without_model(self) -> None:
        tokenizer = ByteTokenizer()
        train_text = "sleep sleeper die fie lie."
        marked_text = "Thou let'st thy fortune slee[[p--d]]ie, rather;"
        ranked = rank_lattice_candidates_by_prior(
            build_lattice_candidate_rows(
                tokenizer=tokenizer,
                marked_text=marked_text,
                guide=BigramGuide.from_text(train_text, tokenizer),
                train_text=train_text,
                visible_limit=4,
                morphology_limit=32,
                surface_limit=4,
            )
        )
        exact = [row for row in ranked if row["predicted_hole"] == "p--d"]
        self.assertEqual(len(exact), 1)
        self.assertLess(exact[0]["prior_rank"], 4)
        self.assertEqual([row["prior_rank"] for row in ranked[:4]], [0, 1, 2, 3])

    def test_dual_lattice_verifier_runs_loo_and_full_hole_probes(self) -> None:
        tokenizer = ByteTokenizer()
        repaired = torch.tensor(tokenizer.encode("xxaaay", add_bos=True, add_eos=True), dtype=torch.long)
        model = _ProbeModel(tokenizer.vocab_size)
        score, scores = score_lattice_verifier(
            model=model,
            tokenizer=tokenizer,
            repaired_ids=repaired,
            hole_start=3,
            hole_end=6,
            guide=None,
            guidance=0.0,
            temperature=1.0,
            top_k=16,
            verifier_mode="dual",
        )
        self.assertTrue(torch.isfinite(torch.tensor(score)).item())
        self.assertEqual(set(scores), {"suture_loo", "full_hole"})
        self.assertEqual(model.mask_counts[-2:], [[1, 1, 1], [3]])

    def test_selector_margin_requires_diffusion_to_clear_prior_anchor(self) -> None:
        anchor = torch.tensor([1])
        challenger = torch.tensor([2])
        selected_score, selected_ids, selected_row, report = select_lattice_row_with_margin(
            [
                (1.0, anchor, {"predicted_hole": "anchor", "prior_rank": 0, "exact": True, "byte_accuracy": 1.0}),
                (
                    1.4,
                    challenger,
                    {"predicted_hole": "challenger", "prior_rank": 1, "exact": False, "byte_accuracy": 0.0},
                ),
            ],
            selector_margin=0.5,
        )
        self.assertEqual(selected_score, 1.0)
        self.assertTrue(torch.equal(selected_ids, anchor))
        self.assertEqual(selected_row["predicted_hole"], "anchor")
        self.assertTrue(report["selector_margin_applied"])
        self.assertFalse(report["raw_best_exact"])
        self.assertTrue(report["anchor_exact"])
        self.assertEqual(report["raw_best_byte_accuracy"], 0.0)
        self.assertEqual(report["anchor_byte_accuracy"], 1.0)
        self.assertEqual(report["selector_effect"], "margin_rescued_exact_anchor")

    def test_selector_margin_sweep_reuses_scored_options(self) -> None:
        anchor = torch.tensor([1])
        challenger = torch.tensor([2])
        sweep = selector_margin_sweep_report(
            [
                (1.0, anchor, {"predicted_hole": "anchor", "prior_rank": 0, "exact": True, "byte_accuracy": 1.0}),
                (
                    1.4,
                    challenger,
                    {"predicted_hole": "challenger", "prior_rank": 1, "exact": False, "byte_accuracy": 0.0},
                ),
            ],
            selector_margins=[0.0, 0.5],
            oracle_candidate_exact=True,
            oracle_candidate_exact_in_scored_set=True,
        )
        self.assertEqual([row["selector_margin"] for row in sweep], [0.0, 0.5])
        self.assertEqual(sweep[0]["selected_hole"], "challenger")
        self.assertEqual(sweep[0]["outcome_category"], "raw_verifier_overrode_exact_anchor")
        self.assertEqual(sweep[1]["selected_hole"], "anchor")
        self.assertEqual(sweep[1]["outcome_category"], "selected_exact")
        self.assertEqual(sweep[1]["selector_effect"], "margin_rescued_exact_anchor")

    def test_parse_selector_margin_sweep_dedupes_and_sorts(self) -> None:
        self.assertEqual(parse_selector_margin_sweep("3, 0,1,3,,2"), [0.0, 1.0, 2.0, 3.0])

    def test_retrieval_lattice_outcome_taxonomy_names_actionable_misses(self) -> None:
        self.assertEqual(
            classify_selector_effect(raw_best_exact=False, anchor_exact=True, margin_applied=False),
            "raw_verifier_overrode_exact_anchor",
        )
        self.assertEqual(
            classify_retrieval_lattice_outcome(
                {
                    "exact": False,
                    "oracle_candidate_exact": True,
                    "oracle_candidate_exact_in_scored_set": True,
                    "selector_effect": "raw_verifier_overrode_exact_anchor",
                }
            ),
            "raw_verifier_overrode_exact_anchor",
        )
        self.assertEqual(
            classify_retrieval_lattice_outcome(
                {
                    "exact": False,
                    "oracle_candidate_exact": True,
                    "oracle_candidate_exact_in_scored_set": False,
                    "selector_effect": "raw_verifier_selected_nonexact",
                }
            ),
            "oracle_outside_scored_set",
        )

    def test_retrieval_lattice_summary_reports_selector_bottlenecks(self) -> None:
        rows = [
            {
                "byte_accuracy": 1.0,
                "exact": True,
                "frozen_context_unchanged": True,
                "candidate_summaries": [{"exact": True}],
                "oracle_candidate_exact": True,
                "oracle_candidate_exact_in_scored_set": True,
                "prior_selected_exact": True,
                "raw_best_exact": False,
                "anchor_exact": True,
                "selector_effect": "margin_rescued_exact_anchor",
                "outcome_category": "selected_exact",
                "selector_margin_applied": True,
                "selector_margin_sweep": [
                    {
                        "selector_margin": 0.0,
                        "exact": False,
                        "byte_accuracy": 0.0,
                        "selector_margin_applied": False,
                        "selector_effect": "raw_verifier_overrode_exact_anchor",
                        "outcome_category": "raw_verifier_overrode_exact_anchor",
                    },
                    {
                        "selector_margin": 0.5,
                        "exact": True,
                        "byte_accuracy": 1.0,
                        "selector_margin_applied": True,
                        "selector_effect": "margin_rescued_exact_anchor",
                        "outcome_category": "selected_exact",
                    },
                ],
                "scored_candidate_count": 4,
                "prior_exact_rank": 0,
            },
            {
                "byte_accuracy": 0.5,
                "exact": False,
                "frozen_context_unchanged": True,
                "candidate_summaries": [{"exact": False}],
                "oracle_candidate_exact": True,
                "oracle_candidate_exact_in_scored_set": False,
                "prior_selected_exact": False,
                "raw_best_exact": False,
                "anchor_exact": False,
                "selector_effect": "raw_verifier_selected_nonexact",
                "outcome_category": "oracle_outside_scored_set",
                "selector_margin_applied": False,
                "selector_margin_sweep": [
                    {
                        "selector_margin": 0.0,
                        "exact": False,
                        "byte_accuracy": 0.5,
                        "selector_margin_applied": False,
                        "selector_effect": "raw_verifier_selected_nonexact",
                        "outcome_category": "oracle_outside_scored_set",
                    },
                    {
                        "selector_margin": 0.5,
                        "exact": False,
                        "byte_accuracy": 0.5,
                        "selector_margin_applied": False,
                        "selector_effect": "raw_verifier_selected_nonexact",
                        "outcome_category": "oracle_outside_scored_set",
                    },
                ],
                "scored_candidate_count": 4,
                "prior_exact_rank": 7,
            },
        ]
        summary = summarize_retrieval_lattice(rows)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["byte_accuracy"], 0.75)
        self.assertEqual(summary["oracle_candidate_exact_rate"], 1.0)
        self.assertEqual(summary["oracle_candidate_exact_in_scored_set_rate"], 0.5)
        self.assertEqual(summary["prior_selected_exact_rate"], 0.5)
        self.assertEqual(summary["raw_best_exact_rate"], 0.0)
        self.assertEqual(summary["anchor_exact_rate"], 0.5)
        self.assertEqual(summary["selector_margin_applied_rate"], 0.5)
        self.assertEqual(summary["avg_scored_candidate_count"], 4.0)
        self.assertEqual(summary["avg_prior_exact_rank"], 3.5)
        self.assertEqual(summary["outcome_categories"], {"oracle_outside_scored_set": 1, "selected_exact": 1})
        self.assertEqual(
            summary["selector_effects"],
            {"margin_rescued_exact_anchor": 1, "raw_verifier_selected_nonexact": 1},
        )
        self.assertEqual(summary["margin_rescued_exact_rate"], 0.5)
        self.assertEqual(summary["raw_verifier_overrode_exact_anchor_rate"], 0.0)
        self.assertEqual(summary["selector_margin_sweep"]["0"]["exact_match_rate"], 0.0)
        self.assertEqual(summary["selector_margin_sweep"]["0.5"]["exact_match_rate"], 0.5)
        self.assertEqual(summary["selector_margin_sweep"]["0.5"]["selector_margin_applied_rate"], 0.5)

    def test_selector_margin_sweep_summary_handles_empty_rows(self) -> None:
        self.assertEqual(summarize_selector_margin_sweep([]), {})

    def test_lattice_oracle_summary_splits_sources(self) -> None:
        rows = [
            {
                "oracle_candidate_exact": True,
                "visible_oracle_exact": False,
                "morphology_oracle_exact": True,
                "surface_oracle_exact": False,
                "bridge_oracle_exact": False,
                "unigram_oracle_exact": False,
                "candidate_count": 7,
                "prior_selected_exact": True,
                "prior_exact_in_top4": True,
                "prior_exact_in_top8": True,
                "prior_exact_rank": 0,
            },
            {
                "oracle_candidate_exact": False,
                "visible_oracle_exact": False,
                "morphology_oracle_exact": False,
                "surface_oracle_exact": False,
                "bridge_oracle_exact": False,
                "unigram_oracle_exact": False,
                "candidate_count": 3,
                "prior_selected_exact": False,
                "prior_exact_in_top4": False,
                "prior_exact_in_top8": False,
                "prior_exact_rank": None,
            },
        ]
        summary = summarize_lattice_oracle(rows)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["oracle_exact_rate"], 0.5)
        self.assertEqual(summary["morphology_oracle_exact_rate"], 0.5)
        self.assertEqual(summary["avg_candidate_count"], 5.0)
        self.assertEqual(summary["prior_selected_exact_rate"], 0.5)
        self.assertEqual(summary["prior_top4_exact_rate"], 0.5)
        self.assertEqual(summary["prior_top8_exact_rate"], 0.5)
        self.assertEqual(summary["avg_prior_exact_rank"], 0.0)


if __name__ == "__main__":
    unittest.main()
