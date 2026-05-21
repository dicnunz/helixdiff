import unittest

from helixdiff.bench import (
    lattice_oracle_case,
    make_marked_cases,
    model_quality_label,
    morphology_candidates,
    nearest_visible_case,
    surface_splice_candidates,
    summarize_lattice_oracle,
    sha256_text,
    split_text,
    visible_suture_candidates,
)
from helixdiff.ngram import BigramGuide
from helixdiff.tokenizer import ByteTokenizer


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
            },
            {
                "oracle_candidate_exact": False,
                "visible_oracle_exact": False,
                "morphology_oracle_exact": False,
                "surface_oracle_exact": False,
                "bridge_oracle_exact": False,
                "unigram_oracle_exact": False,
                "candidate_count": 3,
            },
        ]
        summary = summarize_lattice_oracle(rows)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["oracle_exact_rate"], 0.5)
        self.assertEqual(summary["morphology_oracle_exact_rate"], 0.5)
        self.assertEqual(summary["avg_candidate_count"], 5.0)


if __name__ == "__main__":
    unittest.main()
