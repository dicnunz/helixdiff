import unittest

from helixdiff.bench import (
    make_marked_cases,
    model_quality_label,
    morphology_candidates,
    nearest_visible_case,
    sha256_text,
    split_text,
    visible_suture_candidates,
)
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


if __name__ == "__main__":
    unittest.main()
