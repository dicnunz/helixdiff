import unittest

from helixdiff.calibrate import calibrate_selector_margins, iter_report_objects, retrieval_lattice_cases


def _sweep_row(
    margin: float,
    *,
    exact: bool,
    byte_accuracy: float,
    effect: str,
    outcome: str,
    anchor: str | None = None,
) -> dict:
    row = {
        "selector_margin": margin,
        "exact": exact,
        "byte_accuracy": byte_accuracy,
        "selector_margin_applied": margin > 0.0,
        "selector_effect": effect,
        "outcome_category": outcome,
    }
    if anchor is not None:
        row["selector_anchor"] = anchor
    return row


def _case(*, anchor_gap: float = 0.4) -> dict:
    return {
        "anchor_exact": True,
        "anchor_margin_gap": anchor_gap,
        "selector_margin_sweep": [
            _sweep_row(
                0.0,
                exact=False,
                byte_accuracy=0.25,
                effect="raw_verifier_overrode_exact_anchor",
                outcome="raw_verifier_overrode_exact_anchor",
            ),
            _sweep_row(
                0.5,
                exact=True,
                byte_accuracy=1.0,
                effect="margin_rescued_exact_anchor",
                outcome="selected_exact",
            ),
        ],
    }


def _anchor_case() -> dict:
    return {
        "anchor_exact": True,
        "anchor_margin_gap": 0.4,
        "selector_margin_sweep": [
            _sweep_row(
                0.0,
                exact=False,
                byte_accuracy=0.25,
                effect="raw_verifier_overrode_exact_anchor",
                outcome="raw_verifier_overrode_exact_anchor",
            ),
            _sweep_row(
                0.5,
                exact=False,
                byte_accuracy=0.5,
                effect="raw_verifier_selected_nonexact",
                outcome="scored_exact_not_selected",
            ),
        ],
        "selector_anchor_margin_sweep": [
            _sweep_row(
                0.0,
                exact=False,
                byte_accuracy=0.25,
                effect="raw_verifier_overrode_exact_anchor",
                outcome="raw_verifier_overrode_exact_anchor",
                anchor="prior",
            ),
            _sweep_row(
                0.5,
                exact=False,
                byte_accuracy=0.5,
                effect="raw_verifier_selected_nonexact",
                outcome="scored_exact_not_selected",
                anchor="prior",
            ),
            _sweep_row(
                0.0,
                exact=False,
                byte_accuracy=0.25,
                effect="raw_verifier_selected_nonexact",
                outcome="scored_exact_not_selected",
                anchor="surface",
            ),
            _sweep_row(
                0.5,
                exact=True,
                byte_accuracy=1.0,
                effect="margin_rescued_exact_anchor",
                outcome="selected_exact",
                anchor="surface",
            ),
        ],
    }


def _report(cases: list[dict]) -> dict:
    return {
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": "abc",
        "case_filter": {
            "lattice_selector_margin": 0.5,
            "lattice_selector_anchor": "prior",
            "lattice_selector_anchor_sweep": ["prior", "surface"],
        },
        "infill": {
            "retrieval_lattice": {
                "summary": {"exact_match_rate": 1.0, "byte_accuracy": 1.0},
                "cases": cases,
            }
        },
    }


class CalibrateTest(unittest.TestCase):
    def test_iter_report_objects_accepts_object_and_list(self) -> None:
        self.assertEqual(list(iter_report_objects({"a": 1})), [{"a": 1}])
        self.assertEqual(list(iter_report_objects([{"a": 1}, "skip", {"b": 2}])), [{"a": 1}, {"b": 2}])

    def test_retrieval_lattice_cases_extracts_cases(self) -> None:
        cases = [_case()]
        self.assertEqual(retrieval_lattice_cases(_report(cases)), cases)

    def test_calibration_recommends_lowest_safe_best_margin(self) -> None:
        report = calibrate_selector_margins([_report([_case(), _case(anchor_gap=0.3)])], min_cases=2)
        self.assertEqual(report["cases"], 2)
        self.assertEqual(report["margins"]["0"]["exact_match_rate"], 0.0)
        self.assertEqual(report["margins"]["0.5"]["exact_match_rate"], 1.0)
        self.assertEqual(report["recommendation"]["status"], "candidate_margin")
        self.assertEqual(report["recommendation"]["recommended_margin"], 0.5)
        self.assertEqual(report["recommendation"]["exact_lift_vs_margin_0"], 1.0)
        self.assertEqual(report["anchor_margins"]["prior:0.5"]["exact_match_rate"], 1.0)
        self.assertEqual(report["anchor_recommendation"]["status"], "candidate_anchor_margin")
        self.assertEqual(report["anchor_recommendation"]["recommended_selector_anchor"], "prior")
        self.assertEqual(report["anchor_recommendation"]["recommended_margin"], 0.5)
        self.assertAlmostEqual(report["anchor_gap_diagnostics"]["max_exact_anchor_margin_gap"], 0.4)

    def test_anchor_margin_calibration_can_recommend_surface_anchor(self) -> None:
        report = calibrate_selector_margins([_report([_anchor_case(), _anchor_case()])], min_cases=2)
        self.assertEqual(report["anchor_margins"]["prior:0.5"]["exact_match_rate"], 0.0)
        self.assertEqual(report["anchor_margins"]["surface:0.5"]["exact_match_rate"], 1.0)
        self.assertEqual(report["anchor_margins"]["surface:0.5"]["selector_anchor"], "surface")
        self.assertEqual(report["anchor_recommendation"]["status"], "candidate_anchor_margin")
        self.assertEqual(report["anchor_recommendation"]["recommended_selector_anchor"], "surface")
        self.assertEqual(report["anchor_recommendation"]["recommended_margin"], 0.5)
        self.assertEqual(report["anchor_recommendation"]["exact_lift_vs_prior_margin_0"], 1.0)

    def test_report_summary_includes_configured_selector_anchor_sweep(self) -> None:
        report = calibrate_selector_margins([_report([_case(), _case(anchor_gap=0.3)])], min_cases=2)
        summary = report["reports"][0]
        self.assertEqual(summary["configured_selector_anchor"], "prior")
        self.assertEqual(summary["configured_selector_anchor_sweep"], ["prior", "surface"])

    def test_insufficient_cases_keeps_recommendation_diagnostic(self) -> None:
        report = calibrate_selector_margins([_report([_case()])], min_cases=4)
        self.assertEqual(report["recommendation"]["status"], "diagnostic_only_insufficient_cases")
        self.assertIsNone(report["recommendation"]["recommended_margin"])
        self.assertEqual(report["recommendation"]["diagnostic_best_margin"], 0.5)
        self.assertEqual(report["anchor_recommendation"]["status"], "diagnostic_only_insufficient_cases")
        self.assertIsNone(report["anchor_recommendation"]["recommended_selector_anchor"])
        self.assertEqual(report["anchor_recommendation"]["diagnostic_best_selector_anchor"], "prior")

    def test_blocked_exact_raw_margin_is_not_recommended_by_default(self) -> None:
        risky_case = {
            "anchor_exact": False,
            "anchor_margin_gap": 0.7,
            "selector_margin_sweep": [
                _sweep_row(
                    0.0,
                    exact=True,
                    byte_accuracy=1.0,
                    effect="raw_verifier_selected_exact",
                    outcome="selected_exact",
                ),
                _sweep_row(
                    1.0,
                    exact=False,
                    byte_accuracy=0.0,
                    effect="margin_blocked_exact_raw",
                    outcome="margin_blocked_exact_raw",
                ),
            ],
        }
        report = calibrate_selector_margins([_report([risky_case, risky_case])], min_cases=2)
        self.assertEqual(report["margins"]["1"]["margin_blocked_exact_raw_rate"], 1.0)
        self.assertEqual(report["recommendation"]["status"], "candidate_margin")
        self.assertEqual(report["recommendation"]["recommended_margin"], 0.0)


if __name__ == "__main__":
    unittest.main()
