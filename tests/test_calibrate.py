import unittest

from helixdiff.calibrate import calibrate_selector_margins, iter_report_objects, retrieval_lattice_cases


def _sweep_row(margin: float, *, exact: bool, byte_accuracy: float, effect: str, outcome: str) -> dict:
    return {
        "selector_margin": margin,
        "exact": exact,
        "byte_accuracy": byte_accuracy,
        "selector_margin_applied": margin > 0.0,
        "selector_effect": effect,
        "outcome_category": outcome,
    }


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


def _report(cases: list[dict]) -> dict:
    return {
        "checkpoint": "checkpoint.pt",
        "checkpoint_sha256": "abc",
        "case_filter": {"lattice_selector_margin": 0.5},
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
        self.assertAlmostEqual(report["anchor_gap_diagnostics"]["max_exact_anchor_margin_gap"], 0.4)

    def test_insufficient_cases_keeps_recommendation_diagnostic(self) -> None:
        report = calibrate_selector_margins([_report([_case()])], min_cases=4)
        self.assertEqual(report["recommendation"]["status"], "diagnostic_only_insufficient_cases")
        self.assertIsNone(report["recommendation"]["recommended_margin"])
        self.assertEqual(report["recommendation"]["diagnostic_best_margin"], 0.5)

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
