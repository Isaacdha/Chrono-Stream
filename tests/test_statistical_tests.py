import unittest
from urllib.parse import urlparse

from chrono_stream.statistical_tests import (
    ARIMA_DECISION_AID_KEYS,
    ARIMA_TEST_KEYS,
    SARIMA_DECISION_AID_KEYS,
    SARIMA_TEST_KEYS,
    STATISTICAL_TESTS,
    copy_ready_test_handbook,
    copy_ready_test_note,
    test_keys_for_model,
)


class StatisticalTestInformationTests(unittest.TestCase):
    def test_every_catalog_entry_has_a_complete_audited_review(self) -> None:
        self.assertGreaterEqual(len(STATISTICAL_TESTS), 20)
        for key, item in STATISTICAL_TESTS.items():
            with self.subTest(test=key):
                self.assertGreaterEqual(len(item.purpose), 65)
                self.assertGreater(len(item.statistic), 20)
                self.assertGreater(len(item.reference_distribution), 4)
                self.assertGreater(len(item.decision_rule), 90)
                self.assertGreater(len(item.chrono_stream), 150)
                self.assertGreater(len(item.interpretation), 100)
                self.assertGreater(len(item.assumptions_and_caveats), 150)
                self.assertGreater(len(item.literature_review), 300)
                self.assertGreaterEqual(len(item.references), 1)

                if item.formal:
                    self.assertTrue(item.null_hypothesis.startswith("H0"))
                    self.assertTrue(item.alternative_hypothesis.startswith("H1"))
                    self.assertTrue(
                        "alpha" in item.decision_rule.lower()
                        or "critical" in item.decision_rule.lower()
                    )
                else:
                    self.assertIsNone(item.null_hypothesis)
                    self.assertIsNone(item.alternative_hypothesis)

                for reference in item.references:
                    parsed = urlparse(reference.url)
                    self.assertEqual(parsed.scheme, "https")
                    self.assertTrue(parsed.netloc)
                    self.assertGreater(len(reference.apa), 50)
                    self.assertGreater(len(reference.contribution), 40)

                note = copy_ready_test_note(key)
                for heading in (
                    "Null hypothesis (H0)",
                    "Alternative hypothesis (H1)",
                    "Statistic",
                    "Reference distribution",
                    "Decision rule",
                    "Chrono Stream implementation",
                    "Literature review",
                    "APA 7 references",
                ):
                    self.assertIn(heading, note)
                self.assertNotIn("\n\n\n", note)
                self.assertNotIn("Source-to-test audit", note)
                self.assertNotIn("accept h0", note.lower())

    def test_arima_and_sarima_handbooks_cover_formal_and_nonformal_items(self) -> None:
        arima_keys = test_keys_for_model("arima")
        sarima_keys = test_keys_for_model("sarima")
        self.assertEqual(arima_keys, (*ARIMA_TEST_KEYS, *ARIMA_DECISION_AID_KEYS))
        self.assertEqual(sarima_keys, (*SARIMA_TEST_KEYS, *SARIMA_DECISION_AID_KEYS))
        self.assertEqual(test_keys_for_model("linear"), ())
        self.assertTrue(set(arima_keys).issubset(STATISTICAL_TESTS))
        self.assertTrue(set(sarima_keys).issubset(STATISTICAL_TESTS))
        self.assertTrue(any(STATISTICAL_TESTS[key].formal for key in arima_keys))
        self.assertTrue(any(not STATISTICAL_TESTS[key].formal for key in arima_keys))
        self.assertIn("ocsb", sarima_keys)
        self.assertIn("canova_hansen", sarima_keys)
        self.assertIn("seasonal_acf_rule", sarima_keys)

        handbook = copy_ready_test_handbook(
            sarima_keys, "SARIMA statistical decision handbook"
        )
        self.assertIn("reject H0", handbook)
        self.assertIn("fail to reject H0", handbook)
        self.assertIn("not a hypothesis test", handbook)
        for key in sarima_keys:
            self.assertIn(STATISTICAL_TESTS[key].name, handbook)

    def test_test_family_hypotheses_have_the_correct_orientation(self) -> None:
        self.assertIn("unit root", STATISTICAL_TESTS["adf"].null_hypothesis)
        self.assertIn("stationary", STATISTICAL_TESTS["kpss"].null_hypothesis)
        self.assertIn("seasonal unit root", STATISTICAL_TESTS["ocsb"].null_hypothesis)
        self.assertIn("stable", STATISTICAL_TESTS["canova_hansen"].null_hypothesis)
        self.assertIn("jointly zero", STATISTICAL_TESTS["ljung_box"].null_hypothesis)
        self.assertIn("normal", STATISTICAL_TESTS["jarque_bera"].null_hypothesis)
        self.assertIn("zero", STATISTICAL_TESTS["residual_mean_t"].null_hypothesis)
        self.assertIn("squared residuals", STATISTICAL_TESTS["arch_lm"].null_hypothesis)


if __name__ == "__main__":
    unittest.main()
