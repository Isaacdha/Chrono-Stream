import math
import unittest
from urllib.parse import urlparse

import numpy as np
import pandas as pd

from chrono_stream.evaluation import (
    ACCURACY_METRIC_KEYS,
    evaluate_and_forecast,
    regression_metrics,
)
from chrono_stream.metric_info import (
    METRIC_INFORMATION,
    copy_ready_metric_handbook,
    copy_ready_metric_note,
)
from chrono_stream.metric_literature_reviews import METRIC_LITERATURE_REVIEWS


class AccuracyMetricCalculationTests(unittest.TestCase):
    def test_all_metric_formulas_match_a_hand_calculation(self) -> None:
        metrics = regression_metrics(
            [2.0, 4.0],
            [1.0, 7.0],
            training_actual=[1.0, 2.0, 4.0],
            scale_period=1,
        )

        self.assertEqual(tuple(metrics), ACCURACY_METRIC_KEYS)
        self.assertAlmostEqual(metrics["MAE"], 2.0)
        self.assertAlmostEqual(metrics["RMSE"], math.sqrt(5.0))
        self.assertAlmostEqual(metrics["MASE"], 2.0 / 1.5)
        self.assertAlmostEqual(metrics["RMSSE"], math.sqrt(5.0 / 2.5))
        self.assertAlmostEqual(metrics["MAPE"], 62.5)
        self.assertAlmostEqual(
            metrics["sMAPE"],
            100.0 * ((2.0 / 3.0) + (6.0 / 11.0)) / 2.0,
        )
        self.assertAlmostEqual(metrics["WAPE"], 100.0 * 4.0 / 6.0)

    def test_scaled_metrics_use_the_requested_training_lag(self) -> None:
        metrics = regression_metrics(
            [7.0, 8.0],
            [5.0, 12.0],
            training_actual=[1.0, 10.0, 3.0, 14.0, 5.0],
            scale_period=2,
        )

        self.assertAlmostEqual(metrics["MASE"], 3.0 / (8.0 / 3.0))
        self.assertAlmostEqual(metrics["RMSSE"], math.sqrt(10.0 / 8.0))

    def test_zero_actual_semantics_are_explicit(self) -> None:
        metrics = regression_metrics(
            [0.0, 0.0, 4.0],
            [0.0, 2.0, 5.0],
            training_actual=[0.0, 1.0, 0.0, 2.0],
        )

        self.assertTrue(math.isnan(metrics["MAPE"]))
        self.assertAlmostEqual(metrics["WAPE"], 75.0)
        self.assertAlmostEqual(metrics["sMAPE"], (0.0 + 200.0 + 200.0 / 9.0) / 3.0)

        all_zero = regression_metrics(
            [0.0, 0.0],
            [0.0, 0.0],
            training_actual=[0.0, 1.0, 0.0],
        )
        self.assertTrue(math.isnan(all_zero["MAPE"]))
        self.assertTrue(math.isnan(all_zero["WAPE"]))
        self.assertEqual(all_zero["sMAPE"], 0.0)

    def test_unavailable_scaled_denominators_return_nan(self) -> None:
        without_training = regression_metrics([2.0], [3.0])
        self.assertTrue(math.isnan(without_training["MASE"]))
        self.assertTrue(math.isnan(without_training["RMSSE"]))

        constant_training = regression_metrics(
            [2.0],
            [3.0],
            training_actual=[5.0, 5.0, 5.0],
        )
        self.assertTrue(math.isnan(constant_training["MASE"]))
        self.assertTrue(math.isnan(constant_training["RMSSE"]))

        short_training = regression_metrics(
            [2.0],
            [3.0],
            training_actual=[1.0, 2.0],
            scale_period=2,
        )
        self.assertTrue(math.isnan(short_training["MASE"]))
        self.assertTrue(math.isnan(short_training["RMSSE"]))

    def test_metric_inputs_are_validated(self) -> None:
        with self.assertRaisesRegex(ValueError, "same length"):
            regression_metrics([1.0, 2.0], [1.0])
        for invalid_period in (0, -1, 1.5, True):
            with self.subTest(scale_period=invalid_period):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    regression_metrics(
                        [1.0],
                        [1.0],
                        training_actual=[1.0, 2.0],
                        scale_period=invalid_period,
                    )
        with self.assertRaisesRegex(ValueError, "training values"):
            regression_metrics(
                [1.0],
                [1.0],
                training_actual=[1.0, np.nan],
            )

    def test_evaluation_records_a_leakage_safe_scale_contract(self) -> None:
        values = np.asarray([1, 2, 4, 7, 11, 16, 22, 29, 37, 46], dtype=float)
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=len(values), freq="MS"),
                "Value": values,
            }
        )
        result = evaluate_and_forecast(
            "naive",
            frame,
            horizon=2,
            holdout=2,
            metric_scale_period=2,
            frequency="MS",
        )

        train = values[:-2]
        actual = values[-2:]
        predicted = result["backtest"]["Predicted"].to_numpy()
        expected = regression_metrics(
            actual,
            predicted,
            training_actual=train,
            scale_period=2,
        )
        self.assertEqual(result["metrics"], expected)
        self.assertEqual(result["metric_context"]["scale_period"], 2)
        self.assertIn("Pre-holdout", result["metric_context"]["scale_source"])
        self.assertIn("lag 2", result["metric_context"]["scale_benchmark"])


class AccuracyMetricInformationTests(unittest.TestCase):
    def test_every_reported_metric_has_practical_and_scholarly_material(self) -> None:
        self.assertEqual(set(METRIC_INFORMATION), set(ACCURACY_METRIC_KEYS))
        self.assertEqual(set(METRIC_LITERATURE_REVIEWS), set(ACCURACY_METRIC_KEYS))

        for metric_key in ACCURACY_METRIC_KEYS:
            with self.subTest(metric=metric_key):
                information = METRIC_INFORMATION[metric_key]
                self.assertIn(metric_key, information.display_name)
                self.assertGreater(len(information.formula), 20)
                self.assertGreater(len(information.how_it_works), 140)
                self.assertGreater(len(information.chrono_stream), 140)
                self.assertGreater(len(information.when_to_use), 100)
                self.assertGreater(len(information.limitations), 140)
                self.assertIn("(", information.citation_ready)
                self.assertGreaterEqual(len(information.references), 2)

                review = METRIC_LITERATURE_REVIEWS[metric_key]
                self.assertGreater(len(review), 1_200)
                self.assertGreaterEqual(review.count("\n\n"), 2)

                urls = [reference.url for reference in information.references]
                self.assertEqual(len(urls), len(set(urls)))
                for reference in information.references:
                    parsed = urlparse(reference.url)
                    self.assertEqual(parsed.scheme, "https")
                    self.assertTrue(parsed.netloc)
                    self.assertGreater(len(reference.apa), 50)
                    self.assertGreater(len(reference.contribution), 50)
                    self.assertNotIn("wikipedia.org", parsed.netloc.lower())
                    self.assertNotIn("researchgate.net", parsed.netloc.lower())

                note = copy_ready_metric_note(metric_key)
                self.assertIn(information.display_name, note)
                self.assertIn("Formula", note)
                self.assertIn("Literature review", note)
                self.assertIn(review, note)
                self.assertIn("References (APA 7)", note)
                for reference in information.references:
                    self.assertIn(reference.apa, note)

    def test_mape_documents_zero_behavior_and_scaled_metrics_document_training(self) -> None:
        mape = METRIC_INFORMATION["MAPE"]
        self.assertIn("N/A", mape.chrono_stream)
        self.assertIn("does not delete", mape.chrono_stream)
        self.assertIn("intermittent demand", mape.limitations)

        for metric_key in ("MASE", "RMSSE"):
            information = METRIC_INFORMATION[metric_key]
            self.assertIn("pre-holdout training", information.chrono_stream)
            self.assertIn("N/A", information.chrono_stream)

    def test_complete_handbook_contains_every_metric_and_review(self) -> None:
        handbook = copy_ready_metric_handbook()
        self.assertIn("training observations", handbook)
        for metric_key in ACCURACY_METRIC_KEYS:
            self.assertIn(METRIC_INFORMATION[metric_key].display_name, handbook)
            self.assertIn(METRIC_LITERATURE_REVIEWS[metric_key], handbook)


if __name__ == "__main__":
    unittest.main()
