import unittest

import numpy as np
import pandas as pd

from chrono_stream.methods.statistical.croston import _run_filter
from chrono_stream.evaluation import evaluate_and_forecast, forecast_model


class CrostonFamilyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.values = np.array([0.0, 2.0, 0.0, 0.0, 4.0, 0.0])

    def test_croston_sba_and_tsb_match_the_declared_recursions(self) -> None:
        _fitted, croston, states = _run_filter(
            self.values, variant="croston", alpha=0.2, beta=0.3
        )
        self.assertAlmostEqual(croston, 2.4 / 2.2)
        self.assertAlmostEqual(states["demand_size_state"], 2.4)
        self.assertAlmostEqual(states["interval_state"], 2.2)

        _fitted, sba, _states = _run_filter(
            self.values, variant="sba", alpha=0.2, beta=0.3
        )
        self.assertAlmostEqual(sba, (1.0 - 0.2 / 2.0) * 2.4 / 2.2)

        fitted, tsb, states = _run_filter(
            self.values, variant="tsb", alpha=0.2, beta=0.3
        )
        self.assertAlmostEqual(fitted[5], 2.4 * 0.4715)
        self.assertAlmostEqual(states["occurrence_probability_state"], 0.33005)
        self.assertAlmostEqual(tsb, 2.4 * 0.33005)

    def test_all_zero_rule_and_negative_demand_validation_are_explicit(self) -> None:
        dates = pd.date_range("2020-01-01", periods=12, freq="MS")
        future = pd.date_range("2021-01-01", periods=3, freq="MS")
        result = forecast_model(
            "croston_family",
            np.zeros(12),
            3,
            {"variant": "sba", "automatic": True},
            dates=dates,
            forecast_dates=future,
        )
        np.testing.assert_array_equal(result["forecast"], np.zeros(3))
        np.testing.assert_array_equal(result["lower"], np.zeros(3))
        self.assertEqual(
            result["details"]["selection"], "Explicit all-zero demand rule"
        )

        negative = np.zeros(12)
        negative[5] = -1.0
        with self.assertRaisesRegex(ValueError, "nonnegative demand"):
            forecast_model(
                "croston_family",
                negative,
                3,
                {"variant": "croston", "automatic": False, "alpha": 0.1},
                dates=dates,
                forecast_dates=future,
            )

    def test_automatic_smoothing_cannot_see_the_outer_holdout(self) -> None:
        dates = pd.date_range("2018-01-01", periods=48, freq="MS")
        values = np.zeros(48)
        values[[1, 5, 9, 16, 20, 27, 32, 38, 42, 46]] = [
            3,
            4,
            2,
            5,
            3,
            6,
            4,
            3,
            5,
            4,
        ]
        original = pd.DataFrame({"Date": dates, "Demand": values})
        changed = original.copy()
        changed.loc[changed.index[-4:], "Demand"] = [75.0, 0.0, 60.0, 0.0]
        parameters = {"variant": "tsb", "automatic": True, "criterion": "RMSE"}

        first = evaluate_and_forecast(
            "croston_family",
            original,
            horizon=3,
            holdout=4,
            params=parameters,
            frequency="MS",
        )
        second = evaluate_and_forecast(
            "croston_family",
            changed,
            horizon=3,
            holdout=4,
            params=parameters,
            frequency="MS",
        )
        np.testing.assert_allclose(
            first["backtest"]["Predicted"], second["backtest"]["Predicted"]
        )
        for key in ("alpha", "beta"):
            self.assertEqual(
                first["backtest_model_details"][key],
                second["backtest_model_details"][key],
            )
        self.assertTrue(np.isnan(first["metrics"]["MAPE"]))
        self.assertTrue(np.isfinite(first["metrics"]["WAPE"]))


class MSTLETSForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dates = pd.date_range("2015-01-01", periods=84, freq="MS")
        time = np.arange(84, dtype=float)
        cls.values = (
            40.0
            + 0.08 * time
            + 2.0 * np.sin(2 * np.pi * time / 5)
            + 4.0 * np.cos(2 * np.pi * time / 12)
        )

    def test_mstl_ets_recombines_named_components_and_labels_intervals(self) -> None:
        frame = pd.DataFrame({"Date": self.dates, "Value": self.values})
        result = evaluate_and_forecast(
            "mstl_ets",
            frame,
            horizon=4,
            holdout=4,
            params={
                "seasonal_periods": "5, 12",
                "robust": True,
                "iterations": 2,
                "automatic_ets": False,
                "ets_trend": "Additive",
            },
            frequency="MS",
        )
        self.assertEqual(len(result["forecast"]), 4)
        self.assertTrue(np.isfinite(result["metrics"]["RMSE"]))
        details = result["model_details"]
        self.assertEqual(details["seasonal_periods"], [5, 12])
        self.assertEqual(len(details["seasonal_component_rules"]), 2)
        self.assertEqual(details["selected_ets_structure"], "ETS(A,A,N)")
        self.assertIn("repeat each final MSTL seasonal cycle", details["component_forecast_rule"])
        self.assertIn("Conditional ETS", details["interval_method"])
        self.assertIn("decomposition", details["interval_assumptions"])

    def test_automatic_downstream_ets_retains_candidate_evidence(self) -> None:
        future = pd.date_range("2022-01-01", periods=3, freq="MS")
        result = forecast_model(
            "mstl_ets",
            self.values,
            3,
            {
                "seasonal_periods": "5, 12",
                "robust": False,
                "iterations": 1,
                "automatic_ets": True,
                "criterion": "AICc",
            },
            dates=self.dates,
            forecast_dates=future,
        )
        details = result["details"]
        self.assertEqual(details["selection"], "Automatic downstream ETS search")
        self.assertEqual(len(details["candidate_results"]), 3)
        self.assertGreater(len(details["top_candidates"]), 0)
        self.assertTrue(details["selected_ets_structure"].startswith("ETS(A,"))

    def test_longest_period_requires_more_than_two_complete_cycles(self) -> None:
        values = self.values[:24]
        dates = self.dates[:24]
        future = pd.date_range(dates[-1] + pd.offsets.MonthBegin(), periods=2, freq="MS")
        with self.assertRaisesRegex(ValueError, "more than two repetitions"):
            forecast_model(
                "mstl_ets",
                values,
                2,
                {
                    "seasonal_periods": "4, 12",
                    "automatic_ets": False,
                    "ets_trend": "None",
                },
                dates=dates,
                forecast_dates=future,
            )


if __name__ == "__main__":
    unittest.main()
