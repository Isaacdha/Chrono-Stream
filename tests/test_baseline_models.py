import unittest

import numpy as np
import pandas as pd

from chrono_stream.evaluation import evaluate_and_forecast, forecast_model


class BaselineForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dates = pd.date_range("2021-01-01", periods=36, freq="MS")
        time = np.arange(36, dtype=float)
        cls.values = 20 + 0.5 * time + 3 * np.sin(2 * np.pi * time / 12)
        cls.frame = pd.DataFrame({"Month": cls.dates, "Value": cls.values})

    def direct_forecast(
        self, model_id: str, values: np.ndarray, steps: int, params: dict
    ) -> dict:
        observed_dates = self.dates[: len(values)]
        future_dates = pd.date_range(
            observed_dates[-1] + pd.offsets.MonthBegin(), periods=steps, freq="MS"
        )
        return forecast_model(
            model_id,
            values,
            steps,
            params,
            dates=observed_dates,
            forecast_dates=future_dates,
        )

    def test_naive_repeats_the_last_observation(self) -> None:
        result = self.direct_forecast("naive", self.values, 5, {})
        np.testing.assert_allclose(result["forecast"], self.values[-1])
        np.testing.assert_allclose(result["fitted"][1:], self.values[:-1])
        self.assertTrue(np.isnan(result["fitted"][0]))
        self.assertEqual(
            result["details"]["multi_step_strategy"], "Direct persistence"
        )
        half_widths = result["upper"] - result["forecast"]
        self.assertAlmostEqual(half_widths[1] / half_widths[0], np.sqrt(2.0))

    def test_seasonal_naive_repeats_matching_phases(self) -> None:
        result = self.direct_forecast(
            "seasonal_naive", self.values, 15, {"seasonal_period": 12}
        )
        expected = np.resize(self.values[-12:], 15)
        np.testing.assert_allclose(result["forecast"], expected)
        np.testing.assert_allclose(result["fitted"][12:], self.values[:-12])
        self.assertEqual(result["details"]["seasonal_period"], 12)
        self.assertAlmostEqual(
            result["upper"][12] - result["forecast"][12],
            np.sqrt(2.0) * (result["upper"][0] - result["forecast"][0]),
        )

    def test_seasonal_naive_validates_period_and_history(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            self.direct_forecast(
                "seasonal_naive", self.values, 2, {"seasonal_period": 1}
            )
        with self.assertRaisesRegex(ValueError, "two full seasons"):
            self.direct_forecast(
                "seasonal_naive", self.values[:20], 2, {"seasonal_period": 12}
            )

    def test_drift_extrapolates_the_average_endpoint_change(self) -> None:
        values = np.asarray([4.0, 9.0, 7.0, 13.0, 16.0])
        result = self.direct_forecast("drift", values, 3, {})
        drift = (values[-1] - values[0]) / (len(values) - 1)
        np.testing.assert_allclose(
            result["forecast"], values[-1] + drift * np.arange(1, 4)
        )
        np.testing.assert_allclose(result["fitted"][1:], values[:-1] + drift)
        self.assertTrue(np.isnan(result["fitted"][0]))
        self.assertAlmostEqual(result["details"]["drift_per_step"], drift)
        self.assertEqual(
            result["details"]["multi_step_strategy"],
            "Direct linear drift extrapolation",
        )
        self.assertTrue(np.all(np.diff(result["upper"] - result["forecast"]) > 0))

    def test_outer_holdout_cannot_change_baseline_predictions(self) -> None:
        changed = self.frame.copy()
        changed.iloc[-6:, 1] = changed.iloc[-6:, 1] + 10_000
        cases = {
            "naive": {},
            "seasonal_naive": {"seasonal_period": 12},
            "drift": {},
        }
        for model_id, params in cases.items():
            with self.subTest(model=model_id):
                original = evaluate_and_forecast(
                    model_id,
                    self.frame,
                    horizon=3,
                    holdout=6,
                    params=params,
                    frequency="MS",
                )
                modified = evaluate_and_forecast(
                    model_id,
                    changed,
                    horizon=3,
                    holdout=6,
                    params=params,
                    frequency="MS",
                )
                np.testing.assert_allclose(
                    original["backtest"]["Predicted"],
                    modified["backtest"]["Predicted"],
                )


if __name__ == "__main__":
    unittest.main()
