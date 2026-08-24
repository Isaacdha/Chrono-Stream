import unittest

import numpy as np
import pandas as pd

from chrono_stream.models import (
    MODEL_NAMES,
    evaluate_and_forecast,
    forecast_model,
    regression_metrics,
)


class ForecastModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dates = pd.date_range("2017-01-01", periods=72, freq="MS")
        time = np.arange(72)
        values = 100 + 0.8 * time + 12 * np.sin(2 * np.pi * time / 12)
        cls.data = pd.DataFrame({"Month": dates, "Value": values})

    def assert_valid_result(self, result: dict) -> None:
        self.assertEqual(len(result["fitted"]), len(self.data))
        self.assertEqual(len(result["forecast"]), 6)
        self.assertEqual(len(result["backtest"]), 6)
        self.assertTrue(np.isfinite(result["forecast"]["Forecast"]).all())
        self.assertTrue(np.isfinite(result["metrics"]["RMSE"]))
        self.assertTrue(
            (result["forecast"]["Lower 95%"] <= result["forecast"]["Upper 95%"]).all()
        )

    def test_lightweight_models_share_the_result_contract(self) -> None:
        cases = {
            "moving_average": {"window": 6},
            "weighted_moving_average": {"window": 6, "weighting": "Linear"},
            "single_exponential_smoothing": {"alpha": None},
            "double_exponential_smoothing": {"damped": False},
            "triple_exponential_smoothing": {
                "seasonal_period": 12,
                "trend": "add",
                "seasonal": "add",
            },
            "arima": {"p": 1, "d": 1, "q": 1},
            "sarima": {
                "p": 1,
                "d": 1,
                "q": 0,
                "P": 0,
                "D": 1,
                "Q": 1,
                "seasonal_period": 12,
            },
            "x11": {"seasonal_period": 12, "robust": True},
            "linear": {},
            "quadratic": {},
            "exponential": {},
            "logarithmic": {},
        }
        for model_id, parameters in cases.items():
            with self.subTest(model=model_id):
                result = evaluate_and_forecast(
                    model_id,
                    self.data,
                    horizon=6,
                    holdout=6,
                    params=parameters,
                    frequency="MS",
                )
                self.assert_valid_result(result)

    def test_metrics_handle_zero_actual_values(self) -> None:
        metrics = regression_metrics([0, 2, 4], [1, 2, 5])
        self.assertTrue(np.isfinite(metrics["MAE"]))
        self.assertTrue(np.isfinite(metrics["MAPE"]))
        self.assertTrue(np.isfinite(metrics["sMAPE"]))

    def test_automatic_arima_selects_and_reports_orders(self) -> None:
        result = evaluate_and_forecast(
            "arima",
            self.data,
            horizon=4,
            holdout=6,
            params={
                "automatic": True,
                "max_p": 1,
                "max_d": 2,
                "max_q": 1,
                "differencing_test": "ADF",
                "criterion": "AIC",
            },
            frequency="MS",
        )
        details = result["model_details"]
        self.assertEqual(details["selection"], "Automatic")
        self.assertEqual(len(details["selected_order"]), 3)
        self.assertGreater(details["models_succeeded"], 0)
        self.assertIn("selected_order", result["backtest_model_details"])

    def test_automatic_sarima_selects_and_reports_orders(self) -> None:
        result = evaluate_and_forecast(
            "sarima",
            self.data,
            horizon=4,
            holdout=6,
            params={
                "automatic": True,
                "max_p": 1,
                "max_d": 1,
                "max_q": 1,
                "max_P": 1,
                "max_D": 1,
                "max_Q": 1,
                "seasonal_period": 12,
                "differencing_test": "ADF",
                "criterion": "AIC",
            },
            frequency="MS",
        )
        details = result["model_details"]
        self.assertEqual(details["selection"], "Automatic")
        self.assertEqual(len(details["selected_order"]), 3)
        self.assertEqual(len(details["selected_seasonal_order"]), 4)
        self.assertGreater(details["models_succeeded"], 0)

    def test_automatic_windows_and_manual_holt_coefficients(self) -> None:
        moving = evaluate_and_forecast(
            "moving_average",
            self.data,
            horizon=4,
            holdout=6,
            params={"automatic_window": True, "max_window": 18},
            frequency="MS",
        )
        self.assertEqual(moving["model_details"]["selection"], "Automatic")
        self.assertGreaterEqual(moving["model_details"]["selected_window"], 2)

        double = evaluate_and_forecast(
            "double_exponential_smoothing",
            self.data,
            horizon=4,
            holdout=6,
            params={"alpha": 0.4, "beta": 0.1, "damped": False},
            frequency="MS",
        )
        self.assertEqual(double["model_details"]["selection"], "Manual")
        self.assertAlmostEqual(double["model_details"]["smoothing_level"], 0.4)

        triple = evaluate_and_forecast(
            "triple_exponential_smoothing",
            self.data,
            horizon=4,
            holdout=6,
            params={
                "seasonal_period": 12,
                "trend": "add",
                "seasonal": "add",
                "alpha": 0.4,
                "beta": 0.1,
                "gamma": 0.2,
                "damped": False,
            },
            frequency="MS",
        )
        self.assertEqual(triple["model_details"]["selection"], "Manual")
        self.assertAlmostEqual(triple["model_details"]["smoothing_seasonal"], 0.2)

    def test_exponential_trend_uses_duan_smearing_retransformation(self) -> None:
        time = np.arange(1, 25, dtype=float)
        log_errors = np.tile(np.asarray([-0.45, 0.05, 0.40, 0.00]), 6)
        values = np.exp(1.5 + 0.035 * time + log_errors)
        dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
        future = pd.date_range(dates[-1] + pd.offsets.MonthBegin(), periods=3, freq="MS")

        result = forecast_model(
            "exponential",
            values,
            3,
            {},
            dates=dates,
            forecast_dates=future,
        )
        details = result["details"]
        self.assertEqual(
            details["retransformation"], "Duan nonparametric smearing estimate"
        )
        self.assertGreater(details["smearing_factor"], 1.0)
        future_time = np.arange(len(values) + 1, len(values) + 4, dtype=float)
        naive_median = np.exp(
            details["log_slope"] * future_time + details["log_intercept"]
        )
        np.testing.assert_allclose(
            result["forecast"], details["smearing_factor"] * naive_median
        )

    def test_stl_forecast_is_labeled_as_stl_not_official_x11(self) -> None:
        result = evaluate_and_forecast(
            "x11",
            self.data,
            horizon=4,
            holdout=6,
            params={"seasonal_period": 12, "robust": True},
            frequency="MS",
        )
        self.assertIn("STL", MODEL_NAMES["x11"])
        self.assertEqual(result["model_details"]["decomposition"], "STL")
        self.assertFalse(result["model_details"]["official_x11"])


if __name__ == "__main__":
    unittest.main()
