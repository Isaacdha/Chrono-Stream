import unittest

import numpy as np
import pandas as pd

from chrono_stream.evaluation import evaluate_and_forecast


class MachineLearningSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dates = pd.date_range("2020-01-01", periods=48, freq="MS")
        time = np.arange(48)
        values = 50 + 0.4 * time + 4 * np.sin(2 * np.pi * time / 12)
        cls.data = pd.DataFrame({"Month": dates, "Value": values})

    def assert_forecast(self, model_id: str, parameters: dict) -> dict:
        result = evaluate_and_forecast(
            model_id,
            self.data,
            horizon=3,
            holdout=4,
            params=parameters,
            frequency="MS",
        )
        self.assertEqual(len(result["forecast"]), 3)
        self.assertTrue(np.isfinite(result["forecast"]["Forecast"]).all())
        self.assertTrue(np.isfinite(result["metrics"]["RMSE"]))
        return result

    def test_prophet(self) -> None:
        self.assert_forecast(
            "prophet",
            {
                "seasonality_mode": "additive",
                "changepoint_prior_scale": 0.05,
                "yearly_seasonality": False,
                "weekly_seasonality": False,
            },
        )

    def test_xgboost(self) -> None:
        self.assert_forecast(
            "xgboost",
            {"lookback": 6, "n_estimators": 10, "max_depth": 2, "learning_rate": 0.1},
        )

    def test_lstm(self) -> None:
        self.assert_forecast(
            "lstm",
            {"lookback": 6, "units": 8, "epochs": 1, "batch_size": 8},
        )

    def test_cnn(self) -> None:
        self.assert_forecast(
            "cnn",
            {
                "lookback": 6,
                "filters": 8,
                "kernel_size": 3,
                "epochs": 1,
                "batch_size": 8,
            },
        )

    def test_nbeats(self) -> None:
        result = self.assert_forecast(
            "nbeats",
            {
                "lookback": 6,
                "blocks": 1,
                "hidden_layers": 1,
                "hidden_units": 8,
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
            },
        )
        details = result["model_details"]
        self.assertEqual(details["multi_step_strategy"], "Direct multi-output horizon")
        self.assertIn("N-BEATS", details["architecture"])
        self.assertIn("not the paper's constrained", details["interpretability_note"])

    def test_tcn(self) -> None:
        result = self.assert_forecast(
            "tcn",
            {
                "lookback": 8,
                "filters": 8,
                "kernel_size": 2,
                "dilation_levels": 2,
                "dropout": 0.0,
                "epochs": 1,
                "batch_size": 8,
                "learning_rate": 0.001,
            },
        )
        details = result["model_details"]
        self.assertEqual(details["multi_step_strategy"], "Recursive")
        self.assertEqual(details["dilations"], [1, 2])
        self.assertIn("causal", details["causality"])


if __name__ == "__main__":
    unittest.main()
