import unittest

import numpy as np
import pandas as pd

from chrono_stream.models import evaluate_and_forecast


class MachineLearningSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dates = pd.date_range("2020-01-01", periods=48, freq="MS")
        time = np.arange(48)
        values = 50 + 0.4 * time + 4 * np.sin(2 * np.pi * time / 12)
        cls.data = pd.DataFrame({"Month": dates, "Value": values})

    def assert_forecast(self, model_id: str, parameters: dict) -> None:
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


if __name__ == "__main__":
    unittest.main()
