import unittest

import numpy as np
import pandas as pd

from chrono_stream.arima_pipeline import (
    NoEligibleModelError,
    VarianceTransformer,
    fit_arima_pipeline,
)
from chrono_stream.models import evaluate_and_forecast


class VarianceTransformationTests(unittest.TestCase):
    def test_supported_transformations_round_trip(self) -> None:
        cases = [
            (
                VarianceTransformer(requested_method="Box-Cox", requested_lambda=0.25),
                np.linspace(1.0, 25.0, 80),
            ),
            (
                VarianceTransformer(
                    requested_method="Yeo-Johnson", requested_lambda=0.40
                ),
                np.linspace(-8.0, 20.0, 80),
            ),
            (
                VarianceTransformer(requested_method="Log", allow_shift=True),
                np.linspace(-2.0, 20.0, 80),
            ),
            (
                VarianceTransformer(requested_method="Square root", allow_shift=True),
                np.linspace(-2.0, 20.0, 80),
            ),
        ]
        for transformer, values in cases:
            with self.subTest(method=transformer.requested_method):
                transformed = transformer.fit_transform(values)
                restored = transformer.inverse(transformed)
                np.testing.assert_allclose(restored, values, rtol=1e-10, atol=1e-10)

    def test_box_cox_rejects_nonpositive_data_without_shift(self) -> None:
        transformer = VarianceTransformer(requested_method="Box-Cox")
        with self.assertRaisesRegex(ValueError, "requires positive values"):
            transformer.fit_transform(np.asarray([-1.0, 0.0, 2.0, 4.0]))


class BoxJenkinsPipelineTests(unittest.TestCase):
    @staticmethod
    def ar_process(phi: float, observations: int, seed: int) -> np.ndarray:
        generator = np.random.default_rng(seed)
        innovations = generator.normal(size=observations)
        values = np.zeros(observations)
        for index in range(1, observations):
            values[index] = phi * values[index - 1] + innovations[index]
        return values

    def test_strict_pipeline_accepts_a_well_specified_ar_process(self) -> None:
        values = self.ar_process(0.72, 320, 123)
        result = fit_arima_pipeline(
            values,
            6,
            {
                "automatic": False,
                "search_strategy": "Manual order",
                "p": 1,
                "d": 0,
                "q": 0,
                "difference_mode": "Manual",
                "transformation": "None",
                "diagnostic_policy": "Strict Box-Jenkins",
                "normality_test": "Jarque-Bera",
                "white_noise_test": "Ljung-Box",
                "trend": "None",
                "acf_lags": 30,
                "allow_near_match": False,
            },
            seasonal=False,
        )
        details = result["details"]
        self.assertEqual(details["selected_order"], [1, 0, 0])
        self.assertTrue(details["selected_model_eligible"])
        self.assertTrue(all(details["diagnostic_outcomes"].values()))
        self.assertTrue(details["diagnostic_requirements"]["residual_mean"])
        self.assertTrue(details["diagnostic_outcomes"]["residual_mean"])
        self.assertGreater(len(details["white_noise"]["lags"]), 1)
        self.assertIn("f_statistic", details["heteroskedasticity"])
        self.assertIn("f_p_value", details["heteroskedasticity"])
        self.assertTrue(np.isfinite(result["forecast"]).all())

    def test_strict_pipeline_rejects_autocorrelated_residuals(self) -> None:
        values = self.ar_process(0.90, 320, 123)
        parameters = {
            "automatic": False,
            "search_strategy": "Manual order",
            "p": 0,
            "d": 0,
            "q": 0,
            "difference_mode": "Manual",
            "transformation": "None",
            "diagnostic_policy": "Strict Box-Jenkins",
            "normality_test": "Jarque-Bera",
            "white_noise_test": "Ljung-Box",
            "trend": "None",
            "acf_lags": 30,
            "allow_near_match": False,
        }
        with self.assertRaises(NoEligibleModelError) as raised:
            fit_arima_pipeline(
                values,
                6,
                parameters,
                seasonal=False,
            )
        candidate = raised.exception.details["candidate_results"][0]
        self.assertFalse(candidate["white_noise"])
        self.assertIn("white noise", candidate["failure_reasons"])

        parameters["allow_near_match"] = True
        overridden = fit_arima_pipeline(values, 6, parameters, seasonal=False)
        self.assertTrue(overridden["details"]["selection_override"])
        self.assertFalse(overridden["details"]["selected_model_eligible"])

    def test_auto_transformation_is_inverted_and_does_not_leak_holdout(self) -> None:
        generator = np.random.default_rng(4)
        time = np.arange(120)
        values = np.exp(2.0 + 0.015 * time + 0.15 * generator.normal(size=120))
        frame = pd.DataFrame(
            {
                "Date": pd.date_range("2015-01-01", periods=120, freq="MS"),
                "Value": values,
            }
        )
        parameters = {
            "automatic": True,
            "search_strategy": "Guided ACF/PACF",
            "max_p": 2,
            "max_q": 2,
            "max_order": 4,
            "difference_mode": "Automatic",
            "max_d": 2,
            "differencing_test": "ADF + KPSS consensus",
            "transformation": "Auto",
            "diagnostic_policy": "Advisory",
            "criterion": "AICc",
            "bias_adjust": True,
            "inverse_simulations": 500,
            "acf_lags": 30,
        }
        result = evaluate_and_forecast(
            "arima",
            frame,
            horizon=6,
            holdout=12,
            params=parameters,
            frequency="MS",
        )
        backtest_transformer = VarianceTransformer(requested_method="Auto")
        backtest_transformer.fit_transform(values[:-12])
        backtest_transformation = result["backtest_model_details"]["transformation"]
        self.assertEqual(backtest_transformation["applied_method"], "Box-Cox")
        self.assertAlmostEqual(
            backtest_transformation["lambda"], backtest_transformer.lmbda
        )
        self.assertTrue(
            result["model_details"]["inverse_transformation"]["bias_adjusted"]
        )
        self.assertTrue(np.isfinite(result["forecast"]["Forecast"]).all())
        self.assertGreater(result["forecast"]["Forecast"].median(), 1.0)

    def test_ocsb_selects_seasonal_differencing_and_uses_the_period(self) -> None:
        generator = np.random.default_rng(11)
        values = np.zeros(120)
        values[:12] = 50 + generator.normal(size=12)
        for index in range(12, len(values)):
            values[index] = values[index - 12] + generator.normal(scale=0.6)
        result = fit_arima_pipeline(
            values,
            6,
            {
                "automatic": False,
                "search_strategy": "Manual order",
                "p": 0,
                "q": 0,
                "P": 0,
                "Q": 0,
                "difference_mode": "Manual",
                "d": 0,
                "seasonal_difference_mode": "Automatic",
                "seasonal_differencing_test": "OCSB",
                "max_D": 1,
                "seasonal_period": 12,
                "transformation": "None",
                "diagnostic_policy": "Advisory",
                "trend": "None",
                "acf_lags": 30,
            },
            seasonal=True,
        )
        details = result["details"]
        self.assertEqual(details["selected_D"], 1)
        self.assertEqual(details["selected_seasonal_order"], [0, 1, 0, 12])
        self.assertEqual(details["seasonal_stationarity_history"][0]["method"], "OCSB")

    def test_manual_candidate_lists_can_use_inner_rolling_validation(self) -> None:
        values = self.ar_process(0.72, 140, 123)
        result = fit_arima_pipeline(
            values,
            4,
            {
                "automatic": False,
                "search_strategy": "Manual candidate list",
                "manual_p_values": "0,1,2",
                "manual_q_values": "0",
                "max_order": 3,
                "difference_mode": "Manual",
                "d": 0,
                "transformation": "None",
                "diagnostic_policy": "Advisory",
                "criterion": "CV RMSE",
                "cv_folds": 3,
                "cv_horizon": 2,
                "trend": "None",
                "acf_lags": 20,
            },
            seasonal=False,
        )
        details = result["details"]
        self.assertEqual(details["selection"], "Manual")
        self.assertEqual(details["criterion"], "CV RMSE")
        self.assertTrue(np.isfinite(details["criterion_value"]))
        self.assertEqual(details["selected_order"], [1, 0, 0])
        self.assertTrue(
            all(
                candidate["CV folds completed"] == 3
                for candidate in details["candidate_results"]
            )
        )


if __name__ == "__main__":
    unittest.main()
