import unittest

import numpy as np
import pandas as pd

from chrono_stream.evaluation import ACCURACY_METRIC_KEYS
from chrono_stream.features import (
    LagFeatureConfig,
    expanding_window_splits,
    supervised_rows,
)
from chrono_stream.evaluation import evaluate_and_forecast, forecast_model


class SupervisedFeatureContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dates = pd.date_range("2020-01-01", periods=30, freq="MS")
        self.values = np.arange(30, dtype=float)
        self.config = LagFeatureConfig(
            lookback=4,
            use_rolling=True,
            rolling_window=3,
            seasonal_lag=12,
            use_calendar=True,
        )

    def test_rows_use_only_values_before_each_target(self) -> None:
        original, targets, indices, names, _frequency = supervised_rows(
            self.values, self.dates, self.config
        )
        changed = self.values.copy()
        target_position = int(indices[5])
        changed[target_position] = 999_999.0
        revised, revised_targets, revised_indices, revised_names, _ = supervised_rows(
            changed, self.dates, self.config
        )
        np.testing.assert_allclose(original[5], revised[5])
        self.assertNotEqual(targets[5], revised_targets[5])
        self.assertFalse(np.allclose(original[6], revised[6]))
        np.testing.assert_array_equal(indices, revised_indices)
        self.assertEqual(names, revised_names)

    def test_expanding_splits_never_train_on_validation_or_future_rows(self) -> None:
        folds = expanding_window_splits(24, maximum_splits=4, minimum_train=8)
        self.assertGreaterEqual(len(folds), 2)
        previous_origin = 0
        for training, validation in folds:
            self.assertLess(training.max(), validation.min())
            self.assertEqual(len(np.intersect1d(training, validation)), 0)
            self.assertGreater(len(training), previous_origin)
            previous_origin = len(training)

    def test_irregular_dates_are_rejected(self) -> None:
        irregular = self.dates.delete(10)
        with self.assertRaisesRegex(ValueError, "regularly spaced"):
            supervised_rows(self.values[:-1], irregular, self.config)


class NextBatchForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dates = pd.date_range("2019-01-01", periods=60, freq="MS")
        time = np.arange(60, dtype=float)
        cls.values = (
            30.0
            + 0.15 * time
            + 2.5 * np.sin(2 * np.pi * time / 12)
            + 0.12 * np.cos(time)
        )
        cls.future = pd.date_range("2024-01-01", periods=3, freq="MS")

    def run_forecast(self, model_id: str, params: dict) -> dict:
        result = forecast_model(
            model_id,
            self.values,
            3,
            params,
            dates=self.dates,
            forecast_dates=self.future,
        )
        self.assertEqual(len(result["fitted"]), len(self.values))
        self.assertEqual(len(result["forecast"]), 3)
        self.assertTrue(np.isfinite(result["forecast"]).all())
        if result["details"].get("interval_available", True):
            self.assertTrue(np.isfinite(result["lower"]).all())
            self.assertTrue(np.isfinite(result["upper"]).all())
            self.assertTrue((result["lower"] <= result["upper"]).all())
        else:
            self.assertTrue(np.isnan(result["lower"]).all())
            self.assertTrue(np.isnan(result["upper"]).all())
        return result

    def test_all_ten_manual_paths_share_the_result_contract(self) -> None:
        cases = {
            "lagged_linear": {"automatic": False, "lookback": 6},
            "regularized_regression": {
                "automatic": False,
                "penalty": "Elastic Net",
                "alpha": 0.1,
                "l1_ratio": 0.5,
                "lookback": 6,
            },
            "cart": {
                "automatic": False,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "lookback": 6,
            },
            "random_forest": {
                "automatic": False,
                "n_estimators": 20,
                "max_depth": 5,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "lookback": 6,
            },
            "theta": {
                "seasonality": "No seasonal adjustment",
                "theta": 2.0,
                "use_mle": False,
            },
            "automatic_ets": {
                "automatic": False,
                "seasonal_period": 12,
                "error": "additive",
                "trend": "additive",
                "seasonal": "additive",
                "damped": False,
            },
            "tbats": {
                "automatic": False,
                "seasonal_periods": "12",
                "use_box_cox": False,
                "use_trend": True,
                "use_damped_trend": False,
                "use_arma_errors": False,
            },
            "support_vector_regression": {
                "automatic": False,
                "kernel": "rbf",
                "C": 2.0,
                "epsilon": 0.1,
                "gamma": "scale",
                "lookback": 6,
            },
            "knn_regression": {
                "automatic": False,
                "n_neighbors": 3,
                "weights": "distance",
                "p": 2,
                "lookback": 6,
            },
            "extra_trees": {
                "automatic": False,
                "n_estimators": 20,
                "max_depth": 5,
                "min_samples_leaf": 1,
                "max_features": 1.0,
                "lookback": 6,
            },
        }
        for model_id, params in cases.items():
            with self.subTest(model=model_id):
                result = self.run_forecast(model_id, params)
                self.assertIn("multi_step_strategy", result["details"])

    def test_automatic_paths_report_selection_evidence(self) -> None:
        cases = {
            "lagged_linear": {"automatic": True, "max_lookback": 6},
            "regularized_regression": {
                "automatic": True,
                "penalty": "Ridge",
                "lookback": 6,
            },
            "cart": {"automatic": True, "lookback": 6},
            "random_forest": {"automatic": True, "lookback": 6},
            "support_vector_regression": {"automatic": True, "lookback": 6},
            "knn_regression": {"automatic": True, "lookback": 6},
            "extra_trees": {"automatic": True, "lookback": 6},
        }
        for model_id, params in cases.items():
            with self.subTest(model=model_id):
                details = self.run_forecast(model_id, params)["details"]
                self.assertIn("Automatic", details["selection"])
                self.assertGreater(len(details["candidate_results"]), 0)
                self.assertGreater(len(details["top_candidates"]), 0)

        theta = self.run_forecast(
            "theta",
            {
                "seasonality": "Automatic test",
                "seasonal_period": 12,
                "decomposition": "auto",
                "theta": 2.0,
            },
        )
        self.assertEqual(theta["details"]["selection"], "Automatic test")

        ets = self.run_forecast(
            "automatic_ets",
            {
                "automatic": True,
                "seasonal_period": 12,
                "criterion": "AICc",
                "allow_multiplicative": False,
                "allow_damped": False,
            },
        )
        self.assertGreater(len(ets["details"]["candidate_results"]), 1)
        self.assertTrue(ets["details"]["selected_structure"].startswith("ETS("))
        self.assertTrue(
            all(
                "selection_exclusion_reason" in candidate
                for candidate in ets["details"]["candidate_results"]
            )
        )

        tbats = self.run_forecast(
            "tbats",
            {
                "automatic": True,
                "seasonal_periods": "12",
                "use_arma_errors": False,
            },
        )
        self.assertIn("Automatic", tbats["details"]["selection"])
        self.assertGreaterEqual(len(tbats["details"]["seasonal_harmonics"]), 1)

        trendless_tbats = self.run_forecast(
            "tbats",
            {
                "automatic": False,
                "seasonal_periods": "12",
                "use_box_cox": False,
                "use_trend": False,
                "use_damped_trend": False,
                "use_arma_errors": False,
            },
        )
        self.assertIsNone(trendless_tbats["details"]["damping_phi"])

    def test_seeded_tree_methods_are_reproducible(self) -> None:
        cases = {
            "cart": {
                "automatic": False,
                "max_depth": 4,
                "min_samples_leaf": 2,
                "lookback": 6,
            },
            "random_forest": {
                "automatic": False,
                "n_estimators": 30,
                "max_depth": 5,
                "min_samples_leaf": 1,
                "max_features": 0.7,
                "lookback": 6,
            },
            "extra_trees": {
                "automatic": False,
                "n_estimators": 30,
                "max_depth": 5,
                "min_samples_leaf": 1,
                "max_features": 0.7,
                "lookback": 6,
            },
        }
        for model_id, params in cases.items():
            with self.subTest(model=model_id):
                first = self.run_forecast(model_id, params)
                second = self.run_forecast(model_id, params)
                np.testing.assert_array_equal(first["forecast"], second["forecast"])
                self.assertEqual(first["details"]["random_seed"], 42)

    def test_native_and_approximate_intervals_are_labeled_honestly(self) -> None:
        for model_id, params in {
            "theta": {"seasonality": "No seasonal adjustment"},
            "automatic_ets": {
                "automatic": False,
                "seasonal_period": 12,
                "error": "additive",
                "trend": "additive",
                "seasonal": "additive",
            },
            "tbats": {
                "automatic": False,
                "seasonal_periods": "12",
                "use_box_cox": False,
                "use_trend": True,
                "use_damped_trend": False,
                "use_arma_errors": False,
            },
        }.items():
            with self.subTest(model=model_id):
                result = self.run_forecast(model_id, params)
                self.assertIn("Model-native", result["details"]["interval_method"])

        tree = self.run_forecast(
            "random_forest",
            {
                "automatic": False,
                "n_estimators": 20,
                "lookback": 6,
            },
        )
        self.assertIn("Descriptive", tree["details"]["interval_method"])
        self.assertIsNone(tree["details"]["interval_nominal_coverage"])

        exact_knn = self.run_forecast(
            "knn_regression",
            {
                "automatic": False,
                "n_neighbors": 3,
                "weights": "distance",
                "p": 2,
                "lookback": 6,
            },
        )
        self.assertFalse(exact_knn["details"]["interval_available"])
        self.assertEqual(exact_knn["details"]["interval_method"], "Unavailable")

    def test_all_ten_results_are_comparison_dashboard_compatible(self) -> None:
        frame = pd.DataFrame({"Date": self.dates, "Value": self.values})
        cases = {
            "lagged_linear": {"automatic": False, "lookback": 6},
            "regularized_regression": {
                "automatic": False,
                "penalty": "Ridge",
                "alpha": 1.0,
                "lookback": 6,
            },
            "cart": {"automatic": False, "max_depth": 4, "lookback": 6},
            "random_forest": {
                "automatic": False,
                "n_estimators": 20,
                "lookback": 6,
            },
            "theta": {"seasonality": "No seasonal adjustment"},
            "automatic_ets": {
                "automatic": False,
                "seasonal_period": 12,
                "error": "additive",
                "trend": "additive",
                "seasonal": "additive",
            },
            "tbats": {
                "automatic": False,
                "seasonal_periods": "12",
                "use_box_cox": False,
                "use_trend": True,
                "use_damped_trend": False,
                "use_arma_errors": False,
            },
            "support_vector_regression": {
                "automatic": False,
                "kernel": "linear",
                "C": 1.0,
                "epsilon": 0.1,
                "lookback": 6,
            },
            "knn_regression": {
                "automatic": False,
                "n_neighbors": 3,
                "lookback": 6,
            },
            "extra_trees": {
                "automatic": False,
                "n_estimators": 20,
                "lookback": 6,
            },
        }
        for model_id, params in cases.items():
            with self.subTest(model=model_id):
                result = evaluate_and_forecast(
                    model_id,
                    frame,
                    horizon=2,
                    holdout=3,
                    params=params,
                    frequency="MS",
                )
                self.assertEqual(result["model_id"], model_id)
                self.assertEqual(len(result["forecast"]), 2)
                self.assertEqual(len(result["backtest"]), 3)
                self.assertEqual(
                    tuple(result["metrics"]), ACCURACY_METRIC_KEYS
                )
                self.assertTrue(result["model_details"])
                self.assertTrue(result["backtest_model_details"])

    def test_scaling_and_automatic_tuning_cannot_see_the_outer_holdout(self) -> None:
        original = pd.DataFrame({"Date": self.dates, "Value": self.values})
        changed = original.copy()
        changed.loc[changed.index[-3:], "Value"] = [50_000.0, -40_000.0, 70_000.0]
        params = {
            "automatic": True,
            "penalty": "Elastic Net",
            "lookback": 6,
        }
        first = evaluate_and_forecast(
            "regularized_regression",
            original,
            horizon=2,
            holdout=3,
            params=params,
            frequency="MS",
        )
        second = evaluate_and_forecast(
            "regularized_regression",
            changed,
            horizon=2,
            holdout=3,
            params=params,
            frequency="MS",
        )
        np.testing.assert_allclose(
            first["backtest"]["Predicted"], second["backtest"]["Predicted"]
        )
        self.assertEqual(
            first["backtest_model_details"]["selected_alpha"],
            second["backtest_model_details"]["selected_alpha"],
        )
        self.assertEqual(
            first["backtest_model_details"]["selected_l1_ratio"],
            second["backtest_model_details"]["selected_l1_ratio"],
        )

    def test_svr_target_scaling_is_equivariant_to_large_response_units(self) -> None:
        parameters = {
            "automatic": False,
            "kernel": "rbf",
            "C": 2.0,
            "epsilon": 0.1,
            "gamma": "scale",
            "lookback": 6,
        }
        original = self.run_forecast("support_vector_regression", parameters)
        scale = 1_000_000.0
        scaled = forecast_model(
            "support_vector_regression",
            self.values * scale,
            3,
            parameters,
            dates=self.dates,
            forecast_dates=self.future,
        )
        np.testing.assert_allclose(
            scaled["forecast"] / scale,
            original["forecast"],
            rtol=1e-5,
            atol=1e-5,
        )
        self.assertEqual(
            scaled["details"]["epsilon_units"],
            "Training-target standard deviations",
        )

    def test_theta_uses_deseasonalized_variance_and_discloses_fitted_availability(self) -> None:
        dates = pd.date_range("2020-01-01", periods=24, freq="MS")
        time = np.arange(24, dtype=float)
        values = 100.0 + 40.0 * np.sin(2 * np.pi * time / 12)
        future = pd.date_range("2022-01-01", periods=2, freq="MS")
        result = forecast_model(
            "theta",
            values,
            2,
            {
                "seasonality": "Force seasonal adjustment",
                "seasonal_period": 12,
                "decomposition": "additive",
                "theta": 2.0,
                "use_mle": False,
            },
            dates=dates,
            forecast_dates=future,
        )
        self.assertIn(
            "deseasonalized series",
            result["details"]["innovation_variance_source"],
        )
        self.assertEqual(result["details"]["fitted_values_available"], 0)
        self.assertTrue(np.isnan(result["fitted"]).all())

    def test_dynamic_data_requirements_are_actionable(self) -> None:
        with self.assertRaisesRegex(ValueError, "too few supervised examples"):
            self.run_forecast(
                "lagged_linear", {"automatic": False, "lookback": 57}
            )
        with self.assertRaisesRegex(ValueError, "two repetitions"):
            self.run_forecast(
                "tbats",
                {
                    "automatic": False,
                    "seasonal_periods": "40",
                    "use_box_cox": False,
                    "use_trend": True,
                    "use_damped_trend": False,
                    "use_arma_errors": False,
                },
            )


if __name__ == "__main__":
    unittest.main()
