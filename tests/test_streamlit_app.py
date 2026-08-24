import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest


class StreamlitWorkflowTests(unittest.TestCase):
    APP_PATH = Path(__file__).resolve().parents[1] / "chrono_app.py"
    MODEL_PAGE_PATHS = [
        "method/Smoothing Based Methods/1_Moving Average.py",
        "method/Smoothing Based Methods/2_Weighted Moving Average.py",
        "method/Smoothing Based Methods/3_Single Exponential Smoothing.py",
        "method/Smoothing Based Methods/4_Double Exponential Smoothing.py",
        "method/Smoothing Based Methods/5_Triple Exponential Smoothing.py",
        "method/Statistical Models/1_ARIMA.py",
        "method/Statistical Models/2_SARIMA.py",
        "method/Statistical Models/4_X-11.py",
        "method/Machine Learning Models/1_Prophet.py",
        "method/Machine Learning Models/2_LSTM.py",
        "method/Machine Learning Models/3_CNN.py",
        "method/Machine Learning Models/4_XGBoost.py",
        "method/Deterministic Trend Projection/1_Linear.py",
        "method/Deterministic Trend Projection/2_Quadratic.py",
        "method/Deterministic Trend Projection/3_Exponential.py",
        "method/Deterministic Trend Projection/4_Logarithmic.py",
    ]
    AUTOMATIC_CONTROL_LABELS = {
        "method/Smoothing Based Methods/1_Moving Average.py": "Automatically find the optimal window size",
        "method/Smoothing Based Methods/2_Weighted Moving Average.py": "Automatically find the optimal window size",
        "method/Smoothing Based Methods/3_Single Exponential Smoothing.py": "Estimate smoothing level automatically",
        "method/Smoothing Based Methods/4_Double Exponential Smoothing.py": "Automatically find optimal alpha and beta",
        "method/Smoothing Based Methods/5_Triple Exponential Smoothing.py": "Automatically find optimal alpha, beta, and gamma",
        "method/Statistical Models/1_ARIMA.py": "Automatically select AR (p), differencing (d), and MA (q)",
        "method/Statistical Models/2_SARIMA.py": "Automatically select non-seasonal and seasonal orders",
    }

    @staticmethod
    def widget_with_label(widgets, label: str):
        return next(widget for widget in widgets if widget.label == label)

    def test_every_page_renders_without_data(self) -> None:
        paths = [
            "method/1_App Overview.py",
            "method/2_Data Input.py",
            "method/3_Data Exploration.py",
            "method/4_Result Comparison and Forecasting.py",
            *self.MODEL_PAGE_PATHS,
        ]
        app = AppTest.from_file(self.APP_PATH, default_timeout=30).run()
        for path in paths:
            with self.subTest(page=path):
                app.switch_page(path).run()
                self.assertFalse(app.exception)
                if path in self.MODEL_PAGE_PATHS:
                    self.assertEqual(len(app.get("popover")), 2)
                    self.assertGreaterEqual(len(app.code), 2)
                    tab_labels = {tab.label for tab in app.tabs}
                    for expected_label in (
                        "Method",
                        "Use",
                        "Method review",
                        "References",
                        "Files",
                    ):
                        self.assertIn(expected_label, tab_labels)
                    if path in {
                        "method/Statistical Models/1_ARIMA.py",
                        "method/Statistical Models/2_SARIMA.py",
                    }:
                        self.assertIn("Tests", tab_labels)
                        self.assertIn("Test reviews", tab_labels)

    def test_sample_to_linear_forecast_and_comparison(self) -> None:
        app = AppTest.from_file(self.APP_PATH, default_timeout=30).run()
        self.assertFalse(app.exception)

        app.switch_page("method/2_Data Input.py").run()
        app.radio[0].set_value("Sample Data 2 (CSV)").run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["filtered_df"]), 200)

        for path in ["method/3_Data Exploration.py", *self.MODEL_PAGE_PATHS]:
            with self.subTest(loaded_page=path):
                app.switch_page(path).run()
                self.assertFalse(app.exception)
                if expected_label := self.AUTOMATIC_CONTROL_LABELS.get(path):
                    self.assertTrue(app.toggle)
                    self.assertEqual(app.toggle[0].label, expected_label)
                    self.assertTrue(app.toggle[0].value)

        app.switch_page("method/Deterministic Trend Projection/1_Linear.py").run()
        self.assertFalse(app.exception)
        app.button[0].click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertIn("linear", app.session_state["model_results"])

        app.switch_page("method/4_Result Comparison and Forecasting.py").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "📊 Result Comparison and Forecasting")

    def test_strict_arima_workflow_renders_complete_diagnostics(self) -> None:
        generator = np.random.default_rng(123)
        innovations = generator.normal(size=320)
        values = np.zeros(320)
        for index in range(1, len(values)):
            values[index] = 0.72 * values[index - 1] + innovations[index]

        app = AppTest.from_file(self.APP_PATH, default_timeout=60)
        app.session_state["filtered_df"] = pd.DataFrame(
            {
                "Date": pd.date_range("2000-01-01", periods=320, freq="MS"),
                "Value": values,
            }
        )
        app.session_state["forecast_period"] = 6
        app.session_state["evaluation_period"] = 20
        app.session_state["seasonal_period"] = 12
        app.session_state["data_frequency"] = "MS"
        app.run()
        app.switch_page("method/Statistical Models/1_ARIMA.py").run()

        self.widget_with_label(
            app.toggle, "Automatically select AR (p), differencing (d), and MA (q)"
        ).set_value(False).run()
        self.widget_with_label(
            app.toggle, "Stabilize variance before differencing"
        ).set_value(False).run()
        self.widget_with_label(
            app.number_input, "Regular differencing order (d)"
        ).set_value(0).run()
        self.widget_with_label(app.number_input, "MA order (q)").set_value(0).run()
        self.widget_with_label(app.selectbox, "Deterministic term").set_value(
            "None"
        ).run()
        self.widget_with_label(app.button, "Fit, evaluate, and forecast").click().run(
            timeout=60
        )

        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertIn("arima", app.session_state["model_results"])
        details = app.session_state["model_results"]["arima"]["model_details"]
        self.assertTrue(details["selected_model_eligible"])
        self.assertEqual(details["selected_order"], [1, 0, 0])
        tab_labels = [tab.label for tab in app.tabs]
        for expected_label in (
            "Pipeline",
            "Candidate models",
            "Coefficients",
            "Residual diagnostics",
        ):
            self.assertIn(expected_label, tab_labels)


if __name__ == "__main__":
    unittest.main()
