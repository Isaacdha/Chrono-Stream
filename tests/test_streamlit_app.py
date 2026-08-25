import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from chrono_stream.evaluation import ACCURACY_METRIC_KEYS
from chrono_stream.registry import METHOD_SPECS


def render_registered_method(model_id: str) -> None:
    """Standalone callable used by Streamlit's page-testing harness."""
    from chrono_stream.ui import render_model_page

    render_model_page(model_id)


class StreamlitWorkflowTests(unittest.TestCase):
    APP_PATH = Path(__file__).resolve().parents[1] / "chrono_app.py"
    WORKFLOW_PAGE_PATHS = {
        "overview": "chrono_stream/page_overview.py",
        "data_input": "chrono_stream/page_data_input.py",
        "exploration": "chrono_stream/page_exploration.py",
        "comparison": "chrono_stream/page_comparison.py",
    }
    AUTOMATIC_CONTROL_LABELS = {
        "moving_average": "Automatically find the optimal window size",
        "weighted_moving_average": "Automatically find the optimal window size",
        "single_exponential_smoothing": "Estimate smoothing level automatically",
        "double_exponential_smoothing": "Automatically find optimal alpha and beta",
        "triple_exponential_smoothing": "Automatically find optimal alpha, beta, and gamma",
        "automatic_ets": "Automatically select the ETS structure",
        "arima": "Automatically select AR (p), differencing (d), and MA (q)",
        "sarima": "Automatically select non-seasonal and seasonal orders",
        "tbats": "Automatically select TBATS components by AIC",
        "croston_family": "Automatically select smoothing by causal one-step error",
        "mstl_ets": "Automatically select the downstream ETS trend",
        "lagged_linear": "Automatically select the lag count with expanding-window CV",
        "regularized_regression": "Automatically tune regularization with expanding-window CV",
        "cart": "Automatically tune tree complexity with expanding-window CV",
        "random_forest": "Automatically tune forest complexity with expanding-window CV",
        "support_vector_regression": "Automatically tune SVR with expanding-window CV",
        "knn_regression": "Automatically tune neighbors with expanding-window CV",
        "extra_trees": "Automatically tune ensemble complexity with expanding-window CV",
    }

    SESSION_KEYS = (
        "filtered_df",
        "data_frequency",
        "forecast_period",
        "evaluation_period",
        "seasonal_period",
        "metric_scale_period",
        "source_name",
        "data_signature",
        "configuration_signature",
        "model_results",
    )

    @staticmethod
    def widget_with_label(widgets, label: str):
        return next(widget for widget in widgets if widget.label == label)

    @classmethod
    def copy_session_state(cls, source: AppTest, target: AppTest) -> None:
        for key in cls.SESSION_KEYS:
            try:
                target.session_state[key] = source.session_state[key]
            except KeyError:
                continue

    @classmethod
    def method_app(
        cls, model_id: str, *, source: AppTest | None = None, timeout: float = 30
    ) -> AppTest:
        app = AppTest.from_function(
            render_registered_method,
            args=(model_id,),
            default_timeout=timeout,
        )
        if source is not None:
            cls.copy_session_state(source, app)
        return app.run()

    def test_every_page_renders_without_data(self) -> None:
        app = AppTest.from_file(self.APP_PATH, default_timeout=30).run()
        for name, path in self.WORKFLOW_PAGE_PATHS.items():
            with self.subTest(page=path):
                app.switch_page(path).run()
                self.assertFalse(app.exception)
                if name == "comparison":
                    self.assertEqual(len(app.get("popover")), 2)
                    tab_labels = {tab.label for tab in app.tabs}
                    for expected_label in (
                        "Original scale",
                        "Percentage",
                        "Benchmark scaled",
                        "Metric reviews",
                        "References",
                        "Files",
                        *ACCURACY_METRIC_KEYS,
                    ):
                        self.assertIn(expected_label, tab_labels)

        for spec in METHOD_SPECS:
            with self.subTest(model=spec.model_id):
                method_app = self.method_app(spec.model_id)
                self.assertFalse(method_app.exception)
                self.assertEqual(len(method_app.get("popover")), 2)
                self.assertGreaterEqual(len(method_app.code), 2)
                tab_labels = {tab.label for tab in method_app.tabs}
                for expected_label in (
                    "Method",
                    "Use",
                    "Method review",
                    "References",
                    "Files",
                ):
                    self.assertIn(expected_label, tab_labels)
                if spec.model_id in {"arima", "sarima"}:
                    self.assertIn("Tests", tab_labels)
                    self.assertIn("Test reviews", tab_labels)

    def test_reference_popover_css_is_scoped_and_targets_the_real_button(self) -> None:
        source = (
            self.APP_PATH.parent / "chrono_stream" / "ui.py"
        ).read_text(encoding="utf-8")
        selector = (
            '.st-key-method_reference_actions [data-testid="stPopoverButton"]'
        )
        self.assertIn(selector, source)
        self.assertIn("min-width: 3.5rem !important", source)
        self.assertNotIn('\n        [data-testid="stPopoverButton"] {', source)
        metric_selector = (
            '.st-key-metric_reference_actions [data-testid="stPopoverButton"]'
        )
        self.assertIn(metric_selector, source)
        self.assertIn("min-width: 6.5rem !important", source)

    def test_sample_to_linear_forecast_and_comparison(self) -> None:
        app = AppTest.from_file(self.APP_PATH, default_timeout=30).run()
        self.assertFalse(app.exception)

        app.switch_page(self.WORKFLOW_PAGE_PATHS["data_input"]).run()
        app.radio[0].set_value("Sample Data 2 (CSV)").run()
        self.assertFalse(app.exception)
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state["filtered_df"]), 200)

        app.switch_page(self.WORKFLOW_PAGE_PATHS["exploration"]).run()
        self.assertFalse(app.exception)
        for spec in METHOD_SPECS:
            with self.subTest(loaded_model=spec.model_id):
                method_app = self.method_app(spec.model_id, source=app)
                self.assertFalse(method_app.exception)
                if expected_label := self.AUTOMATIC_CONTROL_LABELS.get(spec.model_id):
                    automatic_control = self.widget_with_label(
                        method_app.toggle, expected_label
                    )
                    self.assertTrue(automatic_control.value)

        linear_app = self.method_app("linear", source=app)
        linear_app.button[0].click().run(timeout=30)
        self.assertFalse(linear_app.exception)
        self.assertIn("linear", linear_app.session_state["model_results"])
        result = linear_app.session_state["model_results"]["linear"]
        self.assertEqual(tuple(result["metrics"]), ACCURACY_METRIC_KEYS)
        self.assertEqual(
            result["metric_context"]["scale_period"],
            app.session_state["metric_scale_period"],
        )

        app.session_state["model_results"] = linear_app.session_state["model_results"]
        app.switch_page(self.WORKFLOW_PAGE_PATHS["comparison"]).run()
        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "📊 Result Comparison and Forecasting")
        self.assertEqual(len(app.get("popover")), 2)

    def test_strict_arima_workflow_renders_complete_diagnostics(self) -> None:
        generator = np.random.default_rng(123)
        innovations = generator.normal(size=320)
        values = np.zeros(320)
        for index in range(1, len(values)):
            values[index] = 0.72 * values[index - 1] + innovations[index]

        app = AppTest.from_function(
            render_registered_method,
            args=("arima",),
            default_timeout=60,
        )
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

    def test_unbounded_box_jenkins_controls_remove_caps_and_warn(self) -> None:
        app = AppTest.from_function(
            render_registered_method,
            args=("sarima",),
            default_timeout=30,
        )
        app.session_state["filtered_df"] = pd.DataFrame(
            {
                "Date": pd.date_range("2010-01-01", periods=120, freq="MS"),
                "Value": np.linspace(10.0, 30.0, 120),
            }
        )
        app.session_state["forecast_period"] = 6
        app.session_state["evaluation_period"] = 12
        app.session_state["seasonal_period"] = 12
        app.session_state["data_frequency"] = "MS"
        app.run()
        checkbox = self.widget_with_label(
            app.checkbox, "Allow unbounded differencing and AR/MA orders"
        )
        self.assertFalse(checkbox.value)
        self.assertIn("very slow", checkbox.help.lower())

        bounded_maxima = {
            "Maximum seasonal differencing (D)": 2,
            "Maximum regular differencing (d)": 2,
            "Maximum AR order (p)": 6,
            "Maximum MA order (q)": 6,
            "Maximum seasonal AR order (P)": 3,
            "Maximum seasonal MA order (Q)": 3,
            "Maximum total AR and MA order": 12,
        }
        for label, maximum in bounded_maxima.items():
            self.assertEqual(
                self.widget_with_label(app.number_input, label).max,
                maximum,
            )

        checkbox.set_value(True).run()
        for label in bounded_maxima:
            self.assertIsNone(self.widget_with_label(app.number_input, label).max)

        extended_values = {
            "Maximum seasonal differencing (D)": 3,
            "Maximum regular differencing (d)": 3,
            "Maximum AR order (p)": 7,
            "Maximum MA order (q)": 7,
            "Maximum seasonal AR order (P)": 4,
            "Maximum seasonal MA order (Q)": 4,
            "Maximum total AR and MA order": 25,
        }
        for label, value in extended_values.items():
            self.widget_with_label(app.number_input, label).set_value(value)
        app.run()

        self.assertFalse(app.exception)
        self.assertTrue(
            any(
                "tentative candidate models" in warning.value
                and "1,600" in warning.value
                for warning in app.warning
            )
        )

        self.widget_with_label(
            app.toggle, "Automatically select non-seasonal and seasonal orders"
        ).set_value(False).run()
        self.widget_with_label(
            app.selectbox, "Seasonal differencing (D)"
        ).set_value("Manual")
        self.widget_with_label(app.selectbox, "Regular differencing (d)").set_value(
            "Manual"
        )
        app.run()
        for label in (
            "Seasonal differencing order (D)",
            "Regular differencing order (d)",
            "AR order (p)",
            "MA order (q)",
            "Seasonal AR order (P)",
            "Seasonal MA order (Q)",
        ):
            self.assertIsNone(self.widget_with_label(app.number_input, label).max)


if __name__ == "__main__":
    unittest.main()
