import importlib
import unittest

from chrono_stream.registry import METHOD_REGISTRY, METHOD_SPECS, MODEL_NAMES


class MethodRegistryTests(unittest.TestCase):
    def test_registry_is_the_single_complete_identity_catalog(self) -> None:
        self.assertEqual(len(METHOD_SPECS), len(METHOD_REGISTRY))
        self.assertEqual(set(METHOD_REGISTRY), set(MODEL_NAMES))
        self.assertEqual(len({spec.url_path for spec in METHOD_SPECS}), len(METHOD_SPECS))
        for spec in METHOD_SPECS:
            with self.subTest(model=spec.model_id):
                self.assertIs(METHOD_REGISTRY[spec.model_id], spec)
                self.assertEqual(MODEL_NAMES[spec.model_id], spec.display_name)
                self.assertTrue(spec.url_path)
                self.assertTrue(callable(spec.forecast))
                self.assertTrue(callable(spec.render_parameters))
                self.assertTrue(spec.multi_step_strategy)
                self.assertTrue(spec.interval_capability)

    def test_each_method_has_its_own_method_facing_module(self) -> None:
        forecast_modules = [spec.forecast.__module__ for spec in METHOD_SPECS]
        self.assertEqual(len(forecast_modules), len(set(forecast_modules)))
        for spec in METHOD_SPECS:
            with self.subTest(model=spec.model_id):
                module = importlib.import_module(spec.forecast.__module__)
                self.assertIs(module.SPEC, spec)
                self.assertTrue(spec.forecast.__module__.startswith("chrono_stream.methods."))

    def test_only_arima_and_sarima_share_a_forecasting_engine(self) -> None:
        from chrono_stream.methods.statistical import arima, box_jenkins, sarima

        self.assertIs(arima.box_jenkins_forecast, box_jenkins.forecast)
        self.assertIs(sarima.box_jenkins_forecast, box_jenkins.forecast)
        for spec in METHOD_SPECS:
            if spec.model_id not in {"arima", "sarima"}:
                self.assertNotEqual(spec.forecast.__module__, box_jenkins.__name__)

    def test_stl_uses_its_accurate_id_and_navigation_group(self) -> None:
        self.assertNotIn("x11", METHOD_REGISTRY)
        stl = METHOD_REGISTRY["stl"]
        self.assertEqual(
            stl.navigation_group, "Decomposition & Seasonal Adjustment"
        )
        self.assertEqual(
            stl.display_name, "STL Decomposition Forecast (X-11-inspired)"
        )


if __name__ == "__main__":
    unittest.main()
