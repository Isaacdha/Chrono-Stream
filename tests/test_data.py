import unittest

import pandas as pd

from chrono_stream.data import future_dates, infer_frequency, prepare_time_series


class DataPreparationTests(unittest.TestCase):
    def test_prepare_time_series_cleans_duplicates_and_fills_gaps(self) -> None:
        raw = pd.DataFrame(
            {
                "when": [
                    "2024-01-01",
                    "2024-02-01",
                    "2024-02-01",
                    "2024-03-01",
                    "2024-04-01",
                    "2024-05-01",
                    "2024-07-01",
                    "2024-08-01",
                    "2024-09-01",
                    "2024-10-01",
                    "bad",
                ],
                "amount": [10, 20, 30, 30, 40, 50, 70, 80, 90, 100, 99],
            }
        )
        prepared, report = prepare_time_series(
            raw,
            "when",
            "amount",
            frequency="MS",
            regularize=True,
            missing_method="Interpolate",
        )

        self.assertEqual(report.invalid_rows_removed, 1)
        self.assertEqual(report.duplicate_timestamps_combined, 1)
        self.assertEqual(report.missing_periods_created, 1)
        self.assertEqual(len(prepared), 10)
        self.assertAlmostEqual(prepared.loc[1, "amount"], 25.0)
        self.assertAlmostEqual(prepared.loc[5, "amount"], 60.163934, places=5)

    def test_monthly_frequency_and_future_dates(self) -> None:
        dates = pd.date_range("2022-01-01", periods=18, freq="MS")
        self.assertEqual(infer_frequency(dates), "MS")
        self.assertEqual(infer_frequency(dates.delete(5)), "MS")
        expected = pd.date_range("2023-07-01", periods=3, freq="MS")
        pd.testing.assert_index_equal(future_dates(dates, 3), expected)

    def test_rejects_non_numeric_series(self) -> None:
        raw = pd.DataFrame(
            {"date": pd.date_range("2024-01-01", periods=8), "value": ["x"] * 8}
        )
        with self.assertRaisesRegex(ValueError, "No rows"):
            prepare_time_series(raw, "date", "value")

    def test_resampling_changes_calendar_anchor_without_losing_values(self) -> None:
        raw = pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=8, freq="MS"),
                "value": range(8),
            }
        )
        prepared, report = prepare_time_series(raw, "date", "value", frequency="ME")
        self.assertEqual(report.frequency, "ME")
        self.assertEqual(prepared["value"].tolist(), list(map(float, range(8))))
        self.assertTrue(pd.DatetimeIndex(prepared["date"]).is_month_end.all())

    def test_sum_resampling_preserves_empty_periods_for_missing_value_handling(self) -> None:
        raw = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2024-01-01",
                        "2024-02-01",
                        "2024-04-01",
                        "2024-05-01",
                        "2024-07-01",
                        "2024-08-01",
                        "2024-09-01",
                        "2024-10-01",
                    ]
                ),
                "value": [10, 20, 40, 50, 70, 80, 90, 100],
            }
        )
        prepared, report = prepare_time_series(
            raw,
            "date",
            "value",
            frequency="MS",
            regularize=True,
            missing_method="Interpolate",
            duplicate_method="Sum",
        )

        self.assertEqual(report.missing_periods_created, 2)
        self.assertNotIn(0.0, prepared["value"].tolist())
        self.assertGreater(prepared.loc[2, "value"], 20.0)
        self.assertLess(prepared.loc[2, "value"], 40.0)
        self.assertGreater(prepared.loc[5, "value"], 50.0)
        self.assertLess(prepared.loc[5, "value"], 70.0)

    def test_day_first_parsing(self) -> None:
        raw = pd.DataFrame(
            {
                "date": [f"{day:02d}/02/2024" for day in range(1, 9)],
                "value": range(8),
            }
        )
        prepared, _ = prepare_time_series(raw, "date", "value", day_first=True)
        self.assertEqual(prepared.loc[0, "date"], pd.Timestamp("2024-02-01"))


if __name__ == "__main__":
    unittest.main()
