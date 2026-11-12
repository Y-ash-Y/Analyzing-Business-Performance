from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from forecasting.data import (
    complete_scope, load_panel, missing_gaps, observed_totals, prepare_training,
)
from forecasting.evaluate import align_predictions, evaluation_windows, score
from forecasting.models import MODEL_NAMES, forecast_dates, forecast_models

ROOT = Path(__file__).resolve().parents[1]


class SourceDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = load_panel(ROOT)

    def test_gap_runs_and_coverage(self):
        panel = self.panel
        self.assertEqual(panel.shape, (143, 168))
        self.assertEqual(int(panel.isna().sum().sum()), 167)
        gaps = missing_gaps(panel).query("company == 'Seven Eleven' and Store == 3 and Dept == 6")
        self.assertEqual(
            list(zip(gaps["start"].dt.strftime("%Y-%m-%d"),
                     gaps["end"].dt.strftime("%Y-%m-%d"), gaps["weeks"])),
            [("2010-02-05", "2010-09-03", 31), ("2010-12-31", "2011-09-23", 39),
             ("2011-10-07", "2012-06-01", 35), ("2012-06-15", "2012-08-24", 11),
             ("2012-09-07", "2012-10-26", 8)],
        )
        self.assertEqual(complete_scope(panel).shape, (143, 164))
        full_holdout = observed_totals(panel).iloc[114:].dropna()
        self.assertEqual(list(full_holdout.index.strftime("%Y-%m-%d")),
                         ["2012-06-08", "2012-08-31"])

    def test_training_is_unchanged_when_future_sales_change(self):
        cutoff = pd.Timestamp("2012-04-06")
        original, original_fills = prepare_training(self.panel, cutoff)
        edited = self.panel.copy()
        edited.loc[edited.index > cutoff, :] = 1_000_000_000
        modified, modified_fills = prepare_training(edited, cutoff)
        pd.testing.assert_series_equal(original, modified)
        pd.testing.assert_frame_equal(original_fills, modified_fills)
        self.assertTrue(original_fills["Date"].le(cutoff).all())
        self.assertEqual(int(self.panel.isna().sum().sum()), 167)

        sparse = original_fills.query("company == 'Seven Eleven' and Store == 3 and Dept == 6")
        carry = sparse.loc[sparse["Date"].eq(cutoff)].iloc[0]
        self.assertEqual(carry["method"], "carry_forward")
        self.assertEqual(carry["imputed_sales"],
                         self.panel.at[pd.Timestamp("2011-09-30"), ("Seven Eleven", 3, 6)])
        first = sparse.iloc[0]
        peer_values = self.panel.loc[first["Date"], ("Seven Eleven", slice(None), 6)]
        self.assertEqual(first["method"], "same_week_peer_median")
        self.assertEqual(first["imputed_sales"], peer_values.median())

    def test_rolling_windows_end_before_holdout_and_expand_training(self):
        windows = list(evaluation_windows(self.panel.index))
        holdout = windows[-1]
        self.assertEqual(holdout[0], "holdout")
        self.assertEqual(holdout[1], pd.Timestamp("2012-04-06"))
        self.assertEqual(len(holdout[2]), 29)
        self.assertEqual([self.panel.index.get_loc(cutoff) + 1 for _, cutoff, _ in windows[:-1]],
                         [104, 106, 108, 110])
        for evaluation, cutoff, dates in windows[:-1]:
            self.assertEqual(evaluation, "rolling")
            self.assertTrue(dates.equals(forecast_dates(cutoff, 4)))
            self.assertLess(dates[-1], holdout[2][0])

    def test_all_models_return_native_friday_forecasts(self):
        cutoff = pd.Timestamp("2012-04-06")
        train, _ = prepare_training(complete_scope(self.panel), cutoff)
        forecasts, _ = forecast_models(train, 2)
        self.assertEqual(list(forecasts), MODEL_NAMES)
        expected = forecast_dates(cutoff, 2)
        for model, forecast in forecasts.items():
            with self.subTest(model=model):
                self.assertTrue(forecast.index.equals(expected))
                self.assertTrue(np.isfinite(forecast).all())


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range("2024-01-05", periods=4, freq="W-FRI")
        self.actual = pd.Series([10., 20., np.nan, 0.], index=self.dates)
        self.prediction = pd.Series([11., 18., 999., 3.], index=self.dates)

    def test_metrics_align_shuffled_dates_and_skip_missing_and_zero_as_applicable(self):
        rows = align_predictions(self.actual, self.prediction.iloc[::-1])
        metrics = score(rows)
        self.assertEqual(metrics["forecast_rows"], 4)
        self.assertEqual(metrics["scored_rows"], 3)
        self.assertEqual(metrics["mape_rows"], 2)
        self.assertAlmostEqual(metrics["mae"], 2)
        self.assertAlmostEqual(metrics["rmse"], np.sqrt(14 / 3))
        self.assertAlmostEqual(metrics["mape_pct"], 10)

    def test_rejects_invalid_forecast_dates_and_values(self):
        invalid = [
            self.prediction.set_axis(self.dates + pd.Timedelta(days=2)),
            self.prediction.set_axis(self.dates + pd.Timedelta(days=7)),
            self.prediction.set_axis([self.dates[0]] * len(self.dates)),
            self.prediction.iloc[:-1],
            self.prediction.astype("Float64").mask(self.prediction.eq(18)),
        ]
        for prediction in invalid:
            with self.subTest(dates=prediction.index), self.assertRaises(ValueError):
                align_predictions(self.actual, prediction)


if __name__ == "__main__":
    unittest.main()
