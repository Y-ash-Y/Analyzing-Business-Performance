"""Run the locked holdout, rolling-origin benchmark, and full-panel sensitivity."""

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from .data import COMPANIES, FREQUENCY, complete_scope, load_panel, missing_gaps
from .data import observed_totals, prepare_training
from .models import SEED, forecast_models, prophet_forecast

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "complete_164"
FULL = "full_168"
ROLLING_TRAIN_WEEKS = (104, 106, 108, 110)
ROLLING_HORIZON = 4
FUTURE_HORIZON = 26


def align_predictions(actual: pd.Series, forecast: pd.Series) -> pd.DataFrame:
    """Require exact Friday coverage, then join by date with no positional pairing."""
    for series in (actual, forecast):
        index = series.index
        if not isinstance(index, pd.DatetimeIndex) or index.tz is not None:
            raise ValueError("Evaluation requires timezone-naive date indexes")
        if not index.is_unique or index.hasnans or (index.dayofweek != 4).any():
            raise ValueError("Evaluation requires unique Friday dates")
        if not index.equals(index.normalize()):
            raise ValueError("Evaluation dates must be normalized to midnight")
    if len(actual) == 0 or not actual.index.sort_values().equals(forecast.index.sort_values()):
        raise ValueError("Forecast and actual calendar must contain the same dates")
    if forecast.isna().any() or not np.isfinite(forecast).all() or np.isinf(actual).any():
        raise ValueError("Forecasts must be finite and actuals must be finite or missing")
    return actual.sort_index().rename("actual").to_frame().join(
        forecast.rename("prediction"), how="left", validate="one_to_one",
    ).rename_axis("Date")


def score(rows: pd.DataFrame) -> dict:
    scored = rows.dropna(subset=["actual"])
    errors = scored["prediction"] - scored["actual"]
    nonzero = scored["actual"].ne(0)
    return {
        "forecast_rows": len(rows),
        "scored_rows": len(scored),
        "mape_rows": int(nonzero.sum()),
        "mae": errors.abs().mean(),
        "rmse": np.sqrt(errors.pow(2).mean()),
        "mape_pct": (errors[nonzero] / scored.loc[nonzero, "actual"]).abs().mean() * 100,
    }


def evaluation_windows(calendar: pd.DatetimeIndex):
    split = int(len(calendar) * 0.8)
    for length in ROLLING_TRAIN_WEEKS:
        if length + ROLLING_HORIZON <= split:
            yield "rolling", calendar[length - 1], calendar[length:length + ROLLING_HORIZON]
    yield "holdout", calendar[split - 1], calendar[split:]


def evaluate(data_dir: Path, out: Path) -> dict[str, pd.DataFrame]:
    panel = load_panel(data_dir)
    scopes = {PRIMARY: complete_scope(panel), FULL: panel}
    forecasts, fills, fits, future = [], [], [], []
    for scope, data in scopes.items():
        actuals = observed_totals(data)
        for evaluation, cutoff, dates in evaluation_windows(data.index):
            if evaluation == "rolling" and actuals.loc[dates].isna().all():
                continue  # No observed target is available to score this fold.
            train, imputed = prepare_training(data, cutoff)
            models, optimizers = forecast_models(train, len(dates))
            adf_statistic, adf_pvalue, *_ = adfuller(train, autolag="AIC")
            fills.append(imputed.assign(scope=scope, evaluation=evaluation, cutoff=cutoff))
            fits.append({"scope": scope, "evaluation": evaluation, "cutoff": cutoff,
                         "train_weeks": len(train), "imputed_rows": len(imputed),
                         "adf_statistic": adf_statistic, "adf_pvalue": adf_pvalue,
                         **{f"{model}_optimizer": method for model, method in optimizers.items()}})
            for model, prediction in models.items():
                aligned = align_predictions(actuals.loc[dates], prediction).reset_index()
                forecasts.append(aligned.assign(
                    scope=scope, evaluation=evaluation, cutoff=cutoff, model=model,
                    horizon=range(1, len(dates) + 1),
                ))

        cutoff = data.index[-1]
        train, imputed = prepare_training(data, cutoff)
        fills.append(imputed.assign(scope=scope, evaluation="future", cutoff=cutoff))
        prediction = prophet_forecast(train, FUTURE_HORIZON, intervals=True)
        interval = prediction[["yhat", "yhat_lower", "yhat_upper"]]
        if not np.isfinite(interval).all().all():
            raise ValueError("Future Prophet forecast contains nonfinite values")
        future.append(interval.reset_index().assign(scope=scope, cutoff=cutoff))

    predictions = pd.concat(forecasts, ignore_index=True)
    keys = ["scope", "evaluation", "model"]
    metrics = [{**dict(zip(keys, group, strict=True)), **score(rows)}
               for group, rows in predictions.groupby(keys, sort=False)]
    fold_keys = [*keys, "cutoff"]
    folds = [{**dict(zip(fold_keys, group, strict=True)), **score(rows)}
             for group, rows in predictions.groupby(fold_keys, sort=False)]
    coverage = panel.notna().sum(axis=1).rename("observed_series").to_frame()
    coverage["missing_series"] = panel.shape[1] - coverage["observed_series"]
    imputed_frames = [frame for frame in fills if not frame.empty]
    reports = {
        "metrics": pd.DataFrame(metrics),
        "fold_metrics": pd.DataFrame(folds),
        "predictions": predictions,
        "missing_gaps": missing_gaps(panel),
        "weekly_coverage": coverage.reset_index(),
        "imputation_audit": pd.concat(imputed_frames, ignore_index=True) if imputed_frames else fills[0],
        "training_diagnostics": pd.DataFrame(fits),
        "prophet_26_week_forecast": pd.concat(future, ignore_index=True),
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in reports.items():
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.8f", date_format="%Y-%m-%d")
    metadata = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.machine()}",
        "versions": {name: version(name) for name in
                     ("numpy", "pandas", "scipy", "statsmodels", "prophet", "cmdstanpy")},
        "seed": SEED,
        "frequency": FREQUENCY,
        "source_sha256": {
            f"company{number}.csv": hashlib.sha256(
                (data_dir / f"company{number}.csv").read_bytes()
            ).hexdigest()
            for number in COMPANIES
        },
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT, help="Directory with company1–4.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    reports = evaluate(args.data_dir, args.output_dir)
    print(reports["metrics"].to_string(index=False, float_format=lambda value: f"{value:,.2f}"))


if __name__ == "__main__":
    main()
