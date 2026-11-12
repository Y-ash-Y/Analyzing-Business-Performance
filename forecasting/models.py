"""Fixed model specifications from the historical notebook and a seasonal baseline."""

import warnings

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from .data import FREQUENCY

SEASON = 52
SEED = 1729
SALES_SCALE = 1_000_000
MODEL_NAMES = ["Moving average", "Holt-Winters", "ARIMA", "SARIMA", "Prophet", "Seasonal naive"]


def forecast_dates(cutoff: pd.Timestamp, horizon: int) -> pd.DatetimeIndex:
    return pd.date_range(cutoff + pd.offsets.Week(weekday=4), periods=horizon,
                         freq=FREQUENCY, name="Date")


def prophet_forecast(train: pd.Series, horizon: int, intervals: bool = False) -> pd.DataFrame:
    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
        interval_width=0.80, uncertainty_samples=1000 if intervals else 0,
    )
    model.fit(pd.DataFrame({"ds": train.index, "y": train.to_numpy()}), seed=SEED)
    future = model.make_future_dataframe(periods=horizon, freq=FREQUENCY, include_history=False)
    if intervals:
        # Prophet's predictive simulation uses NumPy's legacy global random state.
        state = np.random.get_state()
        try:
            np.random.seed(SEED)
            forecast = model.predict(future)
        finally:
            np.random.set_state(state)
    else:
        forecast = model.predict(future)
    forecast = forecast.set_index("ds").rename_axis("Date")
    if not forecast.index.equals(forecast_dates(train.index[-1], horizon)):
        raise ValueError("Prophet did not return the requested Friday dates")
    return forecast


def _converged_fit(name, fit):
    """Retry convergence on the same data and fail if no candidate converges."""
    for optimizer in ("lbfgs", "powell"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = fit(optimizer)
        if (result.mle_retvals["converged"] and np.isfinite(result.llf)
                and np.isfinite(result.params).all()):
            return result, optimizer
    raise RuntimeError(f"{name} did not converge with either optimizer")


def forecast_models(train: pd.Series, horizon: int) -> tuple[dict, dict]:
    """Fit all models on one training series and retain their native date labels."""
    if len(train) < 2 * SEASON:
        raise ValueError("The common model comparison needs at least 104 training weeks")
    dates = forecast_dates(train.index[-1], horizon)
    scaled = train / SALES_SCALE
    smoothing = ExponentialSmoothing(
        scaled, trend="add", seasonal="add", seasonal_periods=SEASON,
    ).fit()
    if not smoothing.mle_retvals.success:
        raise RuntimeError("Holt-Winters did not converge")

    arima_model = ARIMA(scaled, order=(2, 0, 2))
    arima, arima_optimizer = _converged_fit(
        "ARIMA", lambda method: arima_model.fit(
            cov_type="none", method_kwargs={"method": method, "maxiter": 300, "disp": False},
        ),
    )
    sarima_model = SARIMAX(
        np.log1p(train), order=(1, 0, 1), seasonal_order=(0, 0, 1, SEASON),
        enforce_stationarity=True, enforce_invertibility=True,
    )
    sarima, sarima_optimizer = _converged_fit(
        "SARIMA", lambda method: sarima_model.fit(method=method, maxiter=300, disp=False, cov_type="none"),
    )
    seasonal_values = np.resize(train.iloc[-SEASON:].to_numpy(), horizon)
    predictions = {
        "Moving average": pd.Series(train.iloc[-12:].mean(), index=dates),
        "Holt-Winters": smoothing.forecast(horizon) * SALES_SCALE,
        "ARIMA": arima.forecast(horizon) * SALES_SCALE,
        "SARIMA": np.expm1(sarima.get_forecast(horizon).predicted_mean),
        "Prophet": prophet_forecast(train, horizon)["yhat"],
        "Seasonal naive": pd.Series(seasonal_values, index=dates),
    }
    optimizers = {"Holt-Winters": "L-BFGS-B", "ARIMA": arima_optimizer, "SARIMA": sarima_optimizer}
    return predictions, optimizers
