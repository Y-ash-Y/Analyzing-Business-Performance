# Forecasting validation report

## Corrections to the submitted report

The accompanying [report PDF](../final_report.pdf) opens with a correction notice. The [original report and notebook](../historical/README.md) are archived byte-for-byte.

- **Sections 3.2 and 9.6.** Seven Eleven Store 3, Department 6 is missing 124 of its 143 expected Friday observations across five runs. There are 19 observed weeks. The source establishes missing records and their dates. It does not establish the cause or the amount of sales during the gaps.
- **Section 3.3.** The submitted notebook used interpolation and backfill on the complete panel before splitting. The 35-week gap from October 7, 2011 through June 1, 2012 straddles the April 6 training cutoff. Its next observation, June 8, 2012, was available only in the holdout. The revised workflow preserves raw missingness and prepares each training prefix separately.
- **Sections 6–8 and 9.5–10.** The original Prophet evaluation paired Sunday predictions with Friday actuals by array position. Forecasts now use `W-FRI` and join on exact dates. The historical score, chart, future dates, and inventory guidance based on that score or the historical intervals are superseded by the experiment and limitations below.

| First missing Friday | Last missing Friday | Weeks |
|---|---|---:|
| 2010-02-05 | 2010-09-03 | 31 |
| 2010-12-31 | 2011-09-23 | 39 |
| 2011-10-07 | 2012-06-01 | 35 |
| 2012-06-15 | 2012-08-24 | 11 |
| 2012-09-07 | 2012-10-26 | 8 |
| **Total** | | **124** |

The other 43 missing records occur in Store 3 / Department 5 (30), Store 4 / Department 6 (8), and Store 5 / Department 6 (5). The [gap audit](../results/missing_gaps.csv) contains every contiguous run. The [weekly coverage](../results/weekly_coverage.csv) records the number of observed series for each Friday.

## Dataset and scoring rules

The expected panel contains 168 store-department series across four retailers. It spans February 5, 2010 through October 26, 2012. As in the submitted analysis, Spar is restricted to department IDs 1–6, dropping 8,597 records. The loader checks key uniqueness, Friday dates, expected series, and valid sales. The panel preserves missing sales and observed zero values separately.

The primary target sums the **164 consistently observed series** each Friday. This is a retrospective coverage-based cohort defined by the four known incomplete keys above, with membership independent of sales amounts. A fixed cohort gives the same target at every origin and avoids scoring on reconstructed values. It covers 164 of 168 series and omits four Seven Eleven series. The original report used the reconstructed 168-series total.

For the full-panel sensitivity, each model receives only the training prefix through its cutoff. Gaps after a series has recorded sales carry forward its most recent observed sale. A leading gap uses the median of observed stores in the same retailer and department on that Friday. Peer medians use raw observations. Both methods use information dated at or before the imputed Friday, and the pipeline records each filled key, date, value, method, and cutoff in the [imputation audit](../results/imputation_audit.csv). Carry-forward over long runs is a sensitivity assumption. A complete 168-series actual requires all 168 original sales values for that Friday.

## Evaluation design

- **Fixed holdout.** Fit on the first 114 Fridays through April 6, 2012. Forecast April 13 through October 26, 2012 in one 29-week forecast. The same origin and target dates apply to all models.
- **Rolling-origin backtest.** Refit on 104, 106, 108, and 110 weeks at cutoffs January 27, February 10, February 24, and March 9, 2012. Forecast the next four Fridays without updating inside a fold. Every fold ends by April 6. The common minimum of 104 weeks gives Holt–Winters two 52-week cycles for seasonal initialization and keeps the comparison consistent. Only ten weeks remain between that minimum and the holdout, which rules out an independent 26-week rolling comparison under this design.
- **Metrics.** Forecast and actual indexes must have the same unique, normalized Friday dates. MAE and RMSE use sales-value units. MAPE is the mean absolute error as a percentage of the absolute actual and excludes zero actuals. Missing actuals are excluded from all scores. The pooled rolling metrics give each origin–target pair equal weight, including repeated target Fridays at different horizons. There are 16 pairs and 10 distinct Fridays per model.
- **Diagnostics.** ADF is computed on each training prefix only and logged as a diagnostic. The fixed model orders do not depend on the ADF results. Statsmodels issued SARIMA starting-value warnings on the short seasonal histories, and the run records the optimizer that converged. The code checks convergence and requires finite forecasts.

## Model specifications

| Model | Fixed specification |
|---|---|
| Moving average | Mean of the final 12 training weeks, repeated through the horizon |
| Holt–Winters | Additive trend and 52-week additive seasonality |
| ARIMA | (2, 0, 2) with the Statsmodels default constant |
| SARIMA | (1, 0, 1) × (0, 0, 1, 52), no intercept, on `log1p(sales)`, inverted with `expm1` |
| Prophet | Additive trend and yearly seasonality, no daily or weekly seasonal term, no holiday regressors |
| Seasonal naive | Repeat the last observed 52-week pattern |

Holt–Winters and ARIMA fit sales in millions for numerical conditioning and report forecasts in original sales-value units. SARIMA's inverse-transformed point forecasts are reported without log-scale mean bias adjustment to preserve the historical specification. The ARIMA and SARIMA optimizer retries use the same training sample. Prophet uses seed 1729 for fitting and predictive simulations.

## Comparison with the historical submission

| Model | Historical MAPE | Validated holdout MAPE | Validated rolling MAPE |
|---|---:|---:|---:|
| Prophet | 4.80% | 3.58% | 7.04% |
| Seasonal naive | — | 5.02% | 6.52% |
| Holt-Winters | 7.13% | 5.98% | 8.01% |
| Moving average | 9.69% | 9.69% | 14.32% |
| ARIMA | 11.18% | 11.28% | 7.26% |
| SARIMA | 32.58% | 32.58% | 7.13% |

The historical MAPE comes from the submitted notebook. The corrected runs use a different target, calendar alignment, training preprocessing, and pinned environment. The changes in MAPE cannot be attributed to any one fix. Prophet has the lowest holdout MAE, RMSE and MAPE. Seasonal naive has the lowest rolling MAE and MAPE, while ARIMA has the lowest rolling RMSE. The windows cover different forecast horizons and parts of the retail year.

## Full-panel sensitivity

The full-panel holdout has complete observed totals on **June 8 and August 31, 2012**. Each model forecasts all 29 Fridays and is scored on those two dates. The rolling windows contain zero Fridays with complete 168-series actuals, so the full-panel rolling score is unavailable.

| Model | MAE | RMSE | MAPE |
|---|---:|---:|---:|
| Prophet | 180,361.92 | 181,416.25 | 4.79% |
| Seasonal naive | 361,115.19 | 377,641.42 | 9.20% |
| Holt-Winters | 166,116.34 | 179,811.16 | 4.19% |
| Moving average | 378,285.32 | 434,561.67 | 9.38% |
| ARIMA | 386,853.91 | 409,645.30 | 9.81% |
| SARIMA | 641,083.48 | 696,746.08 | 17.54% |

These two dates provide a limited check on the full-coverage target. The missing values in the other 27 weeks prevent a complete holdout comparison for that target.

## 26-week forecast and reproducibility

Prophet is refit through October 26, 2012 for both scopes. The [forecast CSV](../results/prophet_26_week_forecast.csv) contains 26 Fridays per scope from November 2, 2012 through April 26, 2013. Its seeded, nominal 80% model intervals have no demonstrated calibration at the 26-week horizon. The available evidence supports comparisons of aggregate sales forecasts within the stated evaluation windows. Store- and product-level inventory decisions require data and validation at those levels.

[Metrics](../results/metrics.csv) · [Per-origin metrics](../results/fold_metrics.csv) · [Date-level forecasts and actuals](../results/predictions.csv) · [Training diagnostics](../results/training_diagnostics.csv) · [Environment and source checksums](../results/run_metadata.json)
