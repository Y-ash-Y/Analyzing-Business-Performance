# Retail Demand Forecasting & Business Performance Analysis

An IIT Kanpur Consulting Group case study in retail sales analysis and forecasting. The notebook explores weekly sales, data coverage, seasonality and holiday associations, then compares five statistical forecasting approaches with a seasonal-naive baseline.

[Forecasting notebook](Demand_forecasting.ipynb) · [Submitted report](final_report.pdf) · [Historical notebook and saved results](https://github.com/Y-ash-Y/Analyzing-Business-Performance/blob/67e83af15bf4dffb490ae6519920fa586b8d6f4b/Demand_forecasting.ipynb)

## Data and scope

The four case CSVs contain 143 Fridays from February 5, 2010 to October 26, 2012. A row-level comparison matched all 32,454 supplied rows to the Walmart recruiting dataset after restoring the source store IDs. The files map to Walmart stores 1–7, 12–18, 23–29 and 34–40. The project’s retailer and city names are assignment aliases. The source gives anonymized store and department IDs. [Walmart data description](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data) · [Reference CSV used for the comparison](https://github.com/shauryanegi/Walmart-Store-Sales-Forecasting/blob/HEAD/train.csv.zip)

The case scope includes departments 1–6. Company 3’s Store 7 includes 8,597 additional source records for other departments. Company 4 has 167 absent store-department weeks within the case scope. Its Store 3, Department 6 accounts for 124 across **five gaps** of 31, 39, 35, 11 and 8 weeks. The same dates are absent in the source reference, and the reason for the gaps is unknown.

## Validated forecasting results

The target is **aggregate weekly sales value across 164 consistently observed store-department series**. The holdout forecasts 29 Fridays from a single cutoff. The rolling comparison refits each model at four earlier cutoffs and forecasts four weeks per cutoff.

| Model | Holdout MAE | Holdout RMSE | Holdout MAPE | Rolling MAPE |
|---|---:|---:|---:|---:|
| Prophet | 121,689.41 | 181,477.53 | **3.58%** | 7.04% |
| Seasonal naive | 169,307.74 | 313,436.33 | 5.02% | **6.52%** |
| Holt–Winters | 202,354.14 | 265,527.97 | 5.98% | 8.01% |
| Moving average | 330,996.41 | 381,015.37 | 9.69% | 14.32% |
| ARIMA | 383,009.66 | 426,570.21 | 11.28% | 7.26% |
| SARIMA | 1,096,160.22 | 1,147,789.02 | 32.58% | 7.13% |

Prophet leads the holdout on all three measures. Seasonal naive leads the rolling comparison on MAE and MAPE, and ARIMA leads on rolling RMSE. The overlapping rolling windows contain 16 forecast pairs across 10 distinct Fridays. The 29-week holdout includes one flagged holiday Friday, and neither evaluation includes November or December. The year-end backtest below evaluates peak timing and size in a separate window.

### 2011 year-end backtest

The three models trained on 91 weeks through October 28, 2011 and forecast nine Fridays from November 4 through December 30 on the 164-series target. Positive errors indicate overprediction. Peak level error compares each model’s maximum forecast with the observed maximum of 6.94 million on December 23.

| Model | Nine-week MAPE | Thanksgiving-week error | Forecast peak date | Peak level error |
|---|---:|---:|---|---:|
| Seasonal naive | **7.03%** | +0.9% | Dec 23 | +13.0% |
| Prophet | 16.35% | −30.5% | Dec 16 | −15.1% |
| Moving average | 16.72% | −33.7% | All nine weeks tied | −47.1% |

Seasonal naive had the lowest MAE and RMSE and matched the observed peak date. This comparison covers one year-end season. The 26-week forecast intervals remain uncalibrated.

### Validation method

- **Exact dates.** All models forecast Fridays. Scoring validates unique date keys, requires the same date set, and joins predictions to observed actuals by date.
- **Chronological splits.** The holdout trains on 114 weeks through April 6, 2012. The four earlier rolling folds train on 104, 106, 108 and 110 weeks. ADF uses training data and serves as a diagnostic.
- **Missing sales.** The primary cohort excludes the four incomplete series. The separate 168-series sensitivity fills only training gaps using earlier observed sales or same-Friday peer medians for leading gaps. It scores the two complete holdout Fridays, June 8 and August 31, 2012. The forecast notebook keeps the gap audit, fills, fold scores and date-level predictions available as tables.
- **Historical comparison.** The original Prophet run paired Sunday predictions with Friday sales by row position and scored a reconstructed 168-series target. The historical 4.80% MAPE and the current 3.58% MAPE come from different evaluation setups.

## Interpreting the business results

The observed sums reproduce the original rankings within the case groups, including the highest and lowest assigned city totals. They measure recorded sales, with unequal coverage for one group, and provide no evidence on market size, profitability or store efficiency.

The source’s ten flagged dates correspond to Super Bowl, Labor Day, Thanksgiving and Christmas. On the complete cohort, departments 5 and 6 show **57.33%** and **75.26%** higher mean sales on flagged Fridays, respectively. These are descriptive associations. Department names are hypotheses. The files provide no measures of promotions, units demanded, stockouts or margins. [Source holiday calendar](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

## Run locally

Use Python 3.12. From the repository root, create an environment with the versions used for the forecasting rerun.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install \
  cmdstanpy==1.2.5 contourpy==1.4.0 cycler==0.12.1 fonttools==4.68.0 \
  holidays==0.108 importlib_resources==7.1.0 kiwisolver==1.5.1 matplotlib==3.11.2 \
  numpy==2.2.6 packaging==26.3 pandas==2.2.3 patsy==1.0.3 \
  pillow==13.0.0 prophet==1.1.7 pyparsing==3.3.3 python-dateutil==2.9.0.post0 \
  pytz==2026.6 scipy==1.17.1 six==1.17.0 stanio==0.5.1 \
  statsmodels==0.14.5 tqdm==4.70.1 tzdata==2026.6
python -m pip install notebook==7.4.7

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
jupyter notebook Demand_forecasting.ipynb
~~~

In Jupyter, select **Run All**. The notebook computes the exploratory charts, forecast results, and six regression checks in memory.

Repository code is released under the [MIT License](LICENSE).
