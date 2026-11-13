# Retail Demand Forecasting & Business Performance Analysis

**Yash Yadav · IIT Kanpur SNT Summer Project 2023**

A retail consulting case study that examines sales across stores, infers the identities of unnamed departments from sales patterns, and forecasts the next six months of weekly sales. The analysis combines store benchmarking, department profiling, and time-series evaluation to identify where a retail planning team should investigate performance or prepare for seasonal peaks.

## Assignment brief

| Dimension | Case scope |
| --- | --- |
| Retailers | Bigg Bazar, D-Mart, Spar, and Seven Eleven |
| Stores | Seven per retailer, one each in Delhi, Mumbai, Kolkata, Lucknow, Bangalore, Indore, and Jaipur |
| Departments | Beverages, Groceries, Electronics, Clothing & Apparel, Furniture, and Books & Stationery |
| Sales history | 143 weeks from February 2010 through October 2012, with a weekly holiday flag |
| Deliverables | Explain store sales behavior and suggest improvements, infer the six department names from their sales patterns, and produce a six-month sales forecast |

The retailer and city names are the assignment's labels for groups of anonymized Walmart stores. The analysis compares those groups within the case. The input files provide department IDs, and the forecast target is recorded sales value. [Source dataset and holiday calendar](https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data)

## Business performance and recommendations

| Analysis | Finding | Planning implication |
| --- | --- | --- |
| Retailer performance | D-Mart led recorded sales at **158.4 million**, followed by Spar and Bigg Bazar. Seven Eleven had lower observation coverage than the other groups. | Compare stores on a common set of weeks and departments before attributing differences to operating performance. |
| Store performance | Mumbai led total recorded sales across the city groups, followed by Indore and Delhi. The leading retailer varied by location. Seven Eleven led within the Indore group despite ranking fourth overall. | Use the city-by-retailer comparison to select stores for peer review and examine their department mix and weekly sales patterns. |
| Seasonality and holidays | Departments 5 and 6 showed pronounced year-end peaks. Their mean sales were **57.33%** and **75.26%** higher on flagged Fridays, respectively. | Review stock and staffing plans for peak weeks, and use event-level and promotion data to investigate the source of the differences. |

The source flags selected US events, including Thanksgiving and Christmas. The holiday comparisons describe observed associations with those weeks.

### Store performance scorecard

The [notebook](Demand_forecasting.ipynb) profiles all 28 stores with record coverage, sales growth over 39 matched Fridays in 2011 and 2012, and each store's largest department by sales share in the 2012 comparison. Sales and department mix use departments 1–4, which are fully observed at every store. Coverage is calculated separately for all six departments. The matched weeks run from February through October.

Seven Eleven's Indore store grew **11.4%**, while its Kolkata store declined **5.4%** on the same departments and weeks. Kolkata recorded **704 of 858** expected store–department weeks across all six departments. Review its coverage first, then compare department trends with Indore. D-Mart led overall recorded sales, while four of its seven stores declined on the matched comparison, giving the team four locations to investigate.

### Department identification

The assignment asks for a plausible mapping from anonymous IDs to six candidate departments. The proposed mapping uses sales levels, seasonal timing, and differences on flagged holiday weeks.

| Department ID | Proposed category | Observed pattern |
| --- | --- | --- |
| 1 | Clothing & Apparel | Recurring peaks and a modest flagged-week difference |
| 2 | Groceries | Highest mean weekly sales and a small negative flagged-week difference |
| 3 | Books & Stationery | Highest mean weekly sales in August |
| 4 | Beverages | Mid-range sales and a modest flagged-week difference |
| 5 | Electronics | Pronounced year-end peak and higher sales on flagged weeks |
| 6 | Furniture | Lowest sales level and largest proportional flagged-week difference |

These category assignments are hypotheses based on the sales patterns. Confirm them with product or department records before using them for category decisions.

## Forecasting approach

1. **Audit the weekly sales panel.** Validate unique store–department–week keys and Friday dates, retain the six in-scope departments, and document missing records. The expected panel has 168 series and 167 missing observations. One series accounts for 124 missing weeks across five gaps.
2. **Fit models and baselines.** Compare Prophet, Holt–Winters, ARIMA, and SARIMA against a 52-week seasonal naive baseline and a 12-week moving average.
3. **Evaluate on aligned dates.** Train chronologically and match each forecast to the observed total for the same Friday. The main comparisons use the same 164 consistently observed series. The separate full-panel sensitivity check fills training gaps using only observations available at each cutoff and scores only fully observed Friday totals.

## Forecast results

MAPE is mean absolute percentage error. The 29-week holdout uses one cutoff. The rolling evaluation forecasts four weeks at each of four earlier cutoffs. The year-end test forecasts nine Fridays from November through December 2011 and compares three models.

| Model | 29-week holdout MAPE | Four-week rolling MAPE | Nine-week year-end MAPE |
| --- | ---: | ---: | ---: |
| Prophet | **3.58%** | 7.04% | 16.35% |
| Seasonal naive | 5.02% | **6.52%** | **7.03%** |
| Holt–Winters | 5.98% | 8.01% | — |
| 12-week moving average | 9.69% | 14.32% | 16.72% |
| ARIMA | 11.28% | 7.26% | — |
| SARIMA | 32.58% | 7.13% | — |

Seasonal naive matched the December 23 peak date and overestimated its size by **13.0%**. It overpredicted Thanksgiving-week sales by **0.9%**, while Prophet underpredicted them by **30.5%**. These results show why peak timing and magnitude matter alongside average error for seasonal planning.

### Six-month forecast

Prophet was refit on the full 164-series history to project **26 weekly totals from November 2, 2012 through April 26, 2013**. The projected peak is **5.72 million in recorded sales value** for the week ending December 21, 2012. The notebook includes the forecast path and nominal 80% intervals.

The year-end backtest covers one season, the full-panel sensitivity check has two complete holdout weeks, and coverage of the six-month prediction intervals has not been established. Inventory decisions at store or product level require forecasts and validation at those levels.

## Reproduce the analysis

Use **Python 3.12** and run from the repository root with the pinned versions below. Open the notebook and select **Run All** to regenerate the results and charts and run the regression checks.

```bash
python3.12 -m venv .venv
source .venv/bin/activate

python -m pip install \
  cmdstanpy==1.2.5 contourpy==1.4.0 cycler==0.12.1 fonttools==4.68.0 \
  holidays==0.108 importlib_resources==7.1.0 kiwisolver==1.5.1 matplotlib==3.11.2 \
  numpy==2.2.6 packaging==26.3 pandas==2.2.3 patsy==1.0.3 \
  pillow==13.0.0 prophet==1.1.7 pyparsing==3.3.3 python-dateutil==2.9.0.post0 \
  pytz==2026.6 scipy==1.17.1 six==1.17.0 stanio==0.5.1 \
  statsmodels==0.14.5 tqdm==4.70.1 tzdata==2026.6 notebook==7.4.7

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
jupyter notebook Demand_forecasting.ipynb
```

## Repository contents

| Path | Contents |
| --- | --- |
| [`Demand_forecasting.ipynb`](Demand_forecasting.ipynb) | Data audit, exploratory analysis, forecasting, and validation |
| [`company1.csv`](company1.csv) through [`company4.csv`](company4.csv) | Weekly sales inputs |
| [`final-report.pdf`](final-report.pdf) | Project report and forecast findings |
| [`LICENSE`](LICENSE) | MIT license for repository code |
