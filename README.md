# Retail Demand Forecasting & Business Performance Analysis

An IIT Kanpur Consulting Group case study on weekly retail sales across Bigg Bazar, D-Mart, Spar, and Seven Eleven. It compares statistical forecasting models and baselines, and preserves the submitted analysis of company, city, department, and holiday sales patterns.

[Run the notebook](Demand_forecasting.ipynb) · [Read the corrected report](final_report.pdf) · [Review the validation methodology](docs/forecasting-validation.md) · [Inspect the original submission](historical/README.md)

## Validated results

The target is **weekly sales value** summed across a fixed cohort of **164 consistently observed store-department series**. The data spans 143 Fridays from February 2010 to October 2012. The holdout forecasts the last 29 weeks from one cutoff. Rolling backtesting refits each model at four earlier origins and forecasts four weeks per origin.

| Model | Holdout MAE | Holdout RMSE | Holdout MAPE | Rolling MAPE |
|---|---:|---:|---:|---:|
| Prophet | 121,689.41 | 181,477.53 | 3.58% | 7.04% |
| Seasonal naive | 169,307.74 | 313,436.33 | 5.02% | 6.52% |
| Holt-Winters | 202,354.14 | 265,527.97 | 5.98% | 8.01% |
| Moving average | 330,996.41 | 381,015.37 | 9.69% | 14.32% |
| ARIMA | 383,009.66 | 426,570.21 | 11.28% | 7.26% |
| SARIMA | 1,096,160.22 | 1,147,789.02 | 32.58% | 7.13% |

Prophet leads the 29-week holdout. Seasonal naive has the lowest MAE and MAPE in the four-week rolling backtests, and ARIMA has the lowest rolling RMSE. The rolling windows share some target dates and cover 10 distinct Fridays. Full per-fold scores and date-level predictions are available in the [results](results/).

The **168-series sensitivity check** refits all models with cutoff-safe training imputation and scores only the **two** holdout Fridays with every actual observed. The [validation report](docs/forecasting-validation.md#full-panel-sensitivity) gives those scores and explains the coverage limit.

## Data and methodology

- Four CSVs cover seven stores per retailer and six in-scope departments per store. The pipeline filters out-of-scope department identifiers from Spar and validates the date calendar and key uniqueness.
- Seven Eleven has **167 missing store-department weeks**. Store 3, Department 6 accounts for 124 across **five gaps** of 31, 39, 35, 11, and 8 weeks in chronological order. The [gap audit](results/missing_gaps.csv) gives the dates and other affected series.
- Every model forecasts Fridays. Scoring requires exact date coverage and joins actuals and forecasts by date. Training fills use a prior sale from the same series or a same-Friday observed peer median when a series has no earlier sale. Evaluation actuals come directly from the source CSVs.
- The notebook generates 26 Friday-dated Prophet forecasts with nominal 80% intervals for each scope. The observed history is too short to validate interval coverage at that horizon. Sales value is a proxy for demand. The CSVs give sales values without currency metadata, unit volumes, or unmet-demand measurements.

The historical exploratory analysis covers retailer and city performance, seasonal and holiday sales patterns, and hypotheses about the anonymized departments. Department identities and holiday effects remain interpretations of observational data. [Historical notebook and report](historical/README.md)

## Reproduce

Use Python 3.12 and run from the repository root. The pinned environment contains the complete forecasting dependency set.

~~~bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python -m unittest discover -s tests -v
python -m forecasting.evaluate
~~~

The command writes the metrics, predictions, missingness audit, and run metadata to `results/`. Pass `--output-dir PATH` to write elsewhere. To run the notebook interactively, install Jupyter Notebook in the active environment and open `Demand_forecasting.ipynb` from the repository root.

| Path | Contents |
|---|---|
| [forecasting/](forecasting/) | Panel preparation, model adapters, and evaluation |
| [results/](results/) | Versioned rerun outputs and source checksums |
| [tests/](tests/) | Regression tests for Friday alignment, metrics, imputation, and source gaps |
| [historical/](historical/) | Original notebook, saved outputs, and submitted report |
| [company1.csv](company1.csv)–[company4.csv](company4.csv) | Source sales data |

Released under the [MIT License](LICENSE).
