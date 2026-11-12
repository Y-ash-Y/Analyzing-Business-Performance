# Retail Demand Forecasting & Business Performance Analysis

**Business analytics · Statistical forecasting · Retail decision support**

An end-to-end exploratory analytics and time-series forecasting project developed as an IIT Kanpur Consulting Group summer assignment. The project examines weekly retail sales across four companies, seven cities, and six anonymized departments; investigates seasonality and holiday behavior; compares five forecasting methods; and translates findings into retail planning recommendations.

[Explore the notebook](./Demand_forecasting.ipynb) · [Read the full project report](./final_report.pdf) · [View the license](./LICENSE)

> **Project status:** Educational analytics case study with a reproducible notebook and a historical report. The published accuracy figures are **original notebook results, not yet validated after a time-index and leakage audit**. See [Validation notes](#validation-notes-and-known-limitations) before citing them as confirmed out-of-sample performance.

## Executive summary

Retailers must plan inventory and promotions before demand is observed. Excess stock ties up working capital; insufficient stock risks lost sales. This project explores whether historical weekly sales and recurring seasonal patterns can support more informed planning.

**Project scope**

| Dimension | Coverage |
|---|---|
| Retailers | Bigg Bazar, D-Mart, Spar, Seven Eleven |
| Markets | Delhi, Mumbai, Kolkata, Lucknow, Bangalore, Indore, Jaipur |
| Store / department structure | 7 stores per retailer; 6 departments per store |
| Period | 5 February 2010 – 26 October 2012 |
| Frequency | Weekly, week-ending Friday (143 weeks) |
| Analysis-ready panel | 24,024 company × store × department × week rows after reconstruction |
| Forecasting target | **Combined weekly sales value** across all companies, stores, and departments |
| Train / test | 114 training weeks / 29 held-out weeks in the original notebook |
| Forecast horizon | 26 weeks (approximately six months) |

**Important target distinction:** The data records sales value, not physical units sold or unmet demand. Sales forecasts are used as a proxy for demand; they are not directly item-level replenishment forecasts. The CSVs do not explicitly document a currency.

## Business questions

1. Which companies and cities contribute the most recorded sales?
2. How do department-level patterns vary across the year?
3. How do sales on flagged holiday weeks differ from other weeks?
4. Can the six hidden department labels be inferred from their behavioral signatures?
5. How do simple baselines compare with statistical time-series forecasting models?
6. What planning decisions could these descriptive findings inform?

## Methodology

~~~mermaid
flowchart LR
    A[Four weekly CSV datasets] --> B[Schema and completeness checks]
    B --> C[Panel reconstruction and cleaning]
    C --> D[Company, city, department and holiday EDA]
    D --> E[Time-series diagnostics]
    E --> F[Chronological model comparison]
    F --> G[26-week aggregate forecast]
    G --> H[Retail planning recommendations]
~~~

### 1. Data quality and preprocessing

- Standardized date columns and attached retailer identifiers.
- Checked store/department coverage against the assignment's 7 × 6 structure.
- Removed out-of-scope department identifiers in Spar's Store 7 (8,597 rows removed).
- Detected **167 absent store–department–week records** in Seven Eleven by reconstructing the complete expected key grid.
- Reconstructed missing observations with per-series linear interpolation and forward/backward filling in the **original notebook**.
- Validated row counts and produced a combined panel for exploratory analysis.

The reconstruction is documented as historical methodology, **not as an endorsed leakage-safe imputation policy**. One store-department series has only 19 observed weeks out of 143, making imputation particularly uncertain.

### 2. Exploratory business analysis

The notebook produces:

- Retailer-level sales totals and weekly trend comparisons
- Market-level totals and retailer-by-city comparisons
- Department-level weekly and monthly seasonal profiles
- Mean sales for flagged holiday versus non-holiday weeks
- Inferred department labels using sales magnitude, seasonality and holiday sensitivity

The inferred mapping is **hypothesis-based, not ground truth**: clothing, groceries, books/stationery, beverages, electronics and furniture.

### 3. Statistical diagnostics

The time-series section aggregates the panel into one weekly total-sales series and examines:

- Additive trend/seasonal/residual decomposition with a 52-week seasonal period
- 12-week rolling mean and standard deviation
- Augmented Dickey–Fuller (ADF) unit-root test
- Chronological 80/20 train-test split

The original ADF calculation was performed on the full series before the split. It indicates evidence against a **nonseasonal unit root under that test specification**, but does not establish the absence of annual seasonality or other forms of nonstationarity.

### 4. Forecasting models

| Model | Specification in notebook | Role |
|---|---|---|
| Moving average | Last 12 training weeks, repeated across the test horizon | Constant-value baseline |
| Holt–Winters exponential smoothing | Additive trend, additive seasonality, period = 52 | Trend/seasonal statistical baseline |
| ARIMA | (2, 0, 2) | Nonseasonal autoregressive benchmark |
| SARIMA | (1, 0, 1) × (0, 0, 1, 52) on log1p(sales) | Seasonal autoregressive benchmark |
| Prophet | Yearly seasonality enabled | Additive trend/seasonal forecast |

The original experiment evaluates **MAE**, **RMSE**, and **MAPE**. The Prophet model does not explicitly use the IsHoliday column as a custom holiday regressor.

## Results in the original notebook

The following values are reproduced from the existing notebook and report. They are retained for traceability; the table should **not** be interpreted as a validated leaderboard until the calendar alignment and missing-data treatment are corrected.

| Forecasting method | MAE | RMSE | MAPE ↓ |
|---|---:|---:|---:|
| Prophet | 161,658.64 | 236,881.28 | **4.80%** |
| Holt–Winters | 237,834.59 | 360,249.29 | 7.13% |
| Moving average | 330,993.92 | 381,027.96 | 9.69% |
| ARIMA | 379,929.25 | 423,525.98 | 11.18% |
| SARIMA | 1,095,978.69 | 1,147,631.43 | 32.58% |

The historical notebook ranks Prophet first and produces a 26-week forecast with nominal 80% prediction intervals. **This conclusion is provisional** because the original Prophet evaluation uses Sunday forecast dates against Friday actuals, comparing values by position rather than matching timestamps.

## Selected business insights

The figures below summarize descriptive patterns from the submitted report; they do not establish causal relationships.

- **Company performance:** D-Mart records approximately 158M in cumulative sales value; Spar approximately 140M, Bigg Bazar approximately 135M, and Seven Eleven approximately 89M.
- **Regional variation:** Mumbai has the highest aggregate sales, while Bangalore has the lowest. Regional leaders vary; Seven Eleven leads in Indore despite ranking last overall.
- **Holiday association:** Mean sales in the inferred Electronics and Furniture departments are approximately **57%** and **75%** higher, respectively, on flagged holiday weeks. These are descriptive differences, not causal promotion effects.
- **Seasonality:** Large recurring year-end peaks suggest the value of advance capacity planning and seasonal models.
- **Department fingerprints:** Groceries appears comparatively stable; Electronics and Furniture have holiday-linked spikes; Books & Stationery has an August peak.

**Business applications:** Inform seasonal inventory allocation, campaign timing, location-level benchmarking, and data-quality monitoring. Operational deployment would require category/store-level forecasts, lead times, stockout information, service-level targets, and calibrated uncertainty estimates.

## Repository structure

~~~text
Analyzing-Business-Performance/
├── Demand_forecasting.ipynb   # EDA, statistical tests, models, evaluation
├── company1.csv               # Bigg Bazar
├── company2.csv               # D-Mart
├── company3.csv               # Spar
├── company4.csv               # Seven Eleven
├── final_report.pdf           # Written methodology, figures and conclusions
├── README.md
└── LICENSE                    # MIT
~~~

## Run locally

The project is currently a **notebook-based analysis**, not an API or deployed forecasting service. It reads the four CSV files using paths relative to the working directory.

~~~bash
git clone https://github.com/Y-ash-Y/Analyzing-Business-Performance.git
cd Analyzing-Business-Performance

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn prophet notebook

jupyter notebook Demand_forecasting.ipynb
~~~

Run the notebook from the repository root so the CSV paths resolve. The repository does not yet contain pinned dependency versions or an automated test suite; library compatibility and numerical reproducibility may vary across environments. Prophet installation may require additional platform-specific setup.

## Validation notes and known limitations

The original notebook and historical report are preserved for transparency. The following issues were found during a review of the actual code and source CSVs:

| Priority | Finding | Corrective action |
|---|---|---|
| **P0** | Prophet uses weekly Sunday dates (freq="W"); the source series is Friday-based (W-FRI). Test forecasts are compared by array position. | Generate Friday forecasts, merge to actuals on the exact date, and recalculate all metrics. |
| **P0** | Panel imputation occurs before the temporal split. At least one long missing-data interval crosses the train/test boundary, so fitted training values can incorporate later test-period observations. | Define cutoff first; use forecasting-origin-safe reconstruction, audit each gap, and track imputed observations. |
| **P1** | The report calls 124 missing weeks in Seven Eleven's Store 3 / Department 6 "consecutive." They are **not consecutive**: the source has five missing runs (39, 35, 31, 11, and 8 weeks). | Correct the report; treat this series as high-uncertainty or exclude it in sensitivity analysis. |
| **P1** | Model orders and a 52-week season are estimated/evaluated from only about 2.2 annual cycles in the training sample. | Add diagnostics, parameter rationale, simple seasonal-naive baselines, and rolling-origin evaluation. |
| **P1** | ADF and some exploratory transformations use the full sample before a holdout is defined. | Fit/select diagnostics using training data only where they influence model choice. |
| **P2** | The 80% Prophet forecast intervals are not empirically calibrated for the stated horizon. | Evaluate interval coverage and uncertainty by horizon. |
| **P2** | Department labels and promotional explanations rely on inferred behavioral patterns. | Preserve uncertainty; distinguish association from causation. |

**Current evaluation status:** The reported historical figures are reproducible as notebook outputs, but **the corrected accuracy and model ranking have not yet been recomputed**. Avoid describing 4.80% MAPE as a validated result until that work is complete.

## Engineering roadmap

- [ ] Implement a single canonical Friday-indexed time-series preparation function.
- [ ] Add schema and key-uniqueness assertions for company × store × department × week.
- [ ] Replace backward-looking leakage with cutoff-aware missing-data handling and sensitivity checks.
- [ ] Build a common, date-aligned evaluation harness with seasonal-naive and moving-average baselines.
- [ ] Run rolling-origin validation and residual diagnostics; compare seasonal configurations.
- [ ] Record model settings, random seeds, dependency versions, and forecasts as versioned artifacts.
- [ ] Add unit tests for temporal alignment, missing weeks, train/test isolation, and metric calculations.
- [ ] Evaluate hierarchical company/store/department forecasting before using outputs for inventory decisions.

## References

- Hyndman & Athanasopoulos, [*Forecasting: Principles and Practice*](https://otexts.com/fpp3/)
- Taylor & Letham, [*Forecasting at Scale*](https://peerj.com/preprints/3190/)
- [Statsmodels time-series documentation](https://www.statsmodels.org/stable/tsa.html)
- [Prophet documentation](https://facebook.github.io/prophet/)

## License

Released under the [MIT License](./LICENSE).

---

**Project context:** IIT Kanpur Consulting Group summer project. The analysis is an educational case study; all business recommendations should be validated against actual operating constraints before deployment.
