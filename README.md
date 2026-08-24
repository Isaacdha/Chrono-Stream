# Chrono Stream

Chrono Stream is a Streamlit application for preparing, exploring, benchmarking, and forecasting a univariate time series. It accepts CSV and Excel files, includes three sample datasets, evaluates every model on a shared holdout window, and compares saved forecasts in one dashboard.

## What is included

- Robust date and numeric-value validation
- Duplicate timestamp aggregation and missing-period handling
- Automatic frequency detection with manual frequency overrides
- Automatic moving-average window and exponential-smoothing coefficient estimation
- Auditable Box-Jenkins ARIMA and SARIMA workflows with reversible variance transformations, automatic differencing, diagnostic-gated order selection, and manual overrides
- Interactive exploration, seasonal decomposition, autocorrelation, and an Augmented Dickey-Fuller test
- Out-of-sample MAE, RMSE, MAPE, and sMAPE for every fitted model
- Fitted values, future forecasts, approximate 95% intervals, and CSV downloads
- Side-by-side result ranking and forecast visualization
- Two compact controls beside every method title: `!` for method/test guidance and `?` for literature reviews, references, and downloads
- A statistical decision handbook for ARIMA/SARIMA covering hypotheses, statistics, reference distributions, alpha/critical-value rules, assumptions, exact software behavior, and primary literature
- A project-local virtual environment so the system Python installation is not changed

Forecasting methods:

- Smoothing: moving average, weighted moving average, single exponential smoothing, Holt, and Holt-Winters
- Statistical: ARIMA, SARIMA, and a portable STL decomposition forecast (X-11-inspired)
- Machine learning: Prophet, LSTM, 1D CNN, and XGBoost
- Deterministic trends: linear, quadratic, exponential, and logarithmic

## Requirements

- 64-bit Python 3.12
- Windows, macOS, or Linux
- Approximately 2 GB of free disk space for the complete environment (TensorFlow is the largest dependency)

The tested dependency versions are pinned in [`requirements.txt`](requirements.txt). The environment created below is isolated in `.venv`, which is excluded from Git.

## Install and run

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run chrono_app.py
```

After the first installation, `run.bat` starts the app with the project-local interpreter.

### macOS or Linux

```bash
python3.12 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
./.venv/bin/python -m streamlit run chrono_app.py
```

Streamlit prints the local application URL, normally `http://localhost:8501`.

## Input format

Provide a CSV or XLSX file containing at least:

| Date/time column | Numeric value column |
|---|---:|
| 2025-01-01 | 125.4 |
| 2025-02-01 | 131.8 |

The column names can be different. Chrono Stream asks which columns to use and preserves those names in output metadata. Invalid rows are reported before saving. For irregular data, choose a frequency and a missing-period strategy to create a regular series.

## Workflow

1. Open **Data Input & Settings**.
2. Upload a file or select a bundled sample.
3. Choose the date and value columns, preparation rules, forecast horizon, evaluation holdout, and default seasonal period.
4. Save the prepared series.
5. Inspect **Data Exploration** and fit any forecasting methods from the navigation.
6. Open **Compare Results** to rank holdout accuracy and download the combined forecasts.

Model results live in the current Streamlit session. Saving different input data or shared forecast settings clears old results so stale or non-comparable forecasts cannot be mixed accidentally.

## Model notes

- ARIMA and SARIMA provide **Strict Box-Jenkins**, **Forecast-oriented**, and **Custom** diagnostic policies. Strict mode accepts a candidate only when stationarity, optimizer convergence, stationary/invertible roots, coefficient significance, zero residual mean, residual normality, and degrees-of-freedom-corrected multi-lag Ljung-Box or Box-Pierce tests all pass. An optional ARCH LM residual-variance gate is also available.
- Variance stabilization can be disabled or set to Auto, Box-Cox, Yeo-Johnson, logarithm, or square root. Transformation parameters and any positive shift are fitted only from the available training observations. Forecasts and intervals are returned to the original scale; simulation-based inverse transformation can correct nonlinear mean bias.
- Regular differencing supports ADF, KPSS, Phillips-Perron, or an ADF + KPSS consensus. SARIMA determines seasonal differencing first with OCSB, Canova-Hansen, or a seasonal-ACF rule, and always uses the configured seasonal period.
- ACF and PACF can guide the initial order set. Users can instead request an exhaustive bounded grid, a Hyndman-Khandakar-style stepwise search, one manual order, or manual lists of `p/q/P/Q` candidates. Guided and stepwise searches can expand to the complete configured grid when no initial candidate passes.
- Eligible models can be ranked by AICc, AIC, BIC, HQIC, or expanding-window CV RMSE/MAE. Rolling validation stays inside the pre-holdout training partition and refits the variance transformer for each fold. When strict diagnostics reject every candidate, Chrono Stream reports every failure and does not silently label a failing model as best. A near-match can be used only through an explicit override.
- ARIMA/SARIMA results include transformation and stationarity histories, pre-model ACF/PACF, a candidate leaderboard, coefficient estimates and p-values, root magnitudes, normality and residual-mean tests, the complete white-noise lag table, ARCH LM and auxiliary F results, residual plots, residual ACF/PACF, a Q-Q plot, and a downloadable JSON diagnostic report.
- The complete tuning pipeline is run separately on the pre-holdout training data and the complete series. This prevents the holdout from influencing its own evaluation, so the evaluation transformation, differencing, and order can differ from the final forecast pipeline.
- Moving-average methods can select their window by one-step-ahead training RMSE. SES, Holt, and Holt-Winters can estimate their smoothing coefficients automatically. Every automatic control can be disabled to enter values manually.
- The former “X-11-style” page is now labeled **STL Decomposition Forecast (X-11-inspired)**. It runs robust STL, extrapolates the terminal STL trend with a fitted line, repeats phase-averaged seasonality, and does not claim to reproduce the official Census X-11/X-13 filter, trading-day, outlier, or diagnostic workflow.
- Exponential trend forecasts use Duan's nonparametric smearing estimate after log-linear fitting, avoiding the original implementation's naive retransformation bias when targeting the conditional mean.
- Prophet, TensorFlow, and XGBoost are imported only when their pages are fitted, keeping normal app startup responsive.
- LSTM and CNN use a fixed random seed to improve reproducibility, but they are more computationally expensive than the classical models.
- Confidence intervals for models without native probabilistic intervals are residual-based approximations. ARIMA, SARIMA, and Prophet use their model-native intervals.
- Forecast quality depends on data quality and whether a model's assumptions match the series. Holdout scores are evidence, not a guarantee of future performance.

## Built-in method research notes

Every forecasting page has two controls beside its title:

- `!` opens the method guide: how the method works, how this app implements it, when it fits, and its limitations. On ARIMA and SARIMA pages, the **Tests** tab groups each diagnostic by purpose and gives H0, H1, statistic, decision rule, and the result of rejecting or failing to reject H0.
- `?` opens the research material: the short scholarly overview, full method literature review, APA references, and TXT downloads. ARIMA and SARIMA also include the full literature and statistical specification for every test and decision aid.

The test and reference sections use tabs rather than one long selector. References are presented as a clean APA list without repeated source annotations.

## Statistical test and decision handbook

The `!` method guide explains every decision made by the ARIMA/SARIMA workflow, while `?` contains the corresponding copy-ready test literature and downloadable handbook:

- ADF, KPSS, and Phillips–Perron regular-stationarity tests
- OCSB and Canova–Hansen seasonal tests for SARIMA
- Individual coefficient Wald z tests and the one-sample residual-mean t test
- Jarque–Bera, Shapiro–Wilk, Anderson–Darling, and Lilliefors normality tests
- Ljung–Box and Box–Pierce residual portmanteau tests
- Engle's ARCH LM diagnostic and its auxiliary F version
- Approximate pointwise ACF and PACF lag tests

Every formal entry gives `H0`, `H1`, the statistic and its reference distribution, both p-value/alpha and critical-value decision language where applicable, interpretation, assumptions, limitations, exact Chrono Stream behavior, a copy-ready literature review, and linked APA references. The Data Exploration ADF result has the same reference note and also displays its tabulated critical values.

The handbook uses **reject H0** when `p < alpha` and **fail to reject H0** otherwise. It does not use “accept H0,” because a nonsignificant result does not prove the null. The ADF + KPSS consensus, seasonal-ACF rule, polynomial-root check, optimizer convergence, information criteria, rolling-origin validation, and Q–Q plot are separately documented as decision aids rather than assigned fictional hypotheses or p-values.

## Tests

Run the complete test suite, including one-epoch neural-network smoke tests:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

On macOS or Linux, replace the executable with `./.venv/bin/python`.

## Troubleshooting

- If PowerShell blocks activation scripts, use the explicit `.venv\Scripts\python.exe` commands above; activation is not required.
- TensorFlow 2.21 runs on the CPU in native Windows. GPU acceleration on modern TensorFlow requires WSL2.
- If a seasonal model says there is not enough data, reduce the seasonal period or provide at least two complete seasonal cycles.
- If the app reports an irregular frequency, select the correct frequency manually on the data-input page.

## Project layout

```text
chrono_app.py          Streamlit entry point and navigation
chrono_stream/         Shared data, forecasting, UI, and scholarly method-note modules
method/                Streamlit workflow and model-page wrappers
tests/                 Data, statistical, and machine-learning tests
.streamlit/            Theme and image assets
requirements.txt       Tested direct dependencies
run.bat                Windows launcher for the local environment
```

## License

This project is licensed under the terms in [`LICENSE`](LICENSE).
