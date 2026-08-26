<div align="center">

# Chrono Stream ⏱️📈

[![Built with Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-FF4B4B.svg?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*An interactive workbench for preparing, evaluating, and forecasting univariate time series*

[Documentation](#documentation) | [Getting Started](#getting-started) | [Usage](#usage) | [Contributing](#contributing)

</div>

---

## 📊 Overview

Chrono Stream is a Streamlit application for preparing, exploring, benchmarking, and forecasting a univariate time series. It accepts CSV and Excel data, evaluates forecasting methods on one shared chronological holdout, and compares saved future forecasts in a single dashboard.

### Key Features

- 📥 Date and numeric-value validation, duplicate aggregation, frequency detection, and missing-period handling
- 🔍 Interactive plots, seasonal decomposition, ACF/PACF exploration, and an Augmented Dickey–Fuller diagnostic
- 📏 Shared out-of-sample MAE, RMSE, MASE, RMSSE, MAPE, sMAPE, and WAPE evaluation
- ⚙️ Automatic and manual controls for classical, statistical, decomposition, and machine-learning methods
- 📐 Model-native predictive intervals, analytic Gaussian baseline intervals, and clearly labelled descriptive residual bands
- 📊 Side-by-side forecast comparison, diagnostic details, and CSV/JSON/TXT downloads
- 📚 Built-in method guidance, statistical-test explanations, literature reviews, and APA references

### Forecasting Methods

- **Baselines:** naive, seasonal naive, and drift
- **Smoothing:** moving average, weighted moving average, simple exponential smoothing, Holt, Holt–Winters, Theta, and Automatic ETS
- **Statistical:** ARIMA, SARIMA, TBATS, and Croston/SBA/TSB
- **Decomposition:** STL forecasting and MSTL + ETS
- **Machine learning:** Prophet, LSTM, CNN, N-BEATS, TCN, XGBoost, lagged linear and regularized regression, CART, Random Forest, SVR, kNN, and Extra Trees
- **Deterministic trends:** linear, quadratic, exponential, and logarithmic

## 📚 Documentation <a id="documentation"></a>

### Input Format

Upload a CSV or XLSX file containing at least one date/time column and one numeric value column.

| Date | Value |
|---|---:|
| 2025-01-01 | 125.4 |
| 2025-02-01 | 131.8 |

Column names may differ. Chrono Stream asks which columns to use, reports invalid rows, and can regularise the series at a selected frequency. Three sample datasets are included for immediate use.

### Evaluation and Uncertainty

Every model is evaluated against the final shared holdout without using those observations for fitting or tuning. The model is then fitted independently to all available observations for the future forecast.

MASE and RMSSE use only pre-holdout training data for their naive or seasonal-naive scale. Undefined metrics are shown as `N/A`; they are not replaced with zero.

Uncertainty displays depend on the method. ARIMA, SARIMA, Prophet, Theta, Automatic ETS, and TBATS provide model-based intervals. Naive-family baselines use their stated Gaussian random-walk assumptions. Methods without a probabilistic forecast distribution show a descriptive in-sample residual band when estimable, without claiming nominal predictive coverage.

### Method and Research Guidance

Each forecasting page provides:

- `!` for the method explanation, implementation details, appropriate use, limitations, and relevant statistical decisions
- `?` for literature reviews, references, and downloadable research notes

ARIMA and SARIMA additionally expose their transformation, differencing, candidate-selection, coefficient, root, residual, normality, white-noise, and ARCH diagnostics. Conservative order limits are enabled by default; an advanced unbounded-orders checkbox removes those upper limits and warns when the configured search can exceed 150 candidate models.

### Repository Structure

```text
Chrono-Stream/
├── .streamlit/                 # Theme configuration
├── chrono_stream/              # Application package and shared modules
│   ├── methods/                # Forecast implementations by method family
│   │   └── statistical/
│   │       └── box_jenkins_pipeline.py
│   └── page_*.py               # Workflow pages
├── tests/                      # Automated application and forecasting checks
├── chrono_app.py               # Streamlit entry point and navigation
├── requirements.txt            # Pinned direct dependencies
├── run.bat                     # Windows launcher
├── Sample_Data_1.xlsx
├── Sample_Data_2.csv
├── Sample_Data_3.csv
├── LICENSE
└── README.md
```

`__init__.py` files intentionally mark the Python packages. Generated environments, bytecode, tool caches, editor files, logs, build output, and Streamlit secrets are excluded through [`.gitignore`](.gitignore).

## 🚀 Getting Started <a id="getting-started"></a>

### Prerequisites

- 64-bit Python 3.12
- Windows, macOS, or Linux
- Approximately 2 GB of free disk space for the complete environment

Dependency versions are pinned in [`requirements.txt`](requirements.txt). The local `.venv` environment is excluded from Git.

### Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Isaacdha/Chrono-Stream.git
   cd Chrono-Stream
   ```

2. **Create a virtual environment and install the dependencies**

   **Windows PowerShell**

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\python.exe -m pip install --upgrade pip
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

   **macOS or Linux**

   ```bash
   python3.12 -m venv .venv
   ./.venv/bin/python -m pip install --upgrade pip
   ./.venv/bin/python -m pip install -r requirements.txt
   ```

## 💻 Usage <a id="usage"></a>

### Launch the Dashboard

On Windows, start the application with the included launcher:

```powershell
.\run.bat
```

Alternatively, launch Streamlit directly:

```powershell
.\.venv\Scripts\python.exe -B -m streamlit run chrono_app.py
```

On macOS or Linux:

```bash
./.venv/bin/python -B -m streamlit run chrono_app.py
```

Streamlit prints the local URL, normally `http://localhost:8501`.

### Workflow

1. Open **Data Input & Settings**.
2. Upload data or select a bundled sample.
3. Choose preparation rules, forecast horizon, shared holdout, seasonal period, and metric scale period.
4. Save the prepared series.
5. Explore the data and fit any forecasting methods from the navigation.
6. Open **Compare Results** to review holdout accuracy and download forecasts.

Results live in the current Streamlit session. Changing the data or shared forecast settings clears saved model results so incompatible evaluations are not mixed.

## 🛠️ Troubleshooting

- PowerShell activation is optional; the commands above call the local interpreter directly.
- TensorFlow 2.21 uses the CPU on native Windows. Current NVIDIA GPU support requires WSL2.
- If a seasonal method reports insufficient history, provide more complete cycles or choose a shorter defensible period.
- If frequency cannot be inferred, select the correct interval on the data-input page.

## 🤝 Contributing <a id="contributing"></a>

Contributions are welcome. Fork the repository, create a focused branch, and open a pull request describing your changes.

## 📄 License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.

---

<div align="center">

Made by [Isaacdha](https://github.com/Isaacdha)

</div>
