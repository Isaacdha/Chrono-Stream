# Chrono Stream — Next Session Handoff

This file is the source of truth for the next development session. Read it before
changing the application. It records the agreed feature backlog, the current
architecture, the intended method-module refactor, and the checklist that every new
forecasting method must satisfy.

## Current baseline

- The project uses a local `.venv`; do not modify the system Python installation.
- Direct dependencies are pinned in `requirements.txt`. Do not downgrade packages.
- Streamlit model pages are registered in `chrono_app.py`.
- All forecasting methods share the same holdout evaluation and result-comparison
  workflow.
- ARIMA and SARIMA already include the comprehensive Box–Jenkins pipeline,
  transformations, automatic/manual order selection, statistical diagnostics, and
  inverse transformation.
- Every model page has practical method information and scholarly material through
  the `!` and `?` popovers.
- At the time this handoff was written, the complete suite contained 32 passing tests:

  ```powershell
  .\.venv\Scripts\python.exe -m unittest discover -s tests -v
  ```

- Preserve existing worktree changes. Do not reset, revert, commit, or perform other
  Git operations unless the user explicitly requests them.

## Does every method currently have its own Python file?

The answer is **partly**:

- **Yes:** every selectable forecasting method has a separate file under `method/`.
- **No:** those files are only Streamlit page wrappers. For example,
  `method/Statistical Models/1_ARIMA.py` only imports `render_model_page` and calls it
  with the model ID.
- The actual implementations are currently collected in `chrono_stream/models.py`.
- Most model-specific parameter widgets are collected in
  `chrono_stream/ui.py::_model_parameters`.
- Practical method metadata and APA references are collected in
  `chrono_stream/method_info.py`.
- Long literature reviews are collected in `chrono_stream/literature_reviews.py`.
- ARIMA/SARIMA share the separate, appropriately specialized
  `chrono_stream/arima_pipeline.py`.

Therefore, a page file exists for every method, but a self-contained implementation
module does not. Troubleshooting one method can currently require searching several
large shared files. The next architectural task should correct this before the method
catalog grows substantially.

## Agreed user-facing changes

### 1. Move the STL/X-11-inspired method

The current page runs an STL decomposition forecast and is not an implementation of
official Census X-11. It does not belong beside ARIMA and SARIMA merely because all
three concern time series.

Create a navigation group named **Decomposition & Seasonal Adjustment** and move the
page there. The group can eventually contain:

- STL Decomposition Forecast
- Classical decomposition
- MSTL decomposition/forecast
- A genuine X-11 or X-13 implementation, if one is ever added

During this move:

- Rename the internal model ID from the misleading `x11` to `stl` or
  `stl_decomposition_forecast`.
- Rename the page to **STL Decomposition Forecast (X-11-inspired)**.
- Keep the explicit statement that Chrono Stream does not execute Census X-11/X-13.
- Update navigation, overview groupings, registries, research keys, saved-result IDs,
  tests, and both README files together.
- If MSTL is added later, remember that decomposition alone is not a forecast. The
  downstream method used to forecast and recombine its components must be declared.

If a new navigation group is temporarily undesirable, the acceptable fallback is to
place the STL page under **Smoothing**, not **Statistical**.

### 2. Make the `!` and `?` triggers readable

Streamlit adds a chevron to every popover trigger. A one-character label currently
collides visually with that chevron in some layouts.

Preferred behavior:

- Keep `!` for practical method/test information.
- Keep `?` for scholarly reviews, references, and downloads.
- Give both trigger buttons a larger fixed/minimum width and sufficient right padding
  for the chevron (approximately `3.25rem`–`3.5rem` is a reasonable starting point).
- Scope any CSS to the method-reference action container; do not globally resize every
  Streamlit popover.
- If symbols alone remain unclear after the width change, use `! Info` and
  `? Research` as the accessible fallback.
- Verify the closed and open states at desktop width and approximately 430 px mobile
  width.

### 3. Additional forecasting methods

Add methods in deliberate stages rather than all at once.

#### First priority: essential baselines

1. **Naive forecast** — repeat the last observed value.
2. **Seasonal naive forecast** — repeat the last value from the same seasonal phase.
3. **Drift forecast** — extrapolate the average change between the first and last
   observations.

These are essential comparators. A complex model is not useful merely because it has
a low error; it should improve on an appropriate simple baseline.

#### Second priority: interpretable regression and trees

4. **Lagged linear regression** — regress the next value on lagged values and optional
   past-only rolling/calendar features.
5. **Regularized lag regression** — Ridge, Lasso, and/or Elastic Net. These may be one
   conceptual page with a penalty selector if they share one model contract.
6. **CART regression tree** — one interpretable decision tree for a continuous target.
7. **Random Forest regression** — an ensemble of randomized regression trees.

Important terminology and behavior:

- CART and “Decision Tree” should not be duplicate pages. CART is a decision-tree
  methodology; name the forecasting page **CART / Decision Tree Regression**.
- Regression is a family, not one sufficiently precise method name. Distinguish
  deterministic trend projection from lagged regression.
- Tree and ordinary machine-learning regressors do not consume a raw time series
  directly. Build supervised rows from shifted lags, rolling summaries, seasonal lags,
  and calendar features known at prediction time.
- Tree forecasts are piecewise constant and generally extrapolate trends poorly beyond
  the target values represented during training. Present them as nonlinear lag models,
  not universal replacements for statistical time-series models.
- Multi-step behavior must be explicit: recursive, direct, or multi-output. The current
  project commonly uses recursive forecasting, so accumulated forecast error must be
  disclosed.

#### Third priority: classical and specialized forecasting

8. **Theta method** — a lightweight classical univariate method and useful benchmark.
9. **Automatic ETS** — select among appropriate error, trend, damping, and seasonal
   structures rather than duplicating only the existing manual smoothing pages.
10. **Croston-family intermittent-demand methods** — Croston, SBA, and/or TSB for
    sparse, zero-heavy demand.
11. **TBATS** — for long, non-integer, or multiple seasonal cycles.
12. **MSTL-based forecast** — for multiple seasonal patterns, with an explicitly named
    forecasting rule for the decomposed components.

Before adding Croston-family methods, revisit metrics. MAPE is unsuitable around zeros;
consider adding MASE, RMSSE, and/or WAPE to the shared comparison contract.

#### Optional later methods

- Support Vector Regression
- Extra Trees
- k-nearest-neighbor regression
- N-BEATS
- Temporal Convolutional Network

N-BEATS and TCN should remain optional until the lighter classical and tree baselines
are complete. Chrono Stream already includes LSTM and 1D CNN, and another neural page
adds substantial test and installation cost.

## Target architecture: one method-facing module per method

The target rule is:

> One registered model ID and navigation method should have one clearly named
> method-facing Python module. Shared mathematical engines remain shared and must not
> be copied between methods.

A suitable structure is:

```text
chrono_stream/
  core/
    contracts.py              Shared MethodSpec and forecast-result types
    evaluation.py             Holdout/refit workflow and metrics
    intervals.py              Shared interval helpers
    registry.py               Single method registry
  methods/
    baselines/
      naive.py
      seasonal_naive.py
      drift.py
    smoothing/
      moving_average.py
      weighted_moving_average.py
      simple_exponential.py
      holt.py
      holt_winters.py
      theta.py
      automatic_ets.py
    decomposition/
      stl.py
      mstl.py
    statistical/
      arima.py
      sarima.py
      box_jenkins.py          Shared ARIMA/SARIMA engine or facade
      croston.py
      tbats.py
    machine_learning/
      lagged_linear.py
      regularized_regression.py
      cart.py
      random_forest.py
      xgboost.py
    neural/
      lstm.py
      cnn.py
    trend/
      linear.py
      quadratic.py
      exponential.py
      logarithmic.py
  research/
    types.py                  Shared reference/test dataclasses
    statistical_tests.py     Reusable tests and decision aids
method/
  ...                         Thin Streamlit page wrappers only
```

The exact folder names can change, but the ownership rules should not.

### What each method module should own

Each simple method module should expose one method specification containing or linking
to:

- Stable `model_id`, display name, icon, and navigation group
- Forecast handler
- Parameter defaults, validation, and method-specific parameter UI/schema
- Minimum sample and data-domain requirements
- Automatic and manual controls where meaningful
- Multi-step strategy
- Native or approximate interval capability
- Practical background, operation, appropriate use, and limitations
- Scholarly overview, literature review, and APA references
- Applicable statistical-test/decision-aid keys
- Reproducibility settings such as a random seed

A central `MethodSpec`/registry should replace parallel mappings and long `if/elif`
chains. `chrono_app.py`, the UI, forecasting dispatcher, overview, and research popovers
should read from the same registry so a model cannot be added to one place and omitted
from another.

### Complex methods and shared code

“One module per method” does not mean duplicating hundreds of lines:

- `arima.py` and `sarima.py` should each provide their own method specification, but
  continue to call one shared Box–Jenkins engine.
- LSTM and CNN may call shared neural preprocessing/training utilities.
- Linear, quadratic, exponential, and logarithmic trend modules may call shared curve
  and output helpers.
- CART, Random Forest, and XGBoost should share leakage-safe lag-feature construction.
- Shared output, interval, metric, date, and validation helpers belong under `core/`.

This gives every method a clear troubleshooting entry point without copying logic that
must remain consistent.

### Safe migration order

1. Add `MethodSpec`, shared contracts, and a registry without changing behavior.
2. Move one lightweight method, such as Moving Average, into the new structure.
3. Keep the old public `forecast_model` and `evaluate_and_forecast` functions as thin
   compatibility facades while migrating callers.
4. Move the remaining lightweight smoothing and trend methods one at a time, running
   tests after every group.
5. Move Prophet, XGBoost, and neural methods while preserving lazy imports.
6. Give ARIMA and SARIMA separate method specifications but retain their shared tested
   pipeline.
7. Move and rename the STL method and its navigation group.
8. Move parameter rendering out of the giant `_model_parameters` branch and into the
   corresponding method specification/module.
9. Consolidate practical and scholarly metadata through the registry only after the
   forecasting behavior remains green.
10. Add new forecasting methods after the refactor is stable.

Do not combine this refactor with silent mathematical changes to existing forecasts.
Structural moves should preserve behavior; algorithm improvements should be separate,
reviewable changes with dedicated tests.

## Checklist for every new forecasting method

### A. Method identity and taxonomy

- Give the method a unique, stable ID.
- Decide whether it is a forecast model, decomposition, seasonal-adjustment procedure,
  diagnostic, or decision aid.
- Do not create duplicate pages for aliases or variants of the same method.
- Do not name an approximation as though it implements the original method.
- Identify the original or method-defining publication and later sources needed to
  explain the implementation actually used.

### B. Data contract

- State whether the method requires regular spacing, positivity, multiple seasonal
  cycles, a minimum history, or external regressors.
- Validate all requirements before fitting and show actionable errors.
- Never silently reinterpret an irregular series as regularly spaced.
- External-regressor models require known future regressor values; do not invent them.

### C. Leakage prevention

- Split the outer holdout before fitting transformations, scalers, features, or model
  parameters.
- Generate lag and rolling features using past observations only; rolling statistics
  must be shifted before becoming predictors.
- Fit scalers, Box–Cox parameters, encoders, feature selectors, and hyperparameters on
  the current training partition only.
- Use expanding/rolling time splits, never shuffled random train/test splits.
- Keep any tuning CV inside the pre-holdout training data.
- Refit the complete pipeline independently on all observations for the future forecast.

### D. User control

- Provide sensible automatic defaults where justified.
- Let users disable automatic selection and enter important parameters manually.
- Clearly identify which controls affect fitting, diagnostics, or only display.
- Do not hide failed candidates or silently substitute a different model.

### E. Shared result contract

Every forecast handler must return compatible:

- Fitted values aligned to the original observations
- Future point forecasts
- Lower and upper intervals, or an explicit statement that calibrated intervals are
  unavailable
- Finite numeric outputs of the requested horizon
- Model details sufficient to reproduce the run
- The same outer holdout evaluation used by other methods

Continue reporting MAE, RMSE, MAPE, and sMAPE until the metric contract is deliberately
expanded. Do not change metric meaning for only one method.

### F. Forecast intervals

- Prefer model-native predictive intervals when available.
- Label residual-based intervals as approximations.
- Random Forest/CART do not automatically provide ordinary 95% statistical intervals;
  use a documented quantile, bootstrap, or conformal procedure if calibrated intervals
  are claimed.
- Apply inverse transformations to both forecasts and intervals correctly.
- Distinguish a conditional median from an original-scale conditional mean after a
  nonlinear transformation.

### G. Diagnostics and statistical language

- Only genuine hypothesis tests receive H0, H1, a statistic, reference distribution,
  alpha/critical-value decision rule, and reject/fail-to-reject interpretation.
- Never invent H0/H1 for feature importance, information criteria, convergence flags,
  cross-validation, residual plots, or other decision aids.
- For machine-learning methods, emphasize rolling-origin validation, residual behavior,
  stability, and leakage-safe feature importance rather than fictional coefficient
  significance.
- If OLS coefficient tests are exposed, account for time-series error assumptions; do
  not present ordinary iid standard errors as automatically valid.
- Keep “fail to reject H0”; do not write “accept H0.”

### H. Scholarly and practical information

For each method:

- `!` must explain the background, how it works, the exact Chrono Stream
  implementation, appropriate use, limitations, and practical test decisions.
- `?` must provide the scholarly overview, complete literature review, clean APA list,
  test literature where applicable, and TXT downloads.
- Cite original or explicitly method-defining work, not a paper that merely mentions
  the method.
- Use multiple sources when the implementation combines multiple ideas.
- State exactly when Chrono Stream differs from the cited algorithm.
- Avoid UI meta-commentary such as “Why cited,” “ready to copy,” source audits, or prose
  explaining that the prose is scholarly.

### I. Reproducibility and performance

- Fix and report random seeds for stochastic models where practical.
- Record selected hyperparameters, feature lags, seasonal periods, transformations,
  and fitting strategy in model details.
- Lazy-import heavy optional libraries inside the relevant method.
- Do not make normal app startup import TensorFlow, Prophet, or another heavy backend.
- Bound automatic searches and warn users when a configuration can be expensive.
- Clear backend sessions/resources after neural training where applicable.

### J. Tests

Add or update tests for:

- Minimum and invalid data requirements
- Forecast length, alignment, and finite outputs
- Automatic and manual parameter paths
- Transformation/scaling round trips where applicable
- Holdout and feature leakage
- Multi-step recursion/direct strategy
- Reproducibility for seeded methods
- Correct result metadata
- Streamlit page rendering with and without loaded data
- Comparison-dashboard compatibility
- Method-information and literature completeness
- Correct classification of formal tests versus decision aids

Then run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe -m compileall -q chrono_stream chrono_app.py method tests
.\.venv\Scripts\python.exe -m pip check
git diff --check
```

### K. Navigation and documentation

- Add the method page to the correct group in `chrono_app.py`.
- Update the App Overview, main README, this handoff if plans change, and project layout.
- Keep the page wrapper thin; implementation belongs in `chrono_stream/methods/`.
- Make `!` and `?` responsive and visually inspect their open/closed states.
- Ensure the method can be saved and compared with every existing result.

## Definition of done for one new method

A method is not complete merely because it produces a forecast. It is complete only
when:

1. It has one registered method-facing module and a thin Streamlit page.
2. Its data and parameter requirements are validated.
3. Backtesting and final refitting are leakage-safe.
4. Automatic and manual behavior are transparent.
5. Forecasts, intervals, metadata, and comparison metrics follow the shared contract.
6. Practical information and scholarly literature are complete and correctly
   attributed.
7. Formal tests and non-test decision aids are labeled correctly.
8. Focused tests and the complete suite pass.
9. README/navigation/project-layout documentation is current.
10. Desktop and mobile UI have been visually checked.

## Suggested next-session execution order

1. Re-run the existing suite and inspect `git status`.
2. Introduce the method registry/contract and migrate one lightweight method as a
   proof of structure.
3. Complete the behavior-preserving method-module refactor.
4. Move and rename STL into **Decomposition & Seasonal Adjustment**.
5. Widen the `!` and `?` popover triggers and visually verify them.
6. Add Naive, Seasonal Naive, and Drift baselines.
7. Add lagged linear/regularized regression.
8. Add CART, then Random Forest, using one shared leakage-safe lag-feature builder.
9. Add Theta and Automatic ETS.
10. Add intermittent-demand and complex-seasonality methods only after metrics and
    data requirements support them properly.

## Primary-source starting points for planned research

- X-11: Shiskin, Young, and Musgrave (1967), *The X-11 Variant of the Census
  Method II Seasonal Adjustment Program* —
  https://www.census.gov/content/dam/Census/library/working-papers/1967/adrm/shiskinyoungmusgrave1967.pdf
- CART: Breiman, Friedman, Olshen, and Stone (1984), *Classification and Regression
  Trees*.
- Random Forest: Breiman (2001), *Random Forests* —
  https://doi.org/10.1023/A:1010933404324
- Theta: Assimakopoulos and Nikolopoulos (2000), *The Theta Model: A Decomposition
  Approach to Forecasting* — https://doi.org/10.1016/S0169-2070(00)00066-2
- Croston: Croston (1972), *Forecasting and Stock Control for Intermittent Demands* —
  https://doi.org/10.1057/jors.1972.50
- TBATS: De Livera, Hyndman, and Snyder (2011), *Forecasting Time Series With Complex
  Seasonal Patterns Using Exponential Smoothing* —
  https://doi.org/10.1198/jasa.2011.tm09771
- MSTL: Bandara, Hyndman, and Bergmeir, *MSTL: A Seasonal-Trend Decomposition
  Algorithm for Time Series with Multiple Seasonal Patterns* —
  https://arxiv.org/abs/2107.13462

These are starting points, not permission to copy claims without checking the complete
source and the exact implementation used by Chrono Stream.
