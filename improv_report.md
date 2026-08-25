# Chrono Stream Improvement Report

## Scope

Audit of the main application code, forecasting implementations, method information, literature reviews, metrics, and explanatory UI text. Git commands and git configuration were not reviewed or changed.

The findings below were checked against the source and, where practical, reproduced with the project's `.venv`. The original audit did not modify application files; the resolution pass completed on 2026-08-26 independently rechecked each claim before changing code or documentation. `[DONE]` means the claim was accepted and resolved. `[REJECTED]` means the proposed finding was not supported by the implementation or primary evidence and was not applied.

## Priority Findings

### P0: User-visible crashes and invalid results

#### 1. [DONE] TBATS crashes when no trend is selected

- **File:** `chrono_stream/methods/statistical/tbats.py:133`
- **Code:** `"damping_phi": float(fitted_model.params.phi)`
- **Problem:** The `tbats` package sets `params.phi` to `None` when the fitted model has no trend. Automatic TBATS commonly selects a trendless seasonal model, and manual mode can disable the trend. Converting `None` to `float` raises `TypeError`.
- **Reproduction:** A 48-point seasonal series with `automatic=True` raised `TypeError: float() argument must be a string or a real number, not 'NoneType'`.
- **Suggested correction:** Store `float(phi)` only when `phi is not None`; otherwise store `None`.
- **Resolution:** Accepted after reproducing the trendless parameter state. `damping_phi` now preserves `None`, with regression coverage for a trendless TBATS fit.

#### 2. [REJECTED] Seasonal naive minimum-observation metadata is incorrect

- **Files:** `chrono_stream/methods/baselines/seasonal_naive.py:20-23`, `chrono_stream/ui.py:1066-1076`
- **Problem:** The handler requires `2 * seasonal_period` observations, but `SPEC.minimum_observations` remains at the default value `4`. The UI can therefore allow a configuration that later fails during backtesting.
- **Reproduction:** With 30 observations, holdout 20, and period 12, the UI gate passes but the method raises `ValueError: Seasonal naive needs at least two full seasons (24 observations)`.
- **Suggested correction:** Make the minimum observation requirement dependent on the selected seasonal period, or validate `len(data) - holdout >= 2 * period` before fitting.
- **Resolution:** Rejected after tracing the actual UI path. `minimum_observations=4` is the correct global minimum at the smallest allowed period (`m=2`). The parameter renderer receives the pre-holdout training length and caps `m` at `floor(training_length / 2)`, so the reported 30/20/12 UI configuration is changed to a valid maximum of 5 before fitting. Direct API use with an invalid period is already rejected by the handler.

#### 3. [DONE] Sum aggregation silently creates false zero observations

- **File:** `chrono_stream/data.py:143-156`
- **Problem:** Resampling with `.agg("sum")` turns empty periods into `0.0`. The code then reports no missing periods, skips interpolation, and trains on fabricated zeros. This can distort forecasts, seasonal benchmarks, MAPE, WAPE, and MASE.
- **Reproduction:** A monthly series with two missing months produced zero-valued rows for both gaps while `missing_periods_created` remained 0.
- **Suggested correction:** Preserve empty aggregate periods as `NaN`, for example by using a sum with `min_count=1`, then apply the selected missing-value policy.
- **Resolution:** Accepted and reproduced. Sum resampling now uses `min_count=1`, so empty periods remain missing and flow through the selected fill/drop policy; a regression test covers the former false-zero case.

### P1: Forecasting and interval correctness

#### 4. [DONE] Drift fitted values are not one-step drift forecasts

- **File:** `chrono_stream/methods/baselines/drift.py:19`
- **Code:** `fitted[1:] = values[0] + drift_per_step * np.arange(1, len(values))`
- **Problem:** This creates a straight line between the first and final observations. A random walk with drift's in-sample one-step fitted value should be `values[t-1] + drift_per_step`.
- **Effect:** The final fitted value is always exactly equal to the final actual, so the final residual is always zero. Residuals represent deviations from a deterministic endpoint line rather than one-step innovations. This contradicts the explanation that drift is not a least-squares line through levels.
- **Suggested correction:** Use `fitted[1:] = values[:-1] + drift_per_step`. Consider native random-walk-with-drift intervals using innovation variance times `sqrt(h)`.
- **Resolution:** Accepted. Historical fitted values are now causal one-step drift forecasts, and future bounds use the random-walk-with-estimated-drift variance `sigma² h(1 + h/(T-1))`.

#### 5. [DONE] Shared residual intervals are not valid 95% multi-step intervals

- **File:** `chrono_stream/core/intervals.py:23-39`
- **Code:** `scale = np.sqrt(1.0 + np.arange(1, len(forecast) + 1) / max(len(values), 1))`
- **Problem:** The `sqrt(1 + h/n)` factor barely widens intervals and is not the forecast-error accumulation for random walks, recursive models, exponential smoothing, or most machine-learning models.
- **Reproduction:** For a simulated random walk, naive interval coverage fell from approximately 0.944 at horizon 1 to approximately 0.460 at horizon 12. For n=60, the half-width increased only from about 1.96 to 2.13, whereas a random-walk standard deviation grows approximately as `sqrt(h)`.
- **Effect:** Results are labelled `Lower 95%` and `Upper 95%` despite materially lower coverage.
- **Suggested correction:** Use method-specific predictive variance, analytic or simulation-based forecast intervals, or rolling-origin empirical residual quantiles. Until then, avoid presenting these as nominal 95% intervals and surface the approximate method visibly.
- **Resolution:** Accepted. Naive, seasonal-naive, and drift now have method-specific Gaussian predictive intervals. The shared fallback is a constant descriptive in-sample residual band with no nominal-coverage claim, generic table/CSV columns no longer say `95%`, and the UI displays the method and warning.

#### 6. [DONE] Exact-interpolating ML models produce zero-width intervals

- **Files:** `chrono_stream/core/intervals.py:29-39`; defaults in `chrono_stream/methods/machine_learning/knn.py`, `cart.py`, and `extra_trees.py`
- **Problem:** kNN with distance weights, unrestricted CART, and default Extra Trees can interpolate their training targets exactly. The in-sample residual standard deviation is then zero, producing zero-width intervals that are still labelled approximate 95% intervals.
- **Suggested correction:** Use out-of-sample or rolling-origin residuals for interval estimation, or mark intervals unavailable when residual variance is degenerate.
- **Resolution:** Accepted. The shared fallback now returns unavailable (`NaN`) bounds and explicit metadata when usable residual variance is numerically degenerate; it no longer displays a zero-width pseudo-interval.

#### 7. [DONE] SVR target is not scaled

- **File:** `chrono_stream/methods/machine_learning/support_vector.py:55-67`
- **Problem:** Only the lag features are standardized. The target remains in original units, so `epsilon` and `C` operate on raw target magnitudes. On a series around 8.4 million, the manual default `epsilon=0.1` is effectively zero and the forecast became approximately 7.12 million versus a final observation around 8.42 million.
- **Effect:** Nearly every observation becomes a support vector and the documented sparsity interpretation is not useful.
- **Suggested correction:** Scale the target inside the training pipeline and inverse-transform predictions. Make epsilon candidates relative to target dispersion.
- **Resolution:** Accepted. SVR now uses a fold-local `TransformedTargetRegressor` around the feature-scaling pipeline, inverse-transforms predictions, and documents epsilon in training-target standard-deviation units. Scale-equivariance is regression-tested.

#### 8. [DONE] ARCH LM `ddof` handling is described incorrectly and can make the test too lenient

- **Files:** `chrono_stream/arima_pipeline.py:897-898`, `chrono_stream/statistical_tests.py:329-334`
- **Problem:** `het_arch(..., ddof=model_df)` does not change the chi-square degrees of freedom. statsmodels instead computes the statistic as `(nobs - ddof) * R²`, while retaining degrees of freedom equal to the number of lags. The documentation says the code adjusts degrees of freedom for fitted AR/MA terms, which is inaccurate.
- **Effect:** The statistic and p-value are reduced/inflated. With too few residuals, the statistic can become negative and the p-value becomes 1.0, causing a guaranteed pass.
- **Suggested correction:** Use the conventional unadjusted statistic, or document the exact scaling and reject cases where the effective sample size is insufficient.
- **Resolution:** Accepted in part after checking the statsmodels implementation and documentation. The recommended `ddof=p+q+P+Q` correction is retained, but the app now documents the exact `(n_aux-ddof)R²` rescaling and unchanged chi-square lag degrees of freedom, and returns the diagnostic as unavailable/failing when `n_aux <= ddof`.

#### 9. [DONE] Theta default seasonal interval variance uses the wrong series

- **File:** `chrono_stream/methods/smoothing/theta.py:146`
- **Problem:** With the default non-MLE path, statsmodels estimates the variance through a SARIMAX(0,1,1) refit on the raw seasonal series rather than the deseasonalized series used by the Theta fit.
- **Effect:** On strongly seasonal data, the estimated standard deviation can be several times too large, materially widening intervals.
- **Suggested correction:** Estimate variance from deseasonalized residuals, use the MLE path where appropriate, or disclose the raw-series variance source.
- **Resolution:** Accepted. The non-MLE path now estimates innovation variance through an ARIMA(0,1,1)-with-drift refit on the same deseasonalized series used by Theta and records that source; the MLE path continues to use its fitted variance.

#### 10. [DONE] Manual Holt and Holt-Winters permit inadmissible smoothing parameters

- **Files:** `chrono_stream/methods/smoothing/holt.py:29-34`, `holt_winters.py:45-51`
- **Problem:** Optimizer bounds that normally enforce relationships such as `beta <= alpha` and `gamma <= 1 - alpha` are bypassed when `optimized=False`.
- **Suggested correction:** Validate the manual parameters explicitly or constrain the UI controls.
- **Resolution:** Accepted. Manual parameters are now all-or-none and explicitly checked (`beta <= alpha`, `gamma <= 1-alpha`, valid alpha/phi ranges); the UI sliders enforce the same bounds.

## P1: Evaluation and comparison claims

#### 11. [DONE] Saved models do not necessarily share the same holdout

- **Files:** `chrono_stream/ui.py:1073-1076`; claims in `chrono_stream/metric_info.py:126`, `:155`, `:305`, `:348`, `method/1_App Overview.py:13-14`, and `method/4_Result Comparison and Forecasting.py:55`
- **Problem:** The UI clamps holdout independently according to each method's `minimum_observations`, which ranges from 4 to 24. A requested holdout of 20 can therefore become 18 for lagged regression or 6 for MSTL+ETS.
- **Effect:** Cross-model RMSE and other holdout metrics are not always directly comparable, despite documentation claiming an identical/shared holdout.
- **Suggested correction:** Reject a comparison unless one holdout is valid for every selected method, or explicitly state that each method can use its own holdout and avoid ranking them as identical-window results.
- **Resolution:** Accepted. A configured holdout is no longer silently reduced per method; an incompatible fit stops with an actionable error. The comparison page also refuses to rank legacy/stale saved results with mixed holdout lengths.

#### 12. [DONE] AICc silently excludes ETS candidates

- **Files:** `chrono_stream/methods/smoothing/automatic_ets.py:69`, `:140`; documentation at `method_info.py:1785` and `literature_reviews.py:209`
- **Problem:** Non-finite AICc values are converted to `None` and candidates are skipped. On short seasonal series, many trend/seasonal candidates therefore disappear without an exclusion explanation.
- **Suggested correction:** Record a per-candidate exclusion reason and warn when AICc is undefined for a substantial part of the search grid.
- **Resolution:** Accepted. Every attempted candidate now records selection eligibility and a specific exclusion reason. A summary count is retained, and a visible warning appears when at least 25% of fitted candidates have an undefined selected criterion.

#### 13. [DONE] ACF significance flags use a flat threshold at every lag

- **File:** `chrono_stream/arima_pipeline.py:445-462`
- **Problem:** All lags use `1.96 / sqrt(n)`, which is only a simple white-noise reference and does not account for Bartlett lag-dependent variance. These flags feed order-selection logic.
- **Suggested correction:** Use an appropriate Bartlett approximation or label the result as a heuristic reference rather than a 95% significance test.
- **Resolution:** Accepted using the report's labeling alternative. The fixed band remains an order-identification heuristic, but records, controls, charts, tooltips, and handbook text now call it a pointwise white-noise reference rather than a universal significance test. Exploratory ACF separately uses Bartlett intervals.

## P2: Method explanation mismatches

- [DONE] `method_info.py:912`, `:947`, `:982`: "estimated initial level/states" is only true when smoothing parameters are optimized. Manual mode uses statsmodels heuristic initial states. The documentation now distinguishes optimized from heuristic initialization.
- [DONE] `literature_reviews.py:66`: says Holt-Winters supports additive trend, but the code accepts multiplicative trend too. The review now documents both.
- [REJECTED] `literature_reviews.py:57`: describes damping phi as any value between 0 and 1, while the UI/optimizer restricts it to approximately 0.80-0.995. The mathematical statement is correct; the narrower UI range is a deliberate selectable subset, while direct validation permits the full admissible interval.
- [DONE] `method_info.py:1148`: says STL uses the last `max(2m, 8)` trend observations, but the code clamps this to `min(len(values), max(2m, 8))`. The clamp is now stated.
- [DONE] `method_info.py:1754` and `theta.py:171`: promise causal Theta fitted values, but at n=24 and period=12 the implementation produces no finite fitted values. Result metadata and documentation now disclose fitted-value availability at the two-cycle minimum.
- [DONE] `theta.py:113` and `automatic_ets.py:87`: regular-date validation raises an error mentioning "Lag-based regression" for Theta and ETS. The shared error now refers generically to the forecasting method.
- [DONE] `mstl_ets.py:266`: UI slider allows iterations 1-5 while validation accepts 1-10. The UI now exposes 1-10.
- [DONE] `literature_reviews.py:97`: calls the STL intervals "empirical"; they are approximate normal-theory residual bands based on `1.96 * residual_sd * scale`. The text now calls the display a non-calibrated normal-reference residual summary.
- [DONE] `literature_reviews.py:235`: says the chosen time origin can create apparent curvature in quadratic regression. Polynomial least squares is invariant to a shift of the time origin when refitted; training-window choice, outliers, seasonality, or structural breaks are valid concerns instead. The review now makes that distinction.
- [DONE] `literature_reviews.py:233`: for the fitted quadratic curve, first differences are exactly linear and second differences are exactly constant, not merely approximately so. The wording is now exact.
- [DONE] `literature_reviews.py:75`: says Chrono Stream exposes its own search instead of the Hyndman-Khandakar framework, but the Stepwise path calls `pmdarima.auto_arima(stepwise=True)`. The review now separates pmdarima candidate generation from Chrono Stream's later diagnostics and ranking.
- [REJECTED] `method_info.py:1033`: says the final ARIMA model is independently retuned on all observations; the selected full-partition fit is reused and no separate retuning stage exists. Tracing `evaluate_and_forecast` shows two independent pipeline calls: one on pre-holdout training data and another on all observations, and the latter reruns transformation, differencing, candidate generation, diagnostics, and selection.
- [DONE] `box_jenkins.py:360-362` and `arima_pipeline.py:950-960`: the Forecast-oriented UI renders an ARCH requirement toggle, but that policy hardcodes heteroskedasticity as non-mandatory, silently ignoring the toggle. Forecast-oriented policy now honors the toggle.

## P2: Citation and literature issues

### [DONE] SVR attribution

- **Files:** `chrono_stream/method_info.py:554`, `:1632`; `chrono_stream/literature_reviews.py:177`
- **Problem:** The epsilon-insensitive loss and epsilon-SVR should be attributed to Vapnik (and Vapnik, Golowich & Smola), not described as developed by Smola & Schölkopf or introduced by Drucker et al. Drucker et al. is an early application/comparison paper; Smola & Schölkopf (2004) is a tutorial.
- **Suggested correction:** Cite Vapnik (1995) or Vapnik, Golowich & Smola (1997) for the objective and describe Drucker et al. as an early regression application.
- **Resolution:** Added the foundational Vapnik sources and recast Drucker et al. as an early application and Smola and Schölkopf as a later tutorial.

### [REJECTED] ETS attribution

- **Files:** `chrono_stream/method_info.py:1777-1780`, `chrono_stream/literature_reviews.py:205`
- **Problem:** The ETS(·,·,·) notation, AICc-based automatic selection, and the expanded 30-model grid including damped multiplicative trend are attributed too broadly to Hyndman et al. (2002). The 2002 work is an important state-space/likelihood foundation, but the later 2008 framework is the better source for the notation and automatic method family used here.
- **Suggested correction:** Attribute the 2002 state-space foundation separately and cite Hyndman, Koehler, Ord & Snyder (2008) for the ETS notation, expanded taxonomy, AICc, and automatic forecasting framework.
- **Resolution:** Rejected as overstated after checking the primary 2002 paper: it explicitly presents the state-space taxonomy, likelihood, information-criterion model selection, automatic forecasting, and prediction intervals. The existing text already cites the 2008 book separately for comprehensive estimation, admissibility, simulation, and interval treatment.

### [DONE] Generalized Theta citation

- **Files:** `chrono_stream/methods/smoothing/theta.py:209-211`, `method_info.py:1747`
- **Problem:** The UI exposes theta values up to 20, but the listed A&N (2000) reference defines the conventional theta=2 method. The generalized weighting is associated with later work, including Fioruci et al. (2015).
- **Suggested correction:** Add the generalization reference or restrict the control to theta=2.
- **Resolution:** Added Fiorucci et al. (2016) and explicitly distinguishes conventional theta=2 from the generalized/optimized family in the UI and method material.

### [DONE] Pearson citation overreach

- **Files:** `method_info.py:721`, `:808`
- **Problem:** Pearson (1905) provides random-walk terminology/lineage, but does not establish the conditional-mean naive forecast or the drift rule.
- **Suggested correction:** Cite Hyndman & Athanasopoulos for the forecasting equations and retain Pearson only for the historical random-walk lineage.
- **Resolution:** Pearson is now limited to historical lineage; the modern naive and drift forecast equations are attributed separately.

### [DONE] sMAPE and M3

- **File:** `chrono_stream/metric_info.py:228-232`
- **Problem:** The app uses `2|y-yhat|/(|y|+|yhat|)`, bounded 0-200%. The M3 literature is commonly associated with a signed-denominator variant, so saying the exact app formulation "was used" in M3 conflates different formulas.
- **Suggested correction:** State that the app uses an absolute-denominator variant related to, but not identical with, the M3 measure.
- **Resolution:** The metric material now states the exact absolute-denominator convention and its distinction from the signed-denominator M3 variant.

### [DONE] MASE wording

- **File:** `chrono_stream/metric_literature_reviews.py:38`
- **Problem:** The text says the denominator is an average one-step error. The application allows a seasonal lag, and the default metric scale period is the seasonal period, so the denominator can be `mean(abs(y_t - y_{t-m}))`.
- **Suggested correction:** Say "average absolute error of the lag-m naive benchmark" and distinguish the seasonal-lag variant from the original nonseasonal MASE definition.
- **Resolution:** Updated throughout to the lag-m benchmark and explicitly distinguishes `m=1` from seasonal scaling.

### [DONE] Reference-list consistency

- [DONE] `method_info.py:966`: removed the unused `WINTERS_1960` reference from double exponential smoothing.
- [DONE] `method_info.py:1391`: the N-BEATS prose now cites `BEN_TAIEB_2012` for direct multi-step strategy context.
- [DONE] `method_info.py:1432`: the TCN prose now cites `BOROVYKH_2017` for time-series convolutional forecasting lineage.
- [DONE] `method_info.py:901` versus `:927`: Brown (1956) is now described consistently with qualified historical wording.

## P2: UI and explanation issues

- [DONE] `method/3_Data Exploration.py:150`: ADF rejection is now described as evidence against the unit-root null under the test specification, not proof of stationarity.
- [DONE] `method/3_Data Exploration.py:105-109`: Exploration now displays both ACF and PACF with clearly qualified pointwise reference bands.
- [DONE] `literature_reviews.py:77`: the nonexistent named "advisory" UI-policy claim was removed; the text describes the controls that are actually available.
- [DONE] `literature_reviews.py:97`: the self-certifying wording was removed and the STL/X-11 distinction is explained directly.

## Verified areas that should not be unnecessarily changed

- TBATS uses the genuine `tbats` implementation, not a hand-rolled Fourier approximation.
- Croston, SBA, and TSB recursions match the published formulas, including SBA's `1 - alpha/2` correction and TSB's period-by-period probability update.
- N-BEATS implements the key doubly-residual backcast/forecast structure. The code is a compact generic variant, and the documentation appropriately discloses its simplifications.
- TCN uses causal dilated convolutions and residual blocks. It does not use weight normalization, but the documentation does not falsely claim the full benchmark architecture.
- CNN and LSTM implementation descriptions match the TensorFlow/Keras code and use causal rolling windows.
- No train/test leakage was found in the supervised feature construction or scaler fitting.
- Yeo-Johnson implementation and inverse transformation agree with SciPy to numerical precision.
- The cited papers for ADF, KPSS, Phillips-Perron, Ljung-Box, Box-Pierce, ARCH, Croston, SBA, TSB, STL, MSTL, and the main ARIMA diagnostics were generally correctly attributed.

## Resolution outcome

- Numbered findings: 12 accepted and completed; 1 rejected after implementation tracing.
- Additional method, citation, reference-list, and UI findings are individually marked above.
- Verification: `python -m unittest discover -s tests -v` completed with all 77 tests passing.
- Git configuration and unrelated pre-existing working-tree changes were not altered.

## Note for the next session

The resolution pass is complete. Use the inline `[DONE]` and `[REJECTED]` labels as the decision record; rejected items include the reason they were not applied.
