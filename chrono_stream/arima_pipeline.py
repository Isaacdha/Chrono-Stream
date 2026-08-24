"""Auditable Box-Jenkins pipelines shared by the ARIMA and SARIMA pages."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import math
import warnings
from typing import Any, Iterable

import numpy as np


class NoEligibleModelError(ValueError):
    """Raised when strict diagnostics reject every fitted candidate."""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


def _finite_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError):
        return None
    return converted if math.isfinite(converted) else None


def _normalise_name(value: Any) -> str:
    return str(value).strip().lower().replace("_", " ").replace("-", " ")


def _yeo_johnson(values: np.ndarray, lmbda: float) -> np.ndarray:
    """Apply a Yeo-Johnson transform with an explicit lambda."""
    values = np.asarray(values, dtype=float)
    transformed = np.empty_like(values)
    nonnegative = values >= 0
    if abs(lmbda) < 1e-10:
        transformed[nonnegative] = np.log1p(values[nonnegative])
    else:
        transformed[nonnegative] = (
            np.power(values[nonnegative] + 1.0, lmbda) - 1.0
        ) / lmbda
    negative = ~nonnegative
    if abs(lmbda - 2.0) < 1e-10:
        transformed[negative] = -np.log1p(-values[negative])
    else:
        transformed[negative] = -(
            np.power(1.0 - values[negative], 2.0 - lmbda) - 1.0
        ) / (2.0 - lmbda)
    return transformed


def _inverse_yeo_johnson(values: np.ndarray, lmbda: float) -> np.ndarray:
    """Invert a Yeo-Johnson transform with an explicit lambda."""
    values = np.asarray(values, dtype=float)
    restored = np.empty_like(values)
    nonnegative = values >= 0
    with np.errstate(invalid="ignore", over="ignore"):
        if abs(lmbda) < 1e-10:
            restored[nonnegative] = np.expm1(values[nonnegative])
        else:
            restored[nonnegative] = (
                np.power(lmbda * values[nonnegative] + 1.0, 1.0 / lmbda) - 1.0
            )
        negative = ~nonnegative
        if abs(lmbda - 2.0) < 1e-10:
            restored[negative] = 1.0 - np.exp(-values[negative])
        else:
            restored[negative] = 1.0 - np.power(
                1.0 - (2.0 - lmbda) * values[negative],
                1.0 / (2.0 - lmbda),
            )
    return restored


@dataclass
class VarianceTransformer:
    """A fitted, serialisable, reversible variance transformation."""

    requested_method: str = "None"
    requested_lambda: float | None = None
    allow_shift: bool = False
    identity_tolerance: float = 0.10
    applied_method: str = "None"
    lmbda: float | None = None
    shift: float = 0.0
    automatic_identity: bool = False

    def _set_positive_shift(self, values: np.ndarray) -> None:
        minimum = float(np.min(values))
        if minimum > 0:
            self.shift = 0.0
        elif self.allow_shift:
            self.shift = 1.0 - minimum
        else:
            raise ValueError(
                f"{self.requested_method} requires positive values. Enable the "
                "automatic positive shift or select Yeo-Johnson."
            )

    def fit_transform(self, values: Any) -> np.ndarray:
        from scipy import stats

        source = np.asarray(values, dtype=float).reshape(-1)
        if not np.isfinite(source).all():
            raise ValueError("Variance transformation requires finite observations.")
        if np.std(source) <= np.finfo(float).eps:
            self.applied_method = "None"
            return source.copy()

        method = _normalise_name(self.requested_method)
        if method in {"none", "off", "disabled"}:
            self.applied_method = "None"
            return source.copy()

        if method in {"auto", "automatic"}:
            if np.all(source > 0):
                transformed, estimated_lambda = stats.boxcox(source)
                estimated_method = "Box-Cox"
            else:
                transformed, estimated_lambda = stats.yeojohnson(source)
                estimated_method = "Yeo-Johnson"
            self.lmbda = float(estimated_lambda)
            if abs(self.lmbda - 1.0) <= self.identity_tolerance:
                self.applied_method = "None"
                self.automatic_identity = True
                return source.copy()
            self.applied_method = estimated_method
            return np.asarray(transformed, dtype=float)

        if method in {"box cox", "boxcox"}:
            self.applied_method = "Box-Cox"
            self._set_positive_shift(source)
            shifted = source + self.shift
            if self.requested_lambda is None:
                transformed, estimated_lambda = stats.boxcox(shifted)
                self.lmbda = float(estimated_lambda)
            else:
                self.lmbda = float(self.requested_lambda)
                transformed = stats.boxcox(shifted, lmbda=self.lmbda)
            return np.asarray(transformed, dtype=float)

        if method in {"yeo johnson", "yeojohnson"}:
            self.applied_method = "Yeo-Johnson"
            if self.requested_lambda is None:
                transformed, estimated_lambda = stats.yeojohnson(source)
                self.lmbda = float(estimated_lambda)
                return np.asarray(transformed, dtype=float)
            self.lmbda = float(self.requested_lambda)
            return _yeo_johnson(source, self.lmbda)

        if method in {"log", "logarithm"}:
            self.applied_method = "Log"
            self._set_positive_shift(source)
            return np.log(source + self.shift)

        if method in {"square root", "sqrt"}:
            self.applied_method = "Square root"
            minimum = float(np.min(source))
            if minimum < 0:
                if not self.allow_shift:
                    raise ValueError(
                        "Square-root transformation requires non-negative values. "
                        "Enable the automatic shift or select Yeo-Johnson."
                    )
                self.shift = -minimum
            return np.sqrt(source + self.shift)

        raise ValueError(f"Unknown variance transformation: {self.requested_method}")

    def transform(self, values: Any) -> np.ndarray:
        from scipy.special import boxcox

        source = np.asarray(values, dtype=float)
        if self.applied_method == "None":
            return source.copy()
        if self.applied_method == "Box-Cox":
            return np.asarray(boxcox(source + self.shift, self.lmbda), dtype=float)
        if self.applied_method == "Yeo-Johnson":
            return _yeo_johnson(source, float(self.lmbda))
        if self.applied_method == "Log":
            return np.log(source + self.shift)
        if self.applied_method == "Square root":
            return np.sqrt(source + self.shift)
        raise ValueError(f"Cannot apply unknown transformation {self.applied_method}.")

    def inverse(self, values: Any) -> np.ndarray:
        from scipy.special import inv_boxcox

        source = np.asarray(values, dtype=float)
        restored = np.full(source.shape, np.nan, dtype=float)
        finite = np.isfinite(source)
        if not finite.any():
            return restored
        selected = source[finite]
        if self.applied_method == "None":
            inverted = selected
        elif self.applied_method == "Box-Cox":
            inverted = inv_boxcox(selected, float(self.lmbda)) - self.shift
        elif self.applied_method == "Yeo-Johnson":
            inverted = _inverse_yeo_johnson(selected, float(self.lmbda))
        elif self.applied_method == "Log":
            inverted = np.exp(selected) - self.shift
        elif self.applied_method == "Square root":
            inverted = np.where(selected >= 0, np.square(selected) - self.shift, np.nan)
        else:
            raise ValueError(
                f"Cannot invert unknown transformation {self.applied_method}."
            )
        restored[finite] = inverted
        return restored

    def details(self) -> dict[str, Any]:
        return {
            "requested_method": self.requested_method,
            "applied_method": self.applied_method,
            "lambda": _finite_float(self.lmbda),
            "shift": float(self.shift),
            "automatic_identity": self.automatic_identity,
        }


def _difference(values: np.ndarray, d: int, seasonal_d: int, period: int) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    for _ in range(seasonal_d):
        result = result[period:] - result[:-period]
    for _ in range(d):
        result = np.diff(result)
    return result


def _stationarity_results(
    values: np.ndarray,
    method: str,
    alpha: float,
    difference: int,
) -> tuple[bool, list[dict[str, Any]]]:
    from statsmodels.tsa.stattools import adfuller, kpss

    selected = _normalise_name(method)
    tests = ["ADF", "KPSS"] if "consensus" in selected else [method.upper()]
    records: list[dict[str, Any]] = []
    decisions: list[bool] = []
    for test in tests:
        normalised_test = _normalise_name(test)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if normalised_test == "adf":
                    statistic, p_value, used_lag, observations, critical, _ = adfuller(
                        values, autolag="AIC"
                    )
                    stationary = bool(p_value < alpha)
                    record = {
                        "difference": difference,
                        "test": "ADF",
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "used_lag": int(used_lag),
                        "observations": int(observations),
                        "critical_5_percent": _finite_float(critical.get("5%")),
                        "stationary": stationary,
                    }
                elif normalised_test == "kpss":
                    statistic, p_value, used_lag, critical = kpss(
                        values, regression="c", nlags="auto"
                    )
                    stationary = bool(p_value > alpha)
                    record = {
                        "difference": difference,
                        "test": "KPSS",
                        "statistic": float(statistic),
                        "p_value": float(p_value),
                        "used_lag": int(used_lag),
                        "observations": len(values),
                        "critical_5_percent": _finite_float(critical.get("5%")),
                        "stationary": stationary,
                    }
                elif normalised_test in {"pp", "phillips perron"}:
                    from pmdarima.arima import PPTest

                    p_value, should_difference = PPTest(alpha=alpha).should_diff(values)
                    stationary = not bool(should_difference)
                    record = {
                        "difference": difference,
                        "test": "Phillips-Perron",
                        "statistic": None,
                        "p_value": float(p_value),
                        "used_lag": None,
                        "observations": len(values),
                        "critical_5_percent": None,
                        "stationary": stationary,
                    }
                else:
                    raise ValueError(f"Unknown stationarity test: {method}")
        except (ValueError, np.linalg.LinAlgError) as exc:
            stationary = False
            record = {
                "difference": difference,
                "test": test,
                "statistic": None,
                "p_value": None,
                "used_lag": None,
                "observations": len(values),
                "critical_5_percent": None,
                "stationary": False,
                "error": str(exc),
            }
        records.append(record)
        decisions.append(stationary)
    return all(decisions), records


def _regular_difference_order(
    values: np.ndarray,
    params: dict[str, Any],
    seasonal_d: int,
    period: int,
) -> tuple[int, bool, list[dict[str, Any]]]:
    default_mode = "Automatic" if params.get("automatic", False) else "Manual"
    mode = _normalise_name(params.get("difference_mode", default_mode))
    test = str(params.get("differencing_test", "ADF + KPSS consensus"))
    alpha = float(params.get("stationarity_alpha", 0.05))
    maximum = max(0, int(params.get("max_d", 2)))
    seasonal_values = _difference(values, 0, seasonal_d, period)

    if mode in {"none", "disabled", "off"}:
        stationary, records = _stationarity_results(seasonal_values, test, alpha, 0)
        return 0, stationary, records
    if mode == "manual":
        selected = max(0, int(params.get("d", 0)))
        tested = _difference(seasonal_values, selected, 0, period)
        stationary, records = _stationarity_results(tested, test, alpha, selected)
        return selected, stationary, records

    history: list[dict[str, Any]] = []
    current = seasonal_values
    for difference in range(maximum + 1):
        if len(current) < 8 or np.std(current) <= np.finfo(float).eps:
            history.append(
                {
                    "difference": difference,
                    "test": "Constant/short-series rule",
                    "statistic": None,
                    "p_value": None,
                    "used_lag": None,
                    "observations": len(current),
                    "critical_5_percent": None,
                    "stationary": True,
                }
            )
            return difference, True, history
        stationary, records = _stationarity_results(current, test, alpha, difference)
        history.extend(records)
        if stationary:
            return difference, True, history
        if difference < maximum:
            current = np.diff(current)
    return maximum, False, history


def _acf_seasonal_difference_order(
    values: np.ndarray, period: int, maximum: int
) -> int:
    current = np.asarray(values, dtype=float)
    selected = 0
    while selected < maximum and len(current) >= 2 * period + 1:
        left = current[:-period]
        right = current[period:]
        if np.std(left) <= np.finfo(float).eps or np.std(right) <= np.finfo(float).eps:
            break
        correlation = float(np.corrcoef(left, right)[0, 1])
        threshold = max(0.30, 1.96 / math.sqrt(len(current)))
        if not math.isfinite(correlation) or correlation <= threshold:
            break
        current = current[period:] - current[:-period]
        selected += 1
    return selected


def _seasonal_difference_order(
    values: np.ndarray,
    params: dict[str, Any],
    seasonal: bool,
    period: int,
) -> tuple[int, list[dict[str, Any]]]:
    if not seasonal:
        return 0, []
    default_mode = "Automatic" if params.get("automatic", False) else "Manual"
    mode = _normalise_name(params.get("seasonal_difference_mode", default_mode))
    maximum = max(0, int(params.get("max_D", 1)))
    if mode in {"none", "disabled", "off"}:
        return 0, [{"method": "Disabled", "selected_D": 0}]
    if mode == "manual":
        selected = max(0, int(params.get("D", 0)))
        return selected, [{"method": "Manual", "selected_D": selected}]

    method = str(params.get("seasonal_differencing_test", "OCSB"))
    normalised_method = _normalise_name(method)
    if normalised_method in {"acf", "acf significance", "seasonal correlation"}:
        selected = _acf_seasonal_difference_order(values, period, maximum)
        return selected, [
            {
                "method": "Seasonal-lag ACF",
                "selected_D": selected,
                "maximum_D": maximum,
            }
        ]

    from pmdarima.arima import nsdiffs

    test = "ch" if normalised_method in {"ch", "canova hansen"} else "ocsb"
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            selected = int(nsdiffs(values, m=period, max_D=maximum, test=test))
    except (ValueError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            f"{method} could not estimate seasonal differencing for period {period}: {exc}"
        ) from exc
    return selected, [
        {
            "method": "Canova-Hansen" if test == "ch" else "OCSB",
            "selected_D": selected,
            "maximum_D": maximum,
            "seasonal_period": period,
        }
    ]


def _correlation_records(values: np.ndarray, maximum_lag: int) -> dict[str, Any]:
    from statsmodels.tsa.stattools import acf, pacf

    source = np.asarray(values, dtype=float)
    safe_maximum = min(maximum_lag, max(1, len(source) // 2 - 1), len(source) - 2)
    safe_maximum = max(1, safe_maximum)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        acf_values = np.asarray(acf(source, nlags=safe_maximum, fft=True), dtype=float)
        pacf_values = np.asarray(
            pacf(source, nlags=safe_maximum, method="ywm"), dtype=float
        )
    threshold = 1.96 / math.sqrt(len(source))

    def records(coefficients: np.ndarray) -> list[dict[str, Any]]:
        return [
            {
                "lag": lag,
                "value": float(value),
                "lower": -threshold,
                "upper": threshold,
                "significant": bool(lag > 0 and abs(value) > threshold),
            }
            for lag, value in enumerate(coefficients)
        ]

    return {
        "observations": len(source),
        "maximum_lag": safe_maximum,
        "confidence_level": 0.95,
        "acf": records(acf_values),
        "pacf": records(pacf_values),
    }


def _bounded_neighbours(values: Iterable[int], maximum: int) -> list[int]:
    selected = {0}
    if maximum >= 1:
        selected.add(1)
    for value in values:
        for candidate in (value - 1, value, value + 1):
            if 0 <= candidate <= maximum:
                selected.add(candidate)
    return sorted(selected)


def _guided_components(
    correlations: dict[str, Any],
    maximum_p: int,
    maximum_q: int,
    maximum_P: int,
    maximum_Q: int,
    period: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    significant_p = [
        int(row["lag"])
        for row in correlations["pacf"]
        if row["significant"] and int(row["lag"]) <= maximum_p
    ]
    significant_q = [
        int(row["lag"])
        for row in correlations["acf"]
        if row["significant"] and int(row["lag"]) <= maximum_q
    ]
    significant_P = [
        int(row["lag"]) // period
        for row in correlations["pacf"]
        if row["significant"]
        and int(row["lag"]) % period == 0
        and int(row["lag"]) // period <= maximum_P
    ]
    significant_Q = [
        int(row["lag"]) // period
        for row in correlations["acf"]
        if row["significant"]
        and int(row["lag"]) % period == 0
        and int(row["lag"]) // period <= maximum_Q
    ]
    return (
        _bounded_neighbours(significant_p, maximum_p),
        _bounded_neighbours(significant_q, maximum_q),
        _bounded_neighbours(significant_P, maximum_P),
        _bounded_neighbours(significant_Q, maximum_Q),
    )


def _deduplicate_candidates(
    candidates: Iterable[tuple[tuple[int, int, int], tuple[int, int, int, int]]],
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    return list(dict.fromkeys(candidates))


def _grid_candidates(
    d: int,
    seasonal_d: int,
    period: int,
    p_values: Iterable[int],
    q_values: Iterable[int],
    P_values: Iterable[int],
    Q_values: Iterable[int],
    maximum_order: int,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    candidates = []
    for p_order, q_order, seasonal_p, seasonal_q in product(
        p_values, q_values, P_values, Q_values
    ):
        if p_order + q_order + seasonal_p + seasonal_q > maximum_order:
            continue
        candidates.append(
            (
                (int(p_order), int(d), int(q_order)),
                (int(seasonal_p), int(seasonal_d), int(seasonal_q), int(period)),
            )
        )
    return _deduplicate_candidates(candidates)


def _stepwise_candidates(
    values: np.ndarray,
    params: dict[str, Any],
    seasonal: bool,
    period: int,
    d: int,
    seasonal_d: int,
) -> list[tuple[tuple[int, int, int], tuple[int, int, int, int]]]:
    from pmdarima import auto_arima

    criterion = str(params.get("criterion", "AICc")).lower()
    if criterion not in {"aic", "bic", "hqic"}:
        criterion = "aic"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted_models = auto_arima(
            values,
            start_p=0,
            start_q=0,
            max_p=max(0, int(params.get("max_p", 3))),
            max_q=max(0, int(params.get("max_q", 3))),
            d=d,
            max_d=d,
            start_P=0,
            start_Q=0,
            max_P=max(0, int(params.get("max_P", 1))) if seasonal else 0,
            max_Q=max(0, int(params.get("max_Q", 1))) if seasonal else 0,
            D=seasonal_d if seasonal else 0,
            max_D=seasonal_d if seasonal else 0,
            m=period if seasonal else 1,
            seasonal=seasonal,
            max_order=max(1, int(params.get("max_order", 6))),
            information_criterion=criterion,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=False,
            with_intercept="auto",
            return_valid_fits=True,
        )
    if not isinstance(fitted_models, (tuple, list)):
        fitted_models = [fitted_models]
    candidates = []
    for model in fitted_models:
        order = tuple(map(int, model.order))
        seasonal_order = (
            tuple(map(int, model.seasonal_order)) if seasonal else (0, 0, 0, 0)
        )
        candidates.append((order, seasonal_order))
    return _deduplicate_candidates(candidates)


def _trend_code(params: dict[str, Any], d: int, seasonal_d: int) -> str:
    selected = _normalise_name(params.get("trend", "Automatic"))
    if selected in {"none", "off", "disabled", "n"}:
        return "n"
    if selected in {"constant", "intercept", "c"}:
        return "c"
    if selected in {"linear", "linear trend", "time trend", "t"}:
        return "t"
    if selected in {"constant and linear", "ct"}:
        return "ct"
    return "c" if d + seasonal_d == 0 else "n"


def _fit_candidate(
    values: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
    maximum_iterations: int,
) -> Any:
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SARIMAX(
            values,
            order=order,
            seasonal_order=seasonal_order,
            trend=trend,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=maximum_iterations)


def _model_converged(fit: Any) -> bool:
    return bool((getattr(fit, "mle_retvals", {}) or {}).get("converged", True))


def _information_criteria(fit: Any) -> dict[str, float | None]:
    effective_n = int(fit.nobs) - int(getattr(fit, "loglikelihood_burn", 0))
    parameters = len(np.asarray(fit.params).reshape(-1))
    denominator = effective_n - parameters - 1
    aic = float(fit.aic)
    aicc = (
        aic + 2.0 * parameters * (parameters + 1) / denominator
        if denominator > 0
        else math.inf
    )
    return {
        "AIC": _finite_float(aic),
        "AICc": _finite_float(aicc),
        "BIC": _finite_float(fit.bic),
        "HQIC": _finite_float(fit.hqic),
    }


def _coefficient_diagnostics(
    fit: Any, alpha: float
) -> tuple[bool, list[dict[str, Any]]]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        names = list(getattr(fit, "param_names", []))
        parameters = np.asarray(fit.params, dtype=float).reshape(-1)
        standard_errors = np.asarray(fit.bse, dtype=float).reshape(-1)
        p_values = np.asarray(fit.pvalues, dtype=float).reshape(-1)
        confidence = np.asarray(fit.conf_int(alpha=alpha), dtype=float)
    records: list[dict[str, Any]] = []
    evaluated: list[bool] = []
    for index, parameter in enumerate(parameters):
        name = names[index] if index < len(names) else f"parameter_{index}"
        is_variance = name.lower() in {"sigma2", "variance"}
        p_value = _finite_float(p_values[index])
        significant = (
            None if is_variance else bool(p_value is not None and p_value < alpha)
        )
        if significant is not None:
            evaluated.append(significant)
        records.append(
            {
                "component": name,
                "coefficient": _finite_float(parameter),
                "standard_error": _finite_float(standard_errors[index]),
                "z_statistic": _finite_float(
                    parameter / standard_errors[index]
                    if standard_errors[index] != 0
                    else math.nan
                ),
                "p_value": p_value,
                "lower": _finite_float(confidence[index, 0]),
                "upper": _finite_float(confidence[index, 1]),
                "evaluated": not is_variance,
                "significant": significant,
            }
        )
    return all(evaluated), records


def _standardised_residuals(fit: Any) -> np.ndarray:
    try:
        residuals = np.asarray(
            fit.filter_results.standardized_forecasts_error[0], dtype=float
        ).reshape(-1)
    except (AttributeError, IndexError, TypeError):
        residuals = np.asarray(fit.resid, dtype=float).reshape(-1)
    burn = max(
        int(getattr(fit, "loglikelihood_burn", 0)),
        int(getattr(fit, "nobs_diffuse", 0)),
    )
    residuals = residuals[burn:]
    return residuals[np.isfinite(residuals)]


def _normality_diagnostic(
    residuals: np.ndarray, method: str, alpha: float
) -> dict[str, Any]:
    from scipy import stats

    selected = _normalise_name(method)
    if len(residuals) < 8:
        return {
            "method": method,
            "statistic": None,
            "p_value": None,
            "passed": False,
            "error": "At least 8 usable residuals are required.",
        }
    if selected in {"jarque bera", "jb"}:
        result = stats.jarque_bera(residuals)
        statistic, p_value = float(result.statistic), float(result.pvalue)
        passed = p_value > alpha
        return {
            "method": "Jarque-Bera",
            "statistic": statistic,
            "p_value": p_value,
            "passed": bool(passed),
        }
    if selected in {"shapiro", "shapiro wilk"}:
        tested = residuals if len(residuals) <= 5000 else residuals[:5000]
        result = stats.shapiro(tested)
        p_value = float(result.pvalue)
        return {
            "method": "Shapiro-Wilk",
            "statistic": float(result.statistic),
            "p_value": p_value,
            "passed": bool(p_value > alpha),
            "observations_tested": len(tested),
        }
    if selected in {"lilliefors", "lilliefor"}:
        from statsmodels.stats.diagnostic import lilliefors

        statistic, p_value = lilliefors(residuals, dist="norm")
        return {
            "method": "Lilliefors",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "passed": bool(p_value > alpha),
        }
    if selected in {"anderson", "anderson darling"}:
        result = stats.anderson(residuals, dist="norm")
        target = alpha * 100.0
        index = int(np.argmin(np.abs(np.asarray(result.significance_level) - target)))
        critical = float(result.critical_values[index])
        return {
            "method": "Anderson-Darling",
            "statistic": float(result.statistic),
            "p_value": None,
            "critical_value": critical,
            "tested_significance": float(result.significance_level[index]) / 100.0,
            "passed": bool(result.statistic < critical),
        }
    raise ValueError(f"Unknown residual normality test: {method}")


def _requested_lags(value: Any) -> list[int]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
        try:
            return sorted({int(part.strip()) for part in parts if part.strip()})
        except ValueError as exc:
            raise ValueError(
                "Diagnostic lags must be comma-separated integers."
            ) from exc
    return sorted({int(item) for item in value})


def _order_values(value: Any, label: str, maximum: int) -> list[int]:
    if isinstance(value, str):
        parts = value.replace(";", ",").split(",")
        try:
            selected = sorted({int(part.strip()) for part in parts if part.strip()})
        except ValueError as exc:
            raise ValueError(
                f"{label} must be a comma-separated list of integers."
            ) from exc
    elif isinstance(value, Iterable):
        selected = sorted({int(item) for item in value})
    else:
        selected = [int(value)]
    if not selected:
        raise ValueError(f"Enter at least one {label} value.")
    if any(item < 0 or item > maximum for item in selected):
        raise ValueError(f"Every {label} value must be between 0 and {maximum}.")
    return selected


def _diagnostic_lags(
    residual_count: int,
    model_df: int,
    seasonal: bool,
    period: int,
    requested: Any,
) -> list[int]:
    maximum = max(1, min(residual_count - 1, residual_count // 5))
    supplied = _requested_lags(requested)
    proposed = supplied or ([10, period, 2 * period] if seasonal else [10, 20])
    selected = sorted({min(maximum, lag) for lag in proposed if lag > 0})
    selected = [lag for lag in selected if lag > model_df]
    if not selected and maximum > model_df:
        selected = [maximum]
    return selected


def _white_noise_diagnostic(
    residuals: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    params: dict[str, Any],
    alpha: float,
    seasonal: bool,
    period: int,
) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    model_df = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
    lags = _diagnostic_lags(
        len(residuals),
        model_df,
        seasonal,
        period,
        params.get("ljung_box_lags"),
    )
    method = str(params.get("white_noise_test", "Ljung-Box"))
    if not lags:
        return {
            "method": method,
            "model_degrees_of_freedom": model_df,
            "lags": [],
            "passed": False,
            "error": "Too few residuals remain for a degrees-of-freedom-corrected test.",
        }
    box_pierce = _normalise_name(method) == "box pierce"
    table = acorr_ljungbox(
        residuals,
        lags=lags,
        model_df=model_df,
        boxpierce=box_pierce,
        return_df=True,
    )
    prefix = "bp" if box_pierce else "lb"
    rows = []
    for lag, row in table.iterrows():
        p_value = _finite_float(row[f"{prefix}_pvalue"])
        rows.append(
            {
                "lag": int(lag),
                "statistic": _finite_float(row[f"{prefix}_stat"]),
                "p_value": p_value,
                "passed": bool(p_value is not None and p_value > alpha),
            }
        )
    return {
        "method": "Box-Pierce" if box_pierce else "Ljung-Box",
        "model_degrees_of_freedom": model_df,
        "lags": rows,
        "passed": bool(rows and all(row["passed"] for row in rows)),
    }


def _heteroskedasticity_diagnostic(
    residuals: np.ndarray, model_df: int, alpha: float
) -> dict[str, Any]:
    from statsmodels.stats.diagnostic import het_arch

    if len(residuals) < 12:
        return {
            "method": "ARCH LM",
            "statistic": None,
            "p_value": None,
            "passed": False,
            "error": "At least 12 usable residuals are required.",
        }
    lags = max(1, min(10, len(residuals) // 5))
    try:
        statistic, p_value, f_statistic, f_p_value = het_arch(
            residuals, nlags=lags, ddof=model_df
        )
    except (ValueError, np.linalg.LinAlgError) as exc:
        return {
            "method": "ARCH LM",
            "statistic": None,
            "p_value": None,
            "passed": False,
            "error": str(exc),
        }
    return {
        "method": "ARCH LM",
        "lags": lags,
        "statistic": _finite_float(statistic),
        "p_value": _finite_float(p_value),
        "f_statistic": _finite_float(f_statistic),
        "f_p_value": _finite_float(f_p_value),
        "f_passed": bool(math.isfinite(f_p_value) and f_p_value > alpha),
        "passed": bool(math.isfinite(p_value) and p_value > alpha),
    }


def _root_diagnostic(fit: Any, tolerance: float) -> dict[str, Any]:
    ar_roots = np.asarray(getattr(fit, "arroots", []), dtype=complex).reshape(-1)
    ma_roots = np.asarray(getattr(fit, "maroots", []), dtype=complex).reshape(-1)
    ar_magnitudes = np.abs(ar_roots)
    ma_magnitudes = np.abs(ma_roots)
    ar_passed = bool(not len(ar_magnitudes) or np.all(ar_magnitudes > 1.0 + tolerance))
    ma_passed = bool(not len(ma_magnitudes) or np.all(ma_magnitudes > 1.0 + tolerance))
    return {
        "ar_magnitudes": [float(value) for value in ar_magnitudes],
        "ma_magnitudes": [float(value) for value in ma_magnitudes],
        "ar_stationary": ar_passed,
        "ma_invertible": ma_passed,
        "passed": ar_passed and ma_passed,
        "tolerance": tolerance,
    }


def _diagnostic_requirements(params: dict[str, Any]) -> dict[str, bool]:
    policy = _normalise_name(params.get("diagnostic_policy", "Advisory"))
    if policy in {"strict", "strict box jenkins", "box jenkins"}:
        return {
            "stationarity": True,
            "converged": True,
            "roots": True,
            "coefficients": True,
            "normality": True,
            "white_noise": True,
            "residual_mean": True,
            "heteroskedasticity": bool(params.get("require_heteroskedasticity", False)),
        }
    if policy in {"forecast", "forecast oriented"}:
        return {
            "stationarity": True,
            "converged": True,
            "roots": True,
            "coefficients": False,
            "normality": False,
            "white_noise": True,
            "residual_mean": False,
            "heteroskedasticity": False,
        }
    if policy == "custom":
        return {
            "stationarity": bool(params.get("require_stationarity", True)),
            "converged": True,
            "roots": bool(params.get("require_roots", True)),
            "coefficients": bool(params.get("require_significance", True)),
            "normality": bool(params.get("require_normality", True)),
            "white_noise": bool(params.get("require_white_noise", True)),
            "residual_mean": bool(params.get("require_residual_mean", True)),
            "heteroskedasticity": bool(params.get("require_heteroskedasticity", False)),
        }
    return {
        "stationarity": False,
        "converged": False,
        "roots": False,
        "coefficients": False,
        "normality": False,
        "white_noise": False,
        "residual_mean": False,
        "heteroskedasticity": False,
    }


def _candidate_diagnostics(
    fit: Any,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    params: dict[str, Any],
    seasonal: bool,
    period: int,
    stationarity_achieved: bool,
) -> dict[str, Any]:
    from scipy import stats

    alpha = float(params.get("diagnostic_alpha", 0.05))
    residuals = _standardised_residuals(fit)
    converged = _model_converged(fit)
    coefficient_passed, coefficients = _coefficient_diagnostics(fit, alpha)
    roots = _root_diagnostic(fit, float(params.get("root_tolerance", 1e-3)))
    normality = _normality_diagnostic(
        residuals, str(params.get("normality_test", "Jarque-Bera")), alpha
    )
    white_noise = _white_noise_diagnostic(
        residuals, order, seasonal_order, params, alpha, seasonal, period
    )
    model_df = order[0] + order[2] + seasonal_order[0] + seasonal_order[2]
    heteroskedasticity = _heteroskedasticity_diagnostic(residuals, model_df, alpha)
    mean_test = stats.ttest_1samp(residuals, popmean=0.0)
    residual_mean = {
        "method": "One-sample t-test",
        "mean": _finite_float(np.mean(residuals)),
        "statistic": _finite_float(mean_test.statistic),
        "p_value": _finite_float(mean_test.pvalue),
        "passed": bool(math.isfinite(mean_test.pvalue) and mean_test.pvalue > alpha),
    }
    requirements = _diagnostic_requirements(params)
    outcomes = {
        "stationarity": stationarity_achieved,
        "converged": converged,
        "roots": bool(roots["passed"]),
        "coefficients": coefficient_passed,
        "normality": bool(normality["passed"]),
        "white_noise": bool(white_noise["passed"]),
        "residual_mean": bool(residual_mean["passed"]),
        "heteroskedasticity": bool(heteroskedasticity["passed"]),
    }
    failures = [
        name.replace("_", " ")
        for name, required in requirements.items()
        if required and not outcomes[name]
    ]
    maximum_lag = min(int(params.get("acf_lags", 40)), max(1, len(residuals) // 2 - 1))
    residual_correlations = _correlation_records(residuals, maximum_lag)
    if len(residuals) >= 3:
        ordered = np.sort(residuals)
        theoretical = stats.norm.ppf((np.arange(len(ordered)) + 0.5) / len(ordered))
        qq_plot = [
            {"theoretical": float(x_value), "sample": float(y_value)}
            for x_value, y_value in zip(theoretical, ordered, strict=True)
        ]
    else:
        qq_plot = []
    return {
        "alpha": alpha,
        "requirements": requirements,
        "outcomes": outcomes,
        "eligible": not failures,
        "failures": failures,
        "coefficients": coefficients,
        "roots": roots,
        "normality": normality,
        "white_noise": white_noise,
        "heteroskedasticity": heteroskedasticity,
        "residual_mean": residual_mean,
        "residuals": [float(value) for value in residuals],
        "residual_correlations": residual_correlations,
        "qq_plot": qq_plot,
    }


def _candidate_summary(
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    criteria: dict[str, float | None],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    outcomes = diagnostics["outcomes"]
    white_noise = diagnostics["white_noise"]
    normality = diagnostics["normality"]
    coefficients = [row for row in diagnostics["coefficients"] if row.get("evaluated")]
    coefficient_p_values = [
        row["p_value"] for row in coefficients if row.get("p_value") is not None
    ]
    ljung_p_values = [
        row["p_value"]
        for row in white_noise.get("lags", [])
        if row.get("p_value") is not None
    ]
    return {
        "order": list(order),
        "seasonal_order": list(seasonal_order),
        **criteria,
        "CV RMSE": None,
        "CV MAE": None,
        "stationarity_passed": outcomes["stationarity"],
        "converged": outcomes["converged"],
        "stable_and_invertible": outcomes["roots"],
        "coefficients_significant": outcomes["coefficients"],
        "maximum_coefficient_p": max(coefficient_p_values)
        if coefficient_p_values
        else None,
        "normality_test": normality.get("method"),
        "normality_p": normality.get("p_value"),
        "normal_residuals": outcomes["normality"],
        "white_noise_test": white_noise.get("method"),
        "minimum_white_noise_p": min(ljung_p_values) if ljung_p_values else None,
        "white_noise": outcomes["white_noise"],
        "zero_residual_mean": outcomes["residual_mean"],
        "residual_mean_p": diagnostics["residual_mean"].get("p_value"),
        "constant_variance": outcomes["heteroskedasticity"],
        "eligible": diagnostics["eligible"],
        "failed_gate_count": len(diagnostics["failures"]),
        "failure_reasons": ", ".join(diagnostics["failures"]),
    }


def _criterion_value(record: dict[str, Any], criterion: str) -> float:
    value = record.get(criterion)
    return (
        float(value) if value is not None and math.isfinite(float(value)) else math.inf
    )


def _rolling_cv_score(
    source: np.ndarray,
    order: tuple[int, int, int],
    seasonal_order: tuple[int, int, int, int],
    trend: str,
    params: dict[str, Any],
    criterion: str,
) -> tuple[float | None, int, str | None]:
    """Score a fixed candidate on expanding windows inside the outer training data."""
    folds = max(2, int(params.get("cv_folds", 3)))
    horizon = max(1, int(params.get("cv_horizon", 1)))
    initial = len(source) - folds * horizon
    period = int(seasonal_order[3]) if seasonal_order[3] else 1
    minimum = max(16, 2 * period if seasonal_order[3] else 16)
    if initial < minimum:
        return (
            None,
            0,
            (
                f"Rolling validation needs at least {minimum + folds * horizon} "
                "observations for the configured folds and horizon."
            ),
        )

    errors: list[float] = []
    completed = 0
    for fold in range(folds):
        training_end = initial + fold * horizon
        validation_end = training_end + horizon
        training = source[:training_end]
        actual = source[training_end:validation_end]
        transformer = VarianceTransformer(
            requested_method=str(params.get("transformation", "None")),
            requested_lambda=(
                None
                if params.get("transformation_lambda") is None
                else float(params["transformation_lambda"])
            ),
            allow_shift=bool(params.get("allow_transform_shift", False)),
            identity_tolerance=float(params.get("identity_tolerance", 0.10)),
        )
        try:
            transformed_training = transformer.fit_transform(training)
            fold_fit = _fit_candidate(
                transformed_training,
                order,
                seasonal_order,
                trend,
                max(25, int(params.get("maximum_iterations", 150))),
            )
            prediction = fold_fit.get_forecast(steps=len(actual))
            confidence = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
            cv_params = dict(params)
            cv_params["inverse_simulations"] = min(
                500, max(250, int(params.get("inverse_simulations", 500)))
            )
            forecast, _lower, _upper, _details = _inverse_forecast(
                transformer, prediction, confidence, cv_params
            )
        except Exception as exc:
            return (
                None,
                completed,
                f"Fold {fold + 1} failed: {type(exc).__name__}: {exc}",
            )
        if not np.isfinite(forecast).all():
            return None, completed, f"Fold {fold + 1} produced a non-finite forecast."
        errors.extend((actual - forecast).tolist())
        completed += 1

    error_array = np.asarray(errors, dtype=float)
    if criterion == "CV MAE":
        score = float(np.mean(np.abs(error_array)))
    else:
        score = float(np.sqrt(np.mean(np.square(error_array))))
    return score, completed, None


def _inverse_forecast(
    transformer: VarianceTransformer,
    prediction: Any,
    confidence: np.ndarray,
    params: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    transformed_mean = np.asarray(prediction.predicted_mean, dtype=float).reshape(-1)
    direct_mean = transformer.inverse(transformed_mean)
    direct_lower = transformer.inverse(confidence[:, 0])
    direct_upper = transformer.inverse(confidence[:, 1])
    lower = np.minimum(direct_lower, direct_upper)
    upper = np.maximum(direct_lower, direct_upper)
    bias_adjust = bool(params.get("bias_adjust", True))
    simulations = max(250, int(params.get("inverse_simulations", 2000)))
    details = {
        "bias_adjusted": False,
        "simulation_draws": 0,
        "method": "Direct inverse transformation",
    }
    if transformer.applied_method == "None" or not bias_adjust:
        return direct_mean, lower, upper, details

    standard_errors = np.asarray(prediction.se_mean, dtype=float).reshape(-1)
    if (
        len(standard_errors) != len(transformed_mean)
        or not np.isfinite(standard_errors).all()
    ):
        details["fallback_reason"] = "Forecast standard errors were unavailable."
        return direct_mean, lower, upper, details

    generator = np.random.default_rng(int(params.get("simulation_seed", 42)))
    draws = generator.normal(
        loc=transformed_mean,
        scale=np.maximum(standard_errors, np.finfo(float).eps),
        size=(simulations, len(transformed_mean)),
    )
    inverted = transformer.inverse(draws)
    adjusted = np.empty(len(transformed_mean), dtype=float)
    simulated_lower = np.empty(len(transformed_mean), dtype=float)
    simulated_upper = np.empty(len(transformed_mean), dtype=float)
    for index in range(len(transformed_mean)):
        finite = inverted[:, index][np.isfinite(inverted[:, index])]
        if len(finite) < max(100, simulations // 2):
            details["fallback_reason"] = (
                "Too many simulated values were outside the inverse-transform domain."
            )
            return direct_mean, lower, upper, details
        adjusted[index] = float(np.mean(finite))
        simulated_lower[index], simulated_upper[index] = np.quantile(
            finite, [0.025, 0.975]
        )
    details.update(
        {
            "bias_adjusted": True,
            "simulation_draws": simulations,
            "method": "Simulation-based inverse transformation",
        }
    )
    return adjusted, simulated_lower, simulated_upper, details


def _base_failure_details(
    transformer: VarianceTransformer,
    transformed: np.ndarray,
    analysis_values: np.ndarray,
    d: int,
    seasonal_d: int,
    regular_history: list[dict[str, Any]],
    seasonal_history: list[dict[str, Any]],
    correlations: dict[str, Any],
    records: list[dict[str, Any]],
    params: dict[str, Any],
) -> dict[str, Any]:
    return {
        "pipeline_version": 2,
        "selection": "Automatic" if params.get("automatic", False) else "Manual",
        "diagnostic_policy": params.get("diagnostic_policy", "Advisory"),
        "transformation": transformer.details(),
        "transformed_series": [float(value) for value in transformed],
        "analysis_series": [float(value) for value in analysis_values],
        "selected_d": d,
        "selected_D": seasonal_d,
        "regular_stationarity_history": regular_history,
        "seasonal_stationarity_history": seasonal_history,
        "order_correlations": correlations,
        "candidate_results": records,
        "models_evaluated": len(records),
        "models_succeeded": sum("fit_error" not in row for row in records),
        "eligible_models": sum(bool(row.get("eligible")) for row in records),
    }


def fit_arima_pipeline(
    values: Any,
    steps: int,
    params: dict[str, Any],
    *,
    seasonal: bool,
) -> dict[str, Any]:
    """Fit, diagnose, select, forecast, and invert an ARIMA-family model."""
    source = np.asarray(values, dtype=float).reshape(-1)
    if len(source) < 12:
        raise ValueError(
            "The diagnostic ARIMA pipeline requires at least 12 observations."
        )
    period = int(params.get("seasonal_period", 1 if not seasonal else 12))
    if seasonal and period < 2:
        raise ValueError("SARIMA requires a seasonal period of at least 2.")
    if seasonal and len(source) < 2 * period:
        raise ValueError(
            f"SARIMA requires at least two seasonal cycles ({2 * period} observations)."
        )
    if not seasonal:
        period = 1

    transformer = VarianceTransformer(
        requested_method=str(params.get("transformation", "None")),
        requested_lambda=(
            None
            if params.get("transformation_lambda") is None
            else float(params["transformation_lambda"])
        ),
        allow_shift=bool(params.get("allow_transform_shift", False)),
        identity_tolerance=float(params.get("identity_tolerance", 0.10)),
    )
    transformed = transformer.fit_transform(source)

    seasonal_d, seasonal_history = _seasonal_difference_order(
        transformed, params, seasonal, period
    )
    d, stationarity_achieved, regular_history = _regular_difference_order(
        transformed, params, seasonal_d, period
    )
    analysis_values = _difference(transformed, d, seasonal_d, period)
    if len(analysis_values) < 8:
        raise ValueError(
            "Differencing leaves too few observations. Reduce d/D or the seasonal period."
        )
    requested_acf_lags = max(
        int(params.get("acf_lags", 40)),
        (2 * period if seasonal else 1),
    )
    correlations = _correlation_records(analysis_values, requested_acf_lags)

    maximum_p = max(0, int(params.get("max_p", 3)))
    maximum_q = max(0, int(params.get("max_q", 3)))
    maximum_P = max(0, int(params.get("max_P", 1))) if seasonal else 0
    maximum_Q = max(0, int(params.get("max_Q", 1))) if seasonal else 0
    maximum_order = max(0, int(params.get("max_order", 6)))
    strategy = _normalise_name(
        params.get(
            "search_strategy",
            "Exhaustive grid" if params.get("automatic", False) else "Manual order",
        )
    )
    full_candidates = _grid_candidates(
        d,
        seasonal_d,
        period if seasonal else 0,
        range(maximum_p + 1),
        range(maximum_q + 1),
        range(maximum_P + 1),
        range(maximum_Q + 1),
        maximum_order,
    )
    if not seasonal:
        full_candidates = [
            (order, (0, 0, 0, 0)) for order, _seasonal_order in full_candidates
        ]

    if strategy in {"manual candidate list", "manual list"}:
        p_values = _order_values(params.get("manual_p_values", "0,1"), "p", 6)
        q_values = _order_values(params.get("manual_q_values", "0,1"), "q", 6)
        P_values = (
            _order_values(params.get("manual_P_values", "0,1"), "P", 3)
            if seasonal
            else [0]
        )
        Q_values = (
            _order_values(params.get("manual_Q_values", "0,1"), "Q", 3)
            if seasonal
            else [0]
        )
        primary_candidates = _grid_candidates(
            d,
            seasonal_d,
            period if seasonal else 0,
            p_values,
            q_values,
            P_values,
            Q_values,
            maximum_order,
        )
        if not seasonal:
            primary_candidates = [
                (order, (0, 0, 0, 0)) for order, _seasonal_order in primary_candidates
            ]
        fallback_candidates = []
    elif strategy in {"manual", "manual order"}:
        order = (
            int(params.get("p", 1)),
            int(params.get("d", d)),
            int(params.get("q", 1)),
        )
        if _normalise_name(params.get("difference_mode", "Manual")) != "manual":
            order = (order[0], d, order[2])
        seasonal_order = (
            (
                int(params.get("P", 1)),
                int(params.get("D", seasonal_d)),
                int(params.get("Q", 1)),
                period,
            )
            if seasonal
            else (0, 0, 0, 0)
        )
        if (
            seasonal
            and _normalise_name(params.get("seasonal_difference_mode", "Manual"))
            != "manual"
        ):
            seasonal_order = (
                seasonal_order[0],
                seasonal_d,
                seasonal_order[2],
                period,
            )
        primary_candidates = [(order, seasonal_order)]
        fallback_candidates: list[
            tuple[tuple[int, int, int], tuple[int, int, int, int]]
        ] = []
    elif strategy in {"stepwise", "stepwise search"}:
        primary_candidates = _stepwise_candidates(
            transformed, params, seasonal, period, d, seasonal_d
        )
        fallback_candidates = [
            candidate
            for candidate in full_candidates
            if candidate not in set(primary_candidates)
        ]
    elif strategy in {"guided", "guided acf/pacf", "guided acf pacf"}:
        p_values, q_values, P_values, Q_values = _guided_components(
            correlations,
            maximum_p,
            maximum_q,
            maximum_P,
            maximum_Q,
            period,
        )
        primary_candidates = _grid_candidates(
            d,
            seasonal_d,
            period if seasonal else 0,
            p_values,
            q_values,
            P_values,
            Q_values,
            maximum_order,
        )
        if not seasonal:
            primary_candidates = [
                (order, (0, 0, 0, 0)) for order, _seasonal_order in primary_candidates
            ]
        primary_set = set(primary_candidates)
        fallback_candidates = [
            candidate for candidate in full_candidates if candidate not in primary_set
        ]
    else:
        primary_candidates = full_candidates
        fallback_candidates = []

    if not primary_candidates:
        raise ValueError("The configured order limits do not contain any candidates.")
    if len(primary_candidates) + len(fallback_candidates) > 250:
        raise ValueError(
            "The configured search contains more than 250 candidate models. "
            "Reduce the maximum orders or the maximum total order."
        )

    criterion = str(params.get("criterion", "AICc"))
    if criterion.upper() == "AICC":
        criterion = "AICc"
    else:
        criterion = criterion.upper()
    if criterion not in {"AIC", "AICc", "BIC", "HQIC", "CV RMSE", "CV MAE"}:
        raise ValueError(
            "Model ranking criterion must be AIC, AICc, BIC, HQIC, CV RMSE, or CV MAE."
        )
    trend = _trend_code(params, d, seasonal_d)
    maximum_iterations = max(25, int(params.get("maximum_iterations", 150)))
    evaluated: list[dict[str, Any]] = []

    def evaluate_candidates(
        candidates: Iterable[tuple[tuple[int, int, int], tuple[int, int, int, int]]],
        phase: str,
    ) -> None:
        for order, seasonal_order in candidates:
            try:
                fit = _fit_candidate(
                    transformed,
                    order,
                    seasonal_order,
                    trend,
                    maximum_iterations,
                )
                criteria = _information_criteria(fit)
                diagnostics = _candidate_diagnostics(
                    fit,
                    order,
                    seasonal_order,
                    params,
                    seasonal,
                    period,
                    stationarity_achieved,
                )
                summary = _candidate_summary(
                    order, seasonal_order, criteria, diagnostics
                )
                summary["search_phase"] = phase
                evaluated.append(
                    {"fit": fit, "diagnostics": diagnostics, "summary": summary}
                )
            except (
                Exception
            ) as exc:  # Candidate numerical failures are reported, not fatal.
                evaluated.append(
                    {
                        "fit": None,
                        "diagnostics": None,
                        "summary": {
                            "order": list(order),
                            "seasonal_order": list(seasonal_order),
                            "eligible": False,
                            "failed_gate_count": 99,
                            "failure_reasons": "fit error",
                            "fit_error": f"{type(exc).__name__}: {exc}",
                            "search_phase": phase,
                        },
                    }
                )

    evaluate_candidates(primary_candidates, "Primary")
    has_eligible = any(
        item["summary"].get("eligible") for item in evaluated if item["fit"] is not None
    )
    if (
        not has_eligible
        and fallback_candidates
        and bool(params.get("guided_fallback", True))
    ):
        evaluate_candidates(fallback_candidates, "Expanded fallback")

    successful = [item for item in evaluated if item["fit"] is not None]
    summaries = [item["summary"] for item in evaluated]
    if not successful:
        details = _base_failure_details(
            transformer,
            transformed,
            analysis_values,
            d,
            seasonal_d,
            regular_history,
            seasonal_history,
            correlations,
            summaries,
            params,
        )
        raise NoEligibleModelError("Every configured candidate failed to fit.", details)

    eligible = [item for item in successful if item["summary"]["eligible"]]
    if criterion in {"CV RMSE", "CV MAE"}:
        scoring_targets = eligible or (
            successful if bool(params.get("allow_near_match", False)) else []
        )
        for item in scoring_targets:
            order = tuple(item["summary"]["order"])
            seasonal_order = tuple(item["summary"]["seasonal_order"])
            cv_score, completed_folds, cv_error = _rolling_cv_score(
                source,
                order,
                seasonal_order,
                trend,
                params,
                criterion,
            )
            item["summary"][criterion] = cv_score
            item["summary"]["CV folds completed"] = completed_folds
            item["summary"]["CV error"] = cv_error
        if scoring_targets and not any(
            item["summary"].get(criterion) is not None for item in scoring_targets
        ):
            raise ValueError(
                "Rolling validation failed for every selectable candidate. Reduce the "
                "folds/horizon or use an information criterion."
            )
    selection_override = False
    if eligible:
        selected = min(
            eligible,
            key=lambda item: _criterion_value(item["summary"], criterion),
        )
    elif bool(params.get("allow_near_match", False)):
        selected = min(
            successful,
            key=lambda item: (
                int(item["summary"]["failed_gate_count"]),
                _criterion_value(item["summary"], criterion),
            ),
        )
        selection_override = True
    else:
        ranked_summaries = sorted(
            summaries,
            key=lambda row: (
                int(row.get("failed_gate_count", 99)),
                _criterion_value(row, criterion),
            ),
        )
        details = _base_failure_details(
            transformer,
            transformed,
            analysis_values,
            d,
            seasonal_d,
            regular_history,
            seasonal_history,
            correlations,
            ranked_summaries,
            params,
        )
        details["criterion"] = criterion
        raise NoEligibleModelError(
            "No candidate satisfies every mandatory diagnostic. Enable the explicit "
            "near-match override, widen the search, change the transformation/differencing, "
            "or relax a diagnostic requirement.",
            details,
        )

    fit = selected["fit"]
    diagnostics = selected["diagnostics"]
    selected_summary = selected["summary"]
    prediction = fit.get_forecast(steps=steps)
    confidence = np.asarray(prediction.conf_int(alpha=0.05), dtype=float)
    forecast, lower, upper, inverse_details = _inverse_forecast(
        transformer, prediction, confidence, params
    )
    fitted = transformer.inverse(np.asarray(fit.fittedvalues, dtype=float))
    if not np.isfinite(forecast).all():
        raise ValueError(
            "The selected model produced non-finite original-scale forecasts."
        )

    sorted_summaries = sorted(
        summaries,
        key=lambda row: (
            not bool(row.get("eligible")),
            int(row.get("failed_gate_count", 99)),
            _criterion_value(row, criterion),
        ),
    )
    details = _base_failure_details(
        transformer,
        transformed,
        analysis_values,
        d,
        seasonal_d,
        regular_history,
        seasonal_history,
        correlations,
        sorted_summaries,
        params,
    )
    details.update(
        {
            "selection": "Manual" if strategy.startswith("manual") else "Automatic",
            "search_strategy": params.get("search_strategy", strategy.title()),
            "search_expanded": any(
                row.get("search_phase") == "Expanded fallback" for row in summaries
            ),
            "stationarity_achieved": stationarity_achieved,
            "selected_order": list(selected_summary["order"]),
            "selected_seasonal_order": list(selected_summary["seasonal_order"])
            if seasonal
            else None,
            "criterion": criterion,
            "criterion_value": selected_summary.get(criterion),
            "selected_model_eligible": bool(selected_summary["eligible"]),
            "selection_override": selection_override,
            "selected_failures": selected_summary.get("failure_reasons", ""),
            "trend": trend,
            "diagnostic_requirements": diagnostics["requirements"],
            "diagnostic_outcomes": diagnostics["outcomes"],
            "coefficients": diagnostics["coefficients"],
            "root_diagnostics": diagnostics["roots"],
            "normality": diagnostics["normality"],
            "white_noise": diagnostics["white_noise"],
            "heteroskedasticity": diagnostics["heteroskedasticity"],
            "residual_mean": diagnostics["residual_mean"],
            "residuals": diagnostics["residuals"],
            "residual_correlations": diagnostics["residual_correlations"],
            "qq_plot": diagnostics["qq_plot"],
            "inverse_transformation": inverse_details,
            "transformed_forecast": [
                float(value)
                for value in np.asarray(prediction.predicted_mean, dtype=float)
            ],
            "top_candidates": sorted_summaries[:10],
        }
    )
    return {
        "fitted": fitted,
        "forecast": np.asarray(forecast, dtype=float),
        "lower": np.asarray(lower, dtype=float),
        "upper": np.asarray(upper, dtype=float),
        "details": details,
    }
