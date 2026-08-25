"""Single registry for every user-selectable forecasting method."""

from __future__ import annotations

from types import MappingProxyType

from .methods.baselines.drift import SPEC as DRIFT
from .methods.baselines.naive import SPEC as NAIVE
from .methods.baselines.seasonal_naive import SPEC as SEASONAL_NAIVE
from .methods.decomposition.stl import SPEC as STL
from .methods.decomposition.mstl_ets import SPEC as MSTL_ETS
from .methods.machine_learning.cnn import SPEC as CNN
from .methods.machine_learning.cart import SPEC as CART
from .methods.machine_learning.extra_trees import SPEC as EXTRA_TREES
from .methods.machine_learning.knn import SPEC as KNN_REGRESSION
from .methods.machine_learning.lagged_linear import SPEC as LAGGED_LINEAR
from .methods.machine_learning.lstm import SPEC as LSTM
from .methods.machine_learning.nbeats import SPEC as NBEATS
from .methods.machine_learning.prophet import SPEC as PROPHET
from .methods.machine_learning.random_forest import SPEC as RANDOM_FOREST
from .methods.machine_learning.regularized_regression import SPEC as REGULARIZED_REGRESSION
from .methods.machine_learning.support_vector import SPEC as SUPPORT_VECTOR_REGRESSION
from .methods.machine_learning.tcn import SPEC as TCN
from .methods.machine_learning.xgboost import SPEC as XGBOOST
from .methods.smoothing.automatic_ets import SPEC as AUTOMATIC_ETS
from .methods.smoothing.holt import SPEC as HOLT
from .methods.smoothing.holt_winters import SPEC as HOLT_WINTERS
from .methods.smoothing.moving_average import SPEC as MOVING_AVERAGE
from .methods.smoothing.simple_exponential import SPEC as SIMPLE_EXPONENTIAL
from .methods.smoothing.theta import SPEC as THETA
from .methods.smoothing.weighted_moving_average import SPEC as WEIGHTED_MOVING_AVERAGE
from .methods.statistical.arima import SPEC as ARIMA
from .methods.statistical.croston import SPEC as CROSTON_FAMILY
from .methods.statistical.sarima import SPEC as SARIMA
from .methods.statistical.tbats import SPEC as TBATS
from .methods.trend.exponential import SPEC as EXPONENTIAL_TREND
from .methods.trend.linear import SPEC as LINEAR_TREND
from .methods.trend.logarithmic import SPEC as LOGARITHMIC_TREND
from .methods.trend.quadratic import SPEC as QUADRATIC_TREND
from .contracts import MethodSpec


METHOD_SPECS: tuple[MethodSpec, ...] = (
    NAIVE,
    SEASONAL_NAIVE,
    DRIFT,
    MOVING_AVERAGE,
    WEIGHTED_MOVING_AVERAGE,
    SIMPLE_EXPONENTIAL,
    HOLT,
    HOLT_WINTERS,
    THETA,
    AUTOMATIC_ETS,
    ARIMA,
    SARIMA,
    TBATS,
    CROSTON_FAMILY,
    STL,
    MSTL_ETS,
    PROPHET,
    LSTM,
    CNN,
    XGBOOST,
    LAGGED_LINEAR,
    REGULARIZED_REGRESSION,
    CART,
    RANDOM_FOREST,
    SUPPORT_VECTOR_REGRESSION,
    KNN_REGRESSION,
    EXTRA_TREES,
    NBEATS,
    TCN,
    LINEAR_TREND,
    QUADRATIC_TREND,
    EXPONENTIAL_TREND,
    LOGARITHMIC_TREND,
)


def _build_registry(specs: tuple[MethodSpec, ...]) -> dict[str, MethodSpec]:
    registry: dict[str, MethodSpec] = {}
    url_paths: set[str] = set()
    for spec in specs:
        if spec.model_id in registry:
            raise RuntimeError(f"Duplicate method ID in registry: {spec.model_id}")
        if spec.url_path in url_paths:
            raise RuntimeError(f"Duplicate method URL in registry: {spec.url_path}")
        if not spec.model_id or spec.model_id != spec.model_id.lower():
            raise RuntimeError(f"Method IDs must be non-empty lowercase strings: {spec}")
        registry[spec.model_id] = spec
        url_paths.add(spec.url_path)
    return registry


METHOD_REGISTRY = MappingProxyType(_build_registry(METHOD_SPECS))
MODEL_NAMES = MappingProxyType(
    {spec.model_id: spec.display_name for spec in METHOD_SPECS}
)
NAVIGATION_GROUPS = tuple(
    dict.fromkeys(spec.navigation_group for spec in METHOD_SPECS)
)


def get_method(model_id: str) -> MethodSpec:
    """Return one registered method or raise the established public error."""
    try:
        return METHOD_REGISTRY[model_id]
    except KeyError as exc:
        raise ValueError(f"Unknown model: {model_id}") from exc


def methods_for_group(group: str) -> tuple[MethodSpec, ...]:
    """Return registered methods in navigation order for one group."""
    return tuple(spec for spec in METHOD_SPECS if spec.navigation_group == group)
