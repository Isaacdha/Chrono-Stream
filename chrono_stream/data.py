"""Data preparation helpers used by the Streamlit pages and tests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Hashable

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset


@dataclass(frozen=True)
class PreparationReport:
    input_rows: int
    output_rows: int
    invalid_rows_removed: int
    duplicate_timestamps_combined: int
    missing_periods_created: int
    frequency: str | None


def infer_frequency(dates: pd.Series | pd.DatetimeIndex) -> str | None:
    """Infer a useful pandas frequency, falling back to the median interval."""
    index = (
        pd.DatetimeIndex(pd.to_datetime(dates, errors="coerce", format="mixed"))
        .dropna()
        .unique()
    )
    index = pd.DatetimeIndex(index).sort_values()
    if len(index) < 2:
        return None

    if len(index) >= 3:
        try:
            inferred = pd.infer_freq(index)
            if inferred:
                return to_offset(inferred).freqstr
        except (TypeError, ValueError):
            pass

    month_numbers = index.year * 12 + index.month
    month_steps = np.diff(month_numbers)
    if len(month_steps) and (month_steps > 0).all():
        month_step = int(np.gcd.reduce(month_steps))
        if (index.day == 1).all():
            if month_step >= 12 and month_step % 12 == 0:
                return pd.offsets.YearBegin(
                    n=month_step // 12, month=int(index[0].month)
                ).freqstr
            if month_step >= 3 and month_step % 3 == 0:
                return pd.offsets.QuarterBegin(
                    n=month_step // 3, startingMonth=int(index[0].month)
                ).freqstr
            return pd.offsets.MonthBegin(n=month_step).freqstr
        if index.is_month_end.all():
            if month_step >= 12 and month_step % 12 == 0:
                return pd.offsets.YearEnd(
                    n=month_step // 12, month=int(index[0].month)
                ).freqstr
            if month_step >= 3 and month_step % 3 == 0:
                return pd.offsets.QuarterEnd(
                    n=month_step // 3, startingMonth=int(index[0].month)
                ).freqstr
            return pd.offsets.MonthEnd(n=month_step).freqstr

    normalized = index == index.normalize()
    day_steps = np.asarray((index[1:] - index[:-1]) / pd.Timedelta(days=1))
    if (
        normalized.all()
        and len(day_steps)
        and np.equal(day_steps, np.floor(day_steps)).all()
    ):
        day_step = int(np.gcd.reduce(day_steps.astype(int)))
        if (index.weekday == index[0].weekday()).all() and day_step % 7 == 0:
            return pd.offsets.Week(
                n=day_step // 7, weekday=int(index[0].weekday())
            ).freqstr
        return pd.offsets.Day(n=max(day_step, 1)).freqstr

    deltas = pd.Series(index[1:] - index[:-1])
    median_delta = deltas.median()
    if pd.isna(median_delta) or median_delta <= pd.Timedelta(0):
        return None
    try:
        return to_offset(median_delta).freqstr
    except (TypeError, ValueError):
        return None


def prepare_time_series(
    frame: pd.DataFrame,
    date_column: Hashable,
    value_column: Hashable,
    *,
    frequency: str | None = None,
    regularize: bool = True,
    missing_method: str = "Interpolate",
    duplicate_method: str = "Mean",
    day_first: bool = False,
) -> tuple[pd.DataFrame, PreparationReport]:
    """Validate, sort, deduplicate, and optionally regularize a two-column series."""
    if date_column == value_column:
        raise ValueError("Choose different date and value columns.")
    if date_column not in frame or value_column not in frame:
        raise ValueError("The selected columns are not present in the uploaded file.")

    input_rows = len(frame)
    data = frame[[date_column, value_column]].copy()
    data[date_column] = pd.to_datetime(
        data[date_column], errors="coerce", format="mixed", dayfirst=day_first
    )
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")
    finite = np.isfinite(data[value_column].to_numpy(dtype=float, na_value=np.nan))
    valid = data[date_column].notna().to_numpy() & finite
    invalid_rows = int((~valid).sum())
    data = data.loc[valid]

    if data.empty:
        raise ValueError("No rows contain both a valid date and a numeric value.")

    data = data.sort_values(date_column)
    duplicate_count = int(data.duplicated(date_column).sum())
    aggregations = {
        "Mean": "mean",
        "Sum": "sum",
        "First": "first",
        "Last": "last",
    }
    aggregation = aggregations.get(duplicate_method)
    if aggregation is None:
        raise ValueError(f"Unknown duplicate handling method: {duplicate_method}")
    data = data.groupby(date_column, as_index=False)[value_column].agg(aggregation)

    resolved_frequency = frequency or infer_frequency(data[date_column])
    if resolved_frequency:
        try:
            resolved_frequency = to_offset(resolved_frequency).freqstr
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid time frequency '{resolved_frequency}'.") from exc
    missing_periods = 0
    if regularize and resolved_frequency:
        try:
            resampler = data.set_index(date_column)[[value_column]].resample(
                to_offset(resolved_frequency)
            )
            indexed = (
                resampler.sum(min_count=1)
                if aggregation == "sum"
                else resampler.agg(aggregation)
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid time frequency '{resolved_frequency}'.") from exc
        indexed.index.name = date_column
        missing_periods = int(indexed[value_column].isna().sum())
        if missing_method == "Interpolate":
            indexed[value_column] = (
                indexed[value_column].interpolate(method="time").bfill().ffill()
            )
        elif missing_method == "Forward fill":
            indexed[value_column] = indexed[value_column].ffill().bfill()
        elif missing_method == "Backward fill":
            indexed[value_column] = indexed[value_column].bfill().ffill()
        elif missing_method == "Drop gaps":
            indexed = indexed.dropna(subset=[value_column])
        else:
            raise ValueError(f"Unknown missing-value method: {missing_method}")
        data = indexed.reset_index()

    if data[value_column].isna().any():
        raise ValueError(
            "Missing values remain after preparation. Choose a fill method or a compatible frequency."
        )
    data[value_column] = data[value_column].astype(float)
    data = data.reset_index(drop=True)
    if len(data) < 8:
        raise ValueError("At least 8 valid observations are required for forecasting.")
    if data[date_column].duplicated().any():
        raise ValueError("Duplicate timestamps remain after preparation.")

    report = PreparationReport(
        input_rows=input_rows,
        output_rows=len(data),
        invalid_rows_removed=invalid_rows,
        duplicate_timestamps_combined=duplicate_count,
        missing_periods_created=missing_periods,
        frequency=resolved_frequency,
    )
    return data, report


def future_dates(
    dates: pd.Series | pd.DatetimeIndex,
    periods: int,
    frequency: str | None = None,
) -> pd.DatetimeIndex:
    """Create timestamps immediately following the observed series."""
    if periods < 1:
        raise ValueError("Forecast periods must be at least 1.")
    index = pd.DatetimeIndex(pd.to_datetime(dates)).sort_values()
    if index.empty:
        raise ValueError("Cannot create forecast dates without observed dates.")

    resolved_frequency = frequency or infer_frequency(index)
    if resolved_frequency:
        offset = to_offset(resolved_frequency)
    elif len(index) >= 2:
        offset = to_offset(pd.Series(index[1:] - index[:-1]).median())
    else:
        offset = to_offset("D")
    return pd.date_range(start=index[-1] + offset, periods=periods, freq=offset)


def data_signature(frame: pd.DataFrame) -> str:
    """Return a stable signature used to detect when saved model results are stale."""
    digest = hashlib.sha256()
    digest.update("|".join(map(str, frame.columns)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    return digest.hexdigest()
