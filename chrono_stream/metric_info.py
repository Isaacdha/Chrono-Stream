"""Practical guidance, references, and downloads for forecast accuracy metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .evaluation import ACCURACY_METRIC_KEYS
from .metric_literature_reviews import METRIC_LITERATURE_REVIEWS


@dataclass(frozen=True, slots=True)
class MetricReference:
    """One defining or critical accuracy-metric source in APA 7 style."""

    apa: str
    url: str
    contribution: str


@dataclass(frozen=True, slots=True)
class MetricInformation:
    """Practical and scholarly metadata for one reported accuracy metric."""

    display_name: str
    formula: str
    how_it_works: str
    chrono_stream: str
    when_to_use: str
    limitations: str
    citation_ready: str
    references: tuple[MetricReference, ...]


def _reference(apa: str, url: str, contribution: str) -> MetricReference:
    return MetricReference(apa=apa, url=url, contribution=contribution)


GNEITING_2011 = _reference(
    "Gneiting, T. (2011). Making and evaluating point forecasts. Journal of the "
    "American Statistical Association, 106(494), 746–762. "
    "https://doi.org/10.1198/jasa.2011.r10138",
    "https://doi.org/10.1198/jasa.2011.r10138",
    "Establishes how absolute and squared scoring functions target different predictive functionals.",
)
WILLMOTT_MATSUURA_2005 = _reference(
    "Willmott, C. J., & Matsuura, K. (2005). Advantages of the mean absolute "
    "error (MAE) over the root mean square error (RMSE) in assessing average "
    "model performance. Climate Research, 30(1), 79–82. "
    "https://doi.org/10.3354/cr030079",
    "https://doi.org/10.3354/cr030079",
    "Analyzes the distinct information carried by MAE and RMSE and cautions against reading RMSE as average error.",
)
ARMSTRONG_COLLOPY_1992 = _reference(
    "Armstrong, J. S., & Collopy, F. (1992). Error measures for generalizing "
    "about forecasting methods: Empirical comparisons. International Journal "
    "of Forecasting, 8(1), 69–80. "
    "https://doi.org/10.1016/0169-2070(92)90008-W",
    "https://doi.org/10.1016/0169-2070(92)90008-W",
    "Empirically demonstrates that error-measure choice affects forecast-method comparisons across series.",
)
MAKRIDAKIS_1993 = _reference(
    "Makridakis, S. (1993). Accuracy measures: Theoretical and practical "
    "concerns. International Journal of Forecasting, 9(4), 527–529. "
    "https://doi.org/10.1016/0169-2070(93)90079-3",
    "https://doi.org/10.1016/0169-2070(93)90079-3",
    "Discusses practical defects of common accuracy measures and the motivation for symmetric percentage errors.",
)
GOODWIN_LAWTON_1999 = _reference(
    "Goodwin, P., & Lawton, R. (1999). On the asymmetry of the symmetric MAPE. "
    "International Journal of Forecasting, 15(4), 405–408. "
    "https://doi.org/10.1016/S0169-2070(99)00007-2",
    "https://doi.org/10.1016/S0169-2070(99)00007-2",
    "Shows that the symmetric-MAPE label does not eliminate important over/under-forecast asymmetries.",
)
MAKRIDAKIS_HIBON_2000 = _reference(
    "Makridakis, S., & Hibon, M. (2000). The M3-Competition: Results, "
    "conclusions and implications. International Journal of Forecasting, "
    "16(4), 451–476. https://doi.org/10.1016/S0169-2070(00)00057-1",
    "https://doi.org/10.1016/S0169-2070(00)00057-1",
    "Documents the prominent empirical use and aggregation of sMAPE in the M3 forecasting competition.",
)
HYNDMAN_KOEHLER_2006 = _reference(
    "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of "
    "forecast accuracy. International Journal of Forecasting, 22(4), 679–688. "
    "https://doi.org/10.1016/j.ijforecast.2006.03.001",
    "https://doi.org/10.1016/j.ijforecast.2006.03.001",
    "Catalogues degeneracies in percentage and relative errors and defines mean absolute scaled error.",
)
KOLASSA_SCHUTZ_2007 = _reference(
    "Kolassa, S., & Schütz, W. (2007). Advantages of the MAD/Mean ratio over "
    "the MAPE. Foresight: The International Journal of Applied Forecasting, "
    "6, 40–43.",
    "https://econpapers.repec.org/article/forijafaa/y_3a2007_3ai_3a6_3ap_3a40-43.htm",
    "Develops the MAD/Mean ratio now commonly called WAPE and relates it to weighted percentage errors.",
)
MAKRIDAKIS_SPILIOTIS_ASSIMAKOPOULOS_2022 = _reference(
    "Makridakis, S., Spiliotis, E., & Assimakopoulos, V. (2022). M5 accuracy "
    "competition: Results, findings, and conclusions. International Journal "
    "of Forecasting, 38(4), 1346–1364. "
    "https://doi.org/10.1016/j.ijforecast.2021.11.013",
    "https://doi.org/10.1016/j.ijforecast.2021.11.013",
    "Defines the M5 competition's RMSSE-based evaluation and its intermittent-retail-demand context.",
)
HEWAMALAGE_ACKERMANN_BERGMEIR_2023 = _reference(
    "Hewamalage, H., Ackermann, K., & Bergmeir, C. (2023). Forecast evaluation "
    "for data scientists: Common pitfalls and best practices. Data Mining and "
    "Knowledge Discovery, 37, 788–832. "
    "https://doi.org/10.1007/s10618-022-00894-5",
    "https://doi.org/10.1007/s10618-022-00894-5",
    "Synthesizes metric selection, benchmark scaling, data characteristics, and leakage-safe evaluation practice.",
)


METRIC_INFORMATION: dict[str, MetricInformation] = {
    "MAE": MetricInformation(
        display_name="Mean Absolute Error (MAE)",
        formula="MAE = mean(|yₜ − ŷₜ|)",
        how_it_works=(
            "MAE takes the absolute size of every holdout error and averages those "
            "magnitudes. Every additional unit of error has the same weight, regardless "
            "of whether the forecast is above or below the actual value."
        ),
        chrono_stream=(
            "Chrono Stream calculates MAE over finite actual/forecast pairs in the final "
            "chronological holdout. It is displayed in the original series unit and uses "
            "the identical holdout for every saved model."
        ),
        when_to_use=(
            "Use MAE when an error in the original unit is meaningful and the practical "
            "cost grows roughly linearly with absolute error. It is especially clear for "
            "same-series model comparisons."
        ),
        limitations=(
            "MAE is not scale free, so raw values should not be averaged across unrelated "
            "series without deliberate weighting. Absolute loss rewards a conditional "
            "median forecast and may understate the importance of rare large misses."
        ),
        citation_ready=(
            "Mean absolute error averages absolute forecast-error magnitudes in the "
            "response unit. Absolute loss is consistent for a conditional median, so MAE "
            "should be chosen when that target and linear error cost match the decision "
            "problem (Gneiting, 2011; Willmott & Matsuura, 2005)."
        ),
        references=(GNEITING_2011, WILLMOTT_MATSUURA_2005, HEWAMALAGE_ACKERMANN_BERGMEIR_2023),
    ),
    "RMSE": MetricInformation(
        display_name="Root Mean Squared Error (RMSE)",
        formula="RMSE = √mean((yₜ − ŷₜ)²)",
        how_it_works=(
            "RMSE squares each holdout error, averages the squares, and takes the square "
            "root. The squaring makes a few large misses influence the score more strongly "
            "than they influence MAE."
        ),
        chrono_stream=(
            "Chrono Stream reports RMSE in the original series unit on the shared holdout "
            "and currently orders saved same-series models by lowest RMSE. The other metrics "
            "remain visible so users can detect a ranking driven by one large miss."
        ),
        when_to_use=(
            "Use RMSE when large misses deserve disproportionate weight or the point forecast "
            "is intended to target a conditional mean under squared loss."
        ),
        limitations=(
            "RMSE is scale dependent, sensitive to outliers and short-holdout extremes, and "
            "is not simply an average absolute error. It should not be compared across series "
            "with different scales without normalization."
        ),
        citation_ready=(
            "Root mean squared error is the square root of average squared forecast error. "
            "It emphasizes large misses and is consistent with conditional-mean point "
            "forecasting under squared loss (Gneiting, 2011), but it must not be interpreted "
            "as though it were ordinary average error (Willmott & Matsuura, 2005)."
        ),
        references=(GNEITING_2011, WILLMOTT_MATSUURA_2005, ARMSTRONG_COLLOPY_1992),
    ),
    "MAPE": MetricInformation(
        display_name="Mean Absolute Percentage Error (MAPE)",
        formula="MAPE = 100 × mean(|yₜ − ŷₜ| / |yₜ|)",
        how_it_works=(
            "MAPE divides each absolute holdout error by that period's absolute actual and "
            "then averages the percentages. Each period therefore supplies its own scale, "
            "which makes small actuals highly influential."
        ),
        chrono_stream=(
            "Chrono Stream reports MAPE only when every evaluated actual is nonzero. If any "
            "holdout actual is zero, MAPE is N/A: the app does not delete that period, add an "
            "epsilon, or pretend the undefined term is zero."
        ),
        when_to_use=(
            "Use MAPE only for strictly nonzero, meaningfully ratio-scaled data when a familiar "
            "per-period percentage summary is useful and actuals are not clustered near zero."
        ),
        limitations=(
            "MAPE is undefined at zero, unstable near zero, unsuitable for interval-scale or "
            "signed data, and can favor systematically low forecasts. It is not an appropriate "
            "primary metric for intermittent demand."
        ),
        citation_ready=(
            "Mean absolute percentage error averages absolute errors divided by their actual "
            "values. Although easy to communicate, it is undefined at zero and can be dominated "
            "by near-zero actuals; benchmark-scaled measures avoid these degeneracies "
            "(Armstrong & Collopy, 1992; Hyndman & Koehler, 2006)."
        ),
        references=(ARMSTRONG_COLLOPY_1992, MAKRIDAKIS_1993, HYNDMAN_KOEHLER_2006),
    ),
    "sMAPE": MetricInformation(
        display_name="Symmetric Mean Absolute Percentage Error (sMAPE)",
        formula="sMAPE = 100 × mean(2|yₜ − ŷₜ| / (|yₜ| + |ŷₜ|))",
        how_it_works=(
            "sMAPE scales each absolute error by the average magnitude of the actual and "
            "forecast. Chrono Stream's version ranges from 0% to 200% and is unchanged if the "
            "actual and forecast swap positions in the formula."
        ),
        chrono_stream=(
            "Chrono Stream uses absolute values in the denominator. A joint actual=forecast=0 "
            "case contributes 0%; if only one is zero, that period contributes 200%. The exact "
            "0–200 formula is used everywhere in the app."
        ),
        when_to_use=(
            "Use sMAPE as a bounded secondary percentage view when actuals can include zero and "
            "the displayed convention is acceptable. Compare it with non-percentage metrics."
        ),
        limitations=(
            "Several incompatible formulas share the sMAPE name. Despite the label, the score "
            "can treat equally sized over- and under-forecasts differently, behaves awkwardly "
            "near zero, and has no simple business-cost interpretation."
        ),
        citation_ready=(
            "Chrono Stream's sMAPE scales absolute error by the sum of absolute actual and "
            "forecast magnitudes. It is related to, but not identical with, the signed-denominator "
            "variant reported for the M3 competition (Makridakis & Hibon, 2000). The 'symmetric' "
            "label does not remove important directional and large-error asymmetries "
            "(Goodwin & Lawton, 1999)."
        ),
        references=(MAKRIDAKIS_1993, GOODWIN_LAWTON_1999, MAKRIDAKIS_HIBON_2000),
    ),
    "MASE": MetricInformation(
        display_name="Mean Absolute Scaled Error (MASE)",
        formula="MASE = holdout MAE / mean(|yₜ − yₜ₋ₘ|) on training data",
        how_it_works=(
            "MASE divides holdout MAE by the average absolute error of a naive lag-m benchmark "
            "computed on training observations. It is dimensionless; below 1 means the holdout "
            "MAE is below that historical benchmark scale."
        ),
        chrono_stream=(
            "Chrono Stream builds the denominator only from pre-holdout training observations. "
            "The configured metric scale period supplies m for every model. MASE is N/A if the "
            "lag cannot be formed or the training benchmark error is zero."
        ),
        when_to_use=(
            "Use MASE for zero-containing or differently scaled series when absolute-loss "
            "behavior and comparison with a declared naive or seasonal-naive scale are useful."
        ),
        limitations=(
            "MASE depends on the chosen lag and historical regime, is undefined for a constant "
            "training benchmark, and rewards a conditional median. MASE<1 is not by itself proof "
            "that the model beat a naive forecast on the same holdout."
        ),
        citation_ready=(
            "Mean absolute scaled error divides out-of-sample MAE by the in-sample absolute "
            "error of a naive benchmark, avoiding per-observation division by zero and enabling "
            "scale-free comparisons (Hyndman & Koehler, 2006)."
        ),
        references=(HYNDMAN_KOEHLER_2006, GNEITING_2011, HEWAMALAGE_ACKERMANN_BERGMEIR_2023),
    ),
    "RMSSE": MetricInformation(
        display_name="Root Mean Squared Scaled Error (RMSSE)",
        formula="RMSSE = √(holdout MSE / mean((yₜ − yₜ₋ₘ)²) on training data)",
        how_it_works=(
            "RMSSE divides holdout mean squared error by the training mean squared error of a "
            "naive lag-m benchmark, then takes the square root. It combines scale independence "
            "with extra weight on large misses."
        ),
        chrono_stream=(
            "Chrono Stream uses all pre-holdout training observations and the configured scale "
            "period m. It does not apply M5 active-sales trimming, hierarchy weights, or WRMSSE "
            "aggregation. Zero or unavailable benchmark scale produces N/A."
        ),
        when_to_use=(
            "Use RMSSE for zero-containing series when squared-loss sensitivity, a conditional-mean "
            "target, and a declared naive or seasonal-naive scale match the comparison goal."
        ),
        limitations=(
            "RMSSE is sensitive to large misses and to the selected historical scale period. It "
            "is undefined for a constant training benchmark, and Chrono Stream's per-series value "
            "must not be described as the M5 competition's weighted aggregate score."
        ),
        citation_ready=(
            "Root mean squared scaled error normalizes out-of-sample squared error by the "
            "in-sample squared error of a naive benchmark. The M5 Accuracy competition used an "
            "RMSSE-based score for intermittent retail series (Makridakis et al., 2022)."
        ),
        references=(MAKRIDAKIS_SPILIOTIS_ASSIMAKOPOULOS_2022, HYNDMAN_KOEHLER_2006, GNEITING_2011),
    ),
    "WAPE": MetricInformation(
        display_name="Weighted Absolute Percentage Error (WAPE)",
        formula="WAPE = 100 × sum(|yₜ − ŷₜ|) / sum(|yₜ|)",
        how_it_works=(
            "WAPE aggregates absolute errors before dividing by the aggregate absolute actual. "
            "For nonnegative data it equals MAE divided by mean actual and can be read as an "
            "actual-magnitude-weighted percentage error."
        ),
        chrono_stream=(
            "Chrono Stream uses absolute actuals in the denominator so signed values cannot "
            "cancel. Individual zero actuals are valid, but an all-zero holdout makes WAPE N/A. "
            "Every saved model uses the same holdout denominator."
        ),
        when_to_use=(
            "Use WAPE as an aggregate percentage for a common holdout, particularly when isolated "
            "zero actuals make MAPE undefined but the holdout has nonzero total magnitude."
        ),
        limitations=(
            "WAPE is undefined for an all-zero holdout, weights high-volume periods more heavily, "
            "can hide timing errors, depends on the holdout level, and retains absolute loss's "
            "conditional-median orientation."
        ),
        citation_ready=(
            "WAPE divides total absolute error by total actual magnitude. For nonnegative series "
            "it is the MAD/Mean ratio described by Kolassa and Schütz (2007), allowing individual "
            "zero actuals while remaining undefined when the complete denominator is zero."
        ),
        references=(KOLASSA_SCHUTZ_2007, HYNDMAN_KOEHLER_2006, HEWAMALAGE_ACKERMANN_BERGMEIR_2023),
    ),
}


def copy_ready_metric_note(metric_key: str) -> str:
    """Return a complete plain-text note for one accuracy metric."""
    information = METRIC_INFORMATION[metric_key]
    sections = [
        information.display_name,
        f"Overview\n{information.citation_ready}",
        f"Formula\n{information.formula}",
        f"Literature review\n{METRIC_LITERATURE_REVIEWS[metric_key]}",
        f"How the metric works\n{information.how_it_works}",
        f"How Chrono Stream calculates it\n{information.chrono_stream}",
        f"Appropriate use\n{information.when_to_use}",
        f"Limitations\n{information.limitations}",
        "References (APA 7)\n"
        + "\n\n".join(reference.apa for reference in information.references),
    ]
    return "\n\n".join(sections)


def copy_ready_metric_handbook() -> str:
    """Return one downloadable note covering the complete metric contract."""
    header = (
        "Chrono Stream forecast accuracy metric handbook\n\n"
        "All measures use the same chronological holdout. MASE and RMSSE obtain "
        "their benchmark scale exclusively from pre-holdout training observations. "
        "No single measure represents every operational loss or data condition."
    )
    notes = [copy_ready_metric_note(key) for key in ACCURACY_METRIC_KEYS]
    return "\n\n".join([header, *notes])


if set(METRIC_INFORMATION) != set(ACCURACY_METRIC_KEYS):
    raise RuntimeError("Metric information must cover the complete accuracy contract.")
if set(METRIC_LITERATURE_REVIEWS) != set(ACCURACY_METRIC_KEYS):
    raise RuntimeError("Metric literature reviews must cover the complete accuracy contract.")
