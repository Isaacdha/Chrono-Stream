"""Audited hypothesis tests and decision aids used by Chrono Stream.

The module keeps statistical claims independent of Streamlit.  Each formal test states
its hypotheses, statistic, reference law, decision rule, assumptions, exact app
behavior, and primary literature.  Heuristics and selection criteria live in the same
catalog but are explicitly marked as non-tests so that they are not assigned fictional
null hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .method_info import (
    AKAIKE_1974,
    ANDERSON_DARLING_1952,
    BOX_JENKINS_1970,
    BOX_PIERCE_1970,
    CANOVA_HANSEN_1995,
    DICKEY_FULLER_1979,
    ENGLE_1982,
    HANNAN_QUINN_1979,
    HEGY_1990,
    HURVICH_TSAI_1989,
    JARQUE_BERA_1980,
    KPSS_1992,
    LILLIEFORS_1967,
    LJUNG_BOX_1978,
    MethodReference,
    OCSB_1988,
    PHILLIPS_PERRON_1988,
    SCHWARZ_1978,
    SHAPIRO_WILK_1965,
    STUDENT_1908,
)


@dataclass(frozen=True)
class StatisticalTestInformation:
    """One formal test or explicitly non-inferential decision aid."""

    name: str
    category: str
    purpose: str
    null_hypothesis: str | None
    alternative_hypothesis: str | None
    statistic: str
    reference_distribution: str
    decision_rule: str
    chrono_stream: str
    interpretation: str
    assumptions_and_caveats: str
    literature_review: str
    references: tuple[MethodReference, ...]

    @property
    def formal(self) -> bool:
        """Whether the item has an inferential null and alternative hypothesis."""
        return self.null_hypothesis is not None


def _source(apa: str, url: str, contribution: str) -> MethodReference:
    return MethodReference(apa=apa, url=url, contribution=contribution)


SAID_DICKEY_1984 = _source(
    "Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in "
    "autoregressive-moving average models of unknown order. Biometrika, 71(3), "
    "599–607. https://doi.org/10.1093/biomet/71.3.599",
    "https://doi.org/10.1093/biomet/71.3.599",
    "Develops the augmented autoregressive approximation that turns the Dickey–Fuller procedure into the ADF test used by the app.",
)
WALD_1943 = _source(
    "Wald, A. (1943). Tests of statistical hypotheses concerning several "
    "parameters when the number of observations is large. Transactions of the "
    "American Mathematical Society, 54(3), 426–482. "
    "https://doi.org/10.1090/S0002-9947-1943-0012401-3",
    "https://doi.org/10.1090/S0002-9947-1943-0012401-3",
    "Establishes the large-sample Wald testing framework underlying coefficient z tests.",
)
BARTLETT_1946 = _source(
    "Bartlett, M. S. (1946). On the theoretical specification and sampling "
    "properties of autocorrelated time-series. Supplement to the Journal of the "
    "Royal Statistical Society, 8(1), 27–41. https://doi.org/10.2307/2983611",
    "https://doi.org/10.2307/2983611",
    "Develops large-sample sampling results for time-series autocorrelations and the approximate correlation bands used in identification.",
)
QUENOUILLE_1949 = _source(
    "Quenouille, M. H. (1949). Approximate tests of correlation in time-series. "
    "Journal of the Royal Statistical Society: Series B (Methodological), 11(1), "
    "68–84. https://doi.org/10.1111/j.2517-6161.1949.tb00023.x",
    "https://doi.org/10.1111/j.2517-6161.1949.tb00023.x",
    "Develops approximate correlation tests and autoregressive goodness-of-fit reasoning relevant to ACF/PACF identification.",
)
STEPHENS_1974 = _source(
    "Stephens, M. A. (1974). EDF statistics for goodness of fit and some "
    "comparisons. Journal of the American Statistical Association, 69(347), "
    "730–737. https://doi.org/10.1080/01621459.1974.10480196",
    "https://doi.org/10.1080/01621459.1974.10480196",
    "Develops practical empirical-distribution-function statistics and parameter-estimation adjustments used for normal goodness-of-fit critical values.",
)
ROYSTON_1995 = _source(
    "Royston, P. (1995). Remark AS R94: A remark on Algorithm AS 181: The W-test "
    "for normality. Journal of the Royal Statistical Society: Series C (Applied "
    "Statistics), 44(4), 547–551. https://doi.org/10.2307/2986146",
    "https://doi.org/10.2307/2986146",
    "Provides the practical Shapiro–Wilk W and p-value approximations used by modern numerical implementations.",
)
NEWEY_WEST_1987 = _source(
    "Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, "
    "heteroskedasticity and autocorrelation consistent covariance matrix. "
    "Econometrica, 55(3), 703–708. https://doi.org/10.2307/1913610",
    "https://doi.org/10.2307/1913610",
    "Provides a foundational long-run covariance estimator closely related to the nuisance-parameter correction used by Phillips–Perron tests.",
)
TASHMAN_2000 = _source(
    "Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An "
    "analysis and review. International Journal of Forecasting, 16(4), 437–450. "
    "https://doi.org/10.1016/S0169-2070(00)00065-0",
    "https://doi.org/10.1016/S0169-2070(00)00065-0",
    "Analyzes rolling-origin evaluation, refitting, multiple origins, and their role in selecting forecasting methods.",
)


STATISTICAL_TESTS: dict[str, StatisticalTestInformation] = {
    "adf": StatisticalTestInformation(
        name="Augmented Dickey–Fuller (ADF) unit-root test",
        category="Mean-stationarity test",
        purpose="Assess whether regular differencing is needed because the series contains a unit root.",
        null_hypothesis="H0: The tested series has a unit root (it is nonstationary in level under the selected deterministic specification).",
        alternative_hypothesis="H1: The tested series is stationary around the included deterministic component; in this app that component is a constant.",
        statistic="The ADF tau statistic is the t-ratio for the lagged level coefficient in an augmented regression of the first difference on the lagged level and lagged differences.",
        reference_distribution="Under H0 the statistic has a nonstandard Dickey–Fuller distribution, not an ordinary Student t distribution. MacKinnon-style p-values and critical values are used; rejection is in the left tail.",
        decision_rule="Reject H0 when p < alpha; equivalently, reject at a tabulated level when tau is more negative than that level's critical value. Otherwise fail to reject H0. Do not write that H0 was accepted.",
        chrono_stream="The ARIMA/SARIMA workflow calls statsmodels.adfuller with its constant-only default and autolag='AIC'. It declares the tested difference stationary only when p < the user-selected stationarity alpha. The Data Exploration page uses the same AIC lag selection with alpha fixed at .05. The app reports tau, p, lags, observations, and the 5% critical value in the ARIMA history.",
        interpretation="Rejection is evidence against a unit root under this regression specification; failure to reject says the data did not supply enough evidence to rule one out. It is not proof that differencing is correct, and rejection is not proof of every form of stationarity.",
        assumptions_and_caveats="Lag augmentation must adequately absorb serial correlation, the deterministic terms must be appropriate, and breaks or nonlinear trends can distort the test. ADF may have low power against persistent stationary alternatives. The app's constant-only regression can reject or fail differently from a trend-including ADF, so visibly trend-stationary series require judgment.",
        literature_review="Dickey and Fuller (1979) derived nonstandard unit-root asymptotics for autoregressive models. Said and Dickey (1984) extended the approach to autoregressive-moving-average processes of unknown order by approximating them with an augmented autoregression, which is the basis of the modern ADF regression. The lag terms address serially correlated disturbances but do not convert the tau statistic into an ordinary t statistic. ADF is commonly paired with a stationarity-null test because failure to reject a unit root may reflect low power rather than genuine integration.",
        references=(DICKEY_FULLER_1979, SAID_DICKEY_1984),
    ),
    "kpss": StatisticalTestInformation(
        name="Kwiatkowski–Phillips–Schmidt–Shin (KPSS) stationarity test",
        category="Mean-stationarity test",
        purpose="Test stationarity directly and complement unit-root-null procedures such as ADF.",
        null_hypothesis="H0: The tested series is level-stationary around a constant (the random-walk component has zero variance).",
        alternative_hypothesis="H1: The series contains a unit root and is difference-stationary.",
        statistic="An LM-type statistic formed from cumulative residual sums divided by a heteroskedasticity-and-autocorrelation-consistent estimate of long-run variance.",
        reference_distribution="The null distribution is nonstandard and is evaluated with KPSS critical values. Large positive statistics are evidence against stationarity, so rejection is in the right tail.",
        decision_rule="Reject H0 when p < alpha or when the statistic exceeds the alpha critical value. Chrono Stream treats p > alpha as a stationarity pass; p <= alpha does not pass. A pass means fail to reject stationarity, not prove it.",
        chrono_stream="The app calls statsmodels.kpss with regression='c' and nlags='auto'. It reports the LM statistic, interpolated/tabulated p-value, selected long-run-variance lag, observations, and 5% critical value. Warnings that the p-value lies outside the available table are suppressed, so boundary p-values should be read cautiously.",
        interpretation="A small p-value is evidence that level stationarity is inadequate. A large p-value means the sample is compatible with stationarity at alpha, but near-unit-root behavior or low power can still matter for forecasting.",
        assumptions_and_caveats="The constant-versus-trend specification matters, as does long-run variance bandwidth. Structural breaks can cause rejection even when segments are stationary. Results should be reported with the deterministic specification, lag rule, statistic, p-value range, and alpha.",
        literature_review="Kwiatkowski et al. (1992) reversed the usual unit-root testing logic by defining stationarity as H0 and expressing the observed process as deterministic component, random walk, and stationary error. The LM statistic tests whether the random-walk variance is zero. This opposing null makes KPSS especially useful with ADF: their joint outcomes distinguish evidence supporting stationarity from cases where both tests are inconclusive or mutually rejecting, although the combination is a decision protocol rather than a new formal test.",
        references=(KPSS_1992, DICKEY_FULLER_1979),
    ),
    "pp": StatisticalTestInformation(
        name="Phillips–Perron (PP) unit-root test",
        category="Mean-stationarity test",
        purpose="Assess a unit root while correcting the Dickey–Fuller regression nonparametrically for weak dependence and heteroskedasticity.",
        null_hypothesis="H0: The tested series has a unit root.",
        alternative_hypothesis="H1: The series is stationary around the deterministic terms included by the implementation.",
        statistic="The installed pmdarima backend uses the PP Z(alpha) statistic: a scaled autoregressive coefficient correction using short-lag and long-run residual variance estimates.",
        reference_distribution="The corrected statistic has a nonstandard unit-root null distribution. pmdarima interpolates simulation/tabulation-based p-values; the app does not pretend it follows an ordinary z or t law.",
        decision_rule="Reject H0 when p <= alpha in the exact pmdarima should_diff decision used here; p > alpha recommends another difference. The conventional prose remains 'reject' versus 'fail to reject,' never 'accept.'",
        chrono_stream="Chrono Stream calls pmdarima.arima.PPTest(alpha=alpha).should_diff with the default short bandwidth. That implementation fits a constant and centered trend, returns a p-value and Boolean differencing decision, but does not expose its internal Z(alpha) statistic through the public result. The app therefore reports the p-value and explicitly records the statistic as unavailable instead of fabricating one.",
        interpretation="A small p-value is evidence against a unit root under the PP deterministic specification. A large p-value supports differencing only in the limited sense that the unit-root null was not rejected.",
        assumptions_and_caveats="PP relies on asymptotic long-run-variance corrections and can show size distortion or low power in finite samples, especially with strong serial correlation or breaks. Its deterministic terms differ from the app's constant-only ADF, so disagreement can reflect specification as well as sampling variation.",
        literature_review="Phillips and Perron (1988) developed unit-root tests that retain a simple Dickey–Fuller-type regression while correcting the statistic nonparametrically for serial correlation and heteroskedasticity. Long-run covariance estimation, also central to Newey and West (1987), makes the method robust to a wider disturbance class asymptotically. That robustness is not a guarantee of good small-sample behavior. The app documents the exact Z(alpha) implementation and its public-output limitation so a result can be reproduced without falsely labeling its statistic as a standard z score.",
        references=(PHILLIPS_PERRON_1988, NEWEY_WEST_1987),
    ),
    "ocsb": StatisticalTestInformation(
        name="Osborn–Chui–Smith–Birchenhall (OCSB) seasonal unit-root test",
        category="Seasonal-differencing test",
        purpose="Determine whether a SARIMA model should include seasonal differencing at the declared period m.",
        null_hypothesis="H0: A seasonal unit root is present, so a seasonal difference is required.",
        alternative_hypothesis="H1: The series is stationary with respect to the tested seasonal root, so that seasonal difference is not required.",
        statistic="A one-sided t-type statistic for the relevant filtered seasonal-regression coefficient after optional autoregressive lag augmentation.",
        reference_distribution="The statistic does not use an ordinary t critical value. Period-specific 5% critical values are obtained from a smoothed simulation approximation in the pmdarima/forecast implementation.",
        decision_rule="For the implemented orientation, a statistic at or below the critical value rejects the seasonal-unit-root H0; a statistic above the critical value fails to reject H0 and pmdarima returns D=1. This backend does not expose a user-adjustable alpha or p-value for OCSB.",
        chrono_stream="Chrono Stream calls pmdarima.nsdiffs(test='ocsb', max_D=user limit) with the declared seasonal period. The backend selects regression augmentation by AIC up to its default maximum lag and iterates until D reaches the cap. The current result records selected D, period, and method; the handbook states that the embedded critical value is a 5% simulation threshold even if the separate regular-stationarity alpha slider differs.",
        interpretation="D=1 means the 5% OCSB procedure did not reject the seasonal unit root and recommends seasonal differencing. D=0 means it rejected that unit root or the test could not proceed under a base-case rule; it does not prove the absence of every kind of seasonality.",
        assumptions_and_caveats="The period must be correct and the series long enough after differencing. Deterministic seasonality, evolving seasonality, outliers, or multiple cycles can affect the regression. The OCSB decision should be compared with the seasonal plot, seasonal ACF, and an alternative-null test such as Canova–Hansen.",
        literature_review="Osborn et al. (1988) studied how seasonal and nonseasonal integration enter consumption models and developed the regression structure behind the OCSB procedure. The test takes a seasonal unit root as H0 and uses nonstandard one-sided critical values. It complements Canova–Hansen, whose null is stable deterministic seasonality, and HEGY-style tests that decompose individual seasonal frequencies (Hylleberg et al., 1990). Because the available OCSB simulation threshold is fixed at 5%, a report should not imply that the app's general alpha control changed this seasonal decision.",
        references=(OCSB_1988, HEGY_1990, CANOVA_HANSEN_1995),
    ),
    "canova_hansen": StatisticalTestInformation(
        name="Canova–Hansen seasonal-stability test",
        category="Seasonal-differencing test",
        purpose="Assess whether an apparently deterministic seasonal pattern is stable or contains stochastic seasonal instability.",
        null_hypothesis="H0: The seasonal pattern is stable/deterministic; equivalently, there is no unit root at the jointly tested seasonal frequencies.",
        alternative_hypothesis="H1: Seasonality is unstable because a unit root occurs at one or more tested seasonal frequencies.",
        statistic="A Lagrange-multiplier seasonal-stability statistic based on cumulative seasonal-regressor residual products and a long-run covariance estimate.",
        reference_distribution="The pmdarima implementation compares the statistic with period-specific critical values (tabulated for common periods and approximated otherwise). Large statistics reject stable seasonality.",
        decision_rule="Reject H0 and select D=1 when the CH statistic is greater than its period-specific critical value. Otherwise fail to reject stable seasonality and select D=0. The pmdarima API used here returns D rather than a p-value.",
        chrono_stream="Chrono Stream calls pmdarima.nsdiffs(test='ch', max_D=user limit). The implementation uses trigonometric seasonal regressors, a period-dependent long-run-variance truncation, and its embedded critical-value table/function. The app reports the selected D and method; alpha is not supplied to this backend.",
        interpretation="A rejection is evidence that fixed seasonal indices are not stable and supports seasonal differencing. Failure to reject means the data remain compatible with stable deterministic seasonality, not that seasonal effects are absent.",
        assumptions_and_caveats="The test is sensitive to period definition, sample length, deterministic specification, and structural breaks. For uncommon periods, the software uses an approximated critical function. A CH decision and OCSB decision can legitimately disagree because their null hypotheses are reversed.",
        literature_review="Canova and Hansen (1995) introduced LM tests of no seasonal unit roots against instability at individual or groups of seasonal frequencies, deriving asymptotic theory and studying size and power by simulation. This reversed the null used by seasonal-unit-root tests such as OCSB and complemented frequency-specific HEGY procedures (Hylleberg et al., 1990). In forecasting automation, a large CH statistic is often translated into a recommendation to seasonally difference. That translation should always retain the original meaning: the evidence is against stable deterministic seasonality, not merely evidence that a seasonal plot has peaks.",
        references=(CANOVA_HANSEN_1995, HEGY_1990, OCSB_1988),
    ),
    "coefficient_wald": StatisticalTestInformation(
        name="Individual coefficient Wald z test",
        category="Parameter-significance test",
        purpose="Assess whether each estimated AR, MA, seasonal, intercept, or trend coefficient differs from zero.",
        null_hypothesis="H0: The individual model coefficient beta_j equals zero.",
        alternative_hypothesis="H1: beta_j is not zero (two-sided).",
        statistic="z_j = beta_hat_j / SE(beta_hat_j), with a confidence interval from the same covariance estimate.",
        reference_distribution="Under regular maximum-likelihood conditions and large samples, z_j is approximately standard normal under H0; equivalently z_j squared is approximately chi-square with one degree of freedom.",
        decision_rule="Reject H0 when p < alpha or |z_j| > z_(1-alpha/2). Otherwise fail to reject H0. Under a strict/custom significance gate, Chrono Stream requires every evaluated coefficient to reject H0.",
        chrono_stream="The SARIMAX result supplies estimates, standard errors, asymptotic p-values, and confidence intervals. Chrono Stream displays its independently calculated estimate/SE z ratio and gates every nonvariance parameter at the selected diagnostic alpha. sigma2/variance is reported but deliberately excluded from this significance gate.",
        interpretation="Rejection indicates that the coefficient is distinguishable from zero conditional on the fitted model and covariance approximation. Failure to reject does not show that the corresponding lag has no predictive role, especially with correlated parameters or small samples.",
        assumptions_and_caveats="The approximation requires identification, an interior parameter, a reliable Hessian/covariance matrix, sufficient sample size, and a correctly specified likelihood. Testing many correlated coefficients inflates the chance of at least one rejection or failure, and requiring every term to be significant can discard jointly useful hierarchical/seasonal models. Forecast-oriented policies therefore keep this result advisory.",
        literature_review="Wald (1943) developed large-sample tests based on the distance between an estimator and a null restriction relative to estimated sampling variation. Box and Jenkins (1970) made parameter estimation and parsimony part of ARIMA identification and checking. Modern SARIMAX summaries apply the Wald logic coefficient by coefficient. Individual significance is informative but is not, by itself, a complete model adequacy criterion: likelihood geometry, polynomial roots, joint restrictions, residual whiteness, and out-of-sample forecasting address different questions.",
        references=(WALD_1943, BOX_JENKINS_1970),
    ),
    "residual_mean_t": StatisticalTestInformation(
        name="One-sample t test for zero residual mean",
        category="Residual-location test",
        purpose="Check whether standardized forecast errors retain a systematic nonzero mean.",
        null_hypothesis="H0: The population mean of the usable standardized residuals is zero.",
        alternative_hypothesis="H1: The residual mean is not zero (two-sided).",
        statistic="t = residual_mean / (residual_sample_sd / sqrt(n)).",
        reference_distribution="For independent normal observations with unknown variance, t follows Student's t distribution with n-1 degrees of freedom under H0.",
        decision_rule="Reject H0 when p < alpha or |t| exceeds the two-sided t critical value. Chrono Stream marks the diagnostic as passed only when p > alpha; otherwise it does not pass. A pass is failure to reject, not acceptance of exact zero bias.",
        chrono_stream="The app removes state-space burn-in/nonfinite errors, uses standardized one-step forecast errors, and calls scipy.stats.ttest_1samp(popmean=0). It reports residual mean, t statistic, p-value, and the selected diagnostic alpha.",
        interpretation="Rejection indicates a detectable average signed error in sample. Failure to reject says the estimated mean is not distinguishable from zero at alpha, but economically important bias may still be hidden by low power.",
        assumptions_and_caveats="The exact small-sample t law assumes independent normal observations. Estimated time-series residuals are constrained by model fitting and may be autocorrelated or nonnormal, so this is best read alongside white-noise and distribution diagnostics. Statistical and practical significance are not identical.",
        literature_review="Student (1908) derived inference for a mean when variance is estimated from a small normal sample. Box–Jenkins diagnostic checking seeks innovations without remaining systematic structure, making zero residual mean a natural location check. In fitted ARIMA models the textbook t assumptions are only approximate because parameters were estimated and residuals are time ordered. The app therefore exposes the statistic and p-value as one diagnostic gate rather than treating it as proof of unbiased future forecasts.",
        references=(STUDENT_1908, BOX_JENKINS_1970),
    ),
    "jarque_bera": StatisticalTestInformation(
        name="Jarque–Bera residual-normality test",
        category="Residual-distribution test",
        purpose="Test whether residual skewness and kurtosis match a normal distribution.",
        null_hypothesis="H0: The residual distribution is normal, implying population skewness 0 and kurtosis 3.",
        alternative_hypothesis="H1: The residual distribution is nonnormal in skewness, kurtosis, or both.",
        statistic="JB = n[S^2/6 + (K-3)^2/24], where S is sample skewness and K is sample kurtosis under the implementation's moment conventions.",
        reference_distribution="JB is asymptotically chi-square with 2 degrees of freedom under H0. Large values reject normality.",
        decision_rule="Reject H0 when p < alpha or JB > chi-square_(2,1-alpha). Chrono Stream passes the normality gate only when p > alpha.",
        chrono_stream="The app calls scipy.stats.jarque_bera on usable standardized state-space forecast errors and reports JB and its chi-square p-value. At least eight usable residuals are required by the shared app guard.",
        interpretation="Rejection identifies moment-based nonnormality but does not say whether skewness, tails, outliers, or dependence is the primary cause. Failure to reject only says those two moment discrepancies were not detected.",
        assumptions_and_caveats="The chi-square approximation is asymptotic and can be weak in small samples. JB has limited power against alternatives whose skewness and kurtosis resemble normality. Autocorrelation and conditional heteroskedasticity violate the usual independent-sample reasoning, so whiteness and ARCH are tested separately.",
        literature_review="Jarque and Bera (1980) derived efficient Lagrange-multiplier diagnostics for normality, homoscedasticity, and serial independence of regression residuals; the familiar normality statistic combines squared skewness and excess kurtosis. Its attraction is a simple asymptotic chi-square decision, while its limitation is equally clear: it compresses distribution shape into two moments. In ARIMA work, normality mainly supports Gaussian likelihood interpretation and forecast intervals; point forecasts can remain useful after rejection, so the app lets users make this gate advisory.",
        references=(JARQUE_BERA_1980, BOX_JENKINS_1970),
    ),
    "shapiro_wilk": StatisticalTestInformation(
        name="Shapiro–Wilk residual-normality test",
        category="Residual-distribution test",
        purpose="Detect broad departures from normality using the covariance structure of ordered normal observations.",
        null_hypothesis="H0: The residuals are sampled from a normal distribution.",
        alternative_hypothesis="H1: The residual distribution is not normal.",
        statistic="W is a squared weighted combination of ordered observations divided by their centered sum of squares; values near one are more compatible with normality.",
        reference_distribution="W has no simple universal named null distribution. Numerical approximations transform W to obtain a p-value; small W is evidence against normality.",
        decision_rule="Reject H0 when p < alpha. Chrono Stream passes only when p > alpha and otherwise reports failure to pass.",
        chrono_stream="The app calls scipy.stats.shapiro. Because SciPy cautions that p-values may be inaccurate above 5,000 observations, Chrono Stream tests the first 5,000 usable standardized residuals when more are available and reports observations_tested.",
        interpretation="Rejection signals a general distributional mismatch but does not identify its form. Failure to reject is not evidence that tails are exactly Gaussian and can reflect limited power.",
        assumptions_and_caveats="The classical test assumes independent observations. Using the first 5,000 chronological residuals is a transparent computational convention, not a random subsample, and may miss later regime changes. Very large samples can make substantively minor deviations significant, while small samples may miss important tail behavior.",
        literature_review="Shapiro and Wilk (1965) constructed W from the expected values and covariance of normal order statistics and demonstrated strong omnibus power for complete samples. Royston (1995) refined practical algorithms and approximations used by modern implementations. The test addresses marginal distribution, not temporal independence. In a forecast model it should therefore be combined with residual ACF and portmanteau tests, and a Q–Q plot should be used to see whether rejection comes from skew, tails, or isolated observations.",
        references=(SHAPIRO_WILK_1965, ROYSTON_1995),
    ),
    "anderson_darling": StatisticalTestInformation(
        name="Anderson–Darling residual-normality test",
        category="Residual-distribution test",
        purpose="Compare the empirical residual distribution with normality while giving relatively high weight to tail discrepancies.",
        null_hypothesis="H0: The residuals follow a normal distribution with location and scale estimated as required by the implementation.",
        alternative_hypothesis="H1: The residual distribution is not normal.",
        statistic="A^2 is a weighted integral (or ordered-sample sum) of squared empirical-versus-normal CDF differences, with weight greatest near probabilities zero and one.",
        reference_distribution="The null law depends on the target distribution and parameter estimation. SciPy supplies normal-specific critical values at discrete significance levels rather than a p-value.",
        decision_rule="Reject H0 when A^2 is greater than the critical value for the selected significance level. Chrono Stream passes only when A^2 < critical; equality does not pass.",
        chrono_stream="The app calls scipy.stats.anderson(dist='norm'). It chooses the available significance level closest to the user's diagnostic alpha from SciPy's table, reports that tested_significance and critical_value, and deliberately reports p_value=None. Thus an alpha of .04 may be evaluated at the nearest tabulated level rather than exactly .04.",
        interpretation="Rejection indicates a distributional discrepancy with particular sensitivity to tails. Failure to reject says the empirical discrepancy stayed below one tabulated threshold; it does not estimate the probability that normality is true.",
        assumptions_and_caveats="Residual independence is not tested by A^2. Critical values must match both the target distribution and whether parameters were estimated. The discrete-level rule makes the actual tested alpha essential to report, especially when it differs from the slider value.",
        literature_review="Anderson and Darling (1952) developed weighted empirical-distribution-function goodness-of-fit criteria whose weighting emphasizes distribution tails. Stephens (1974) compared EDF statistics and developed practical adjustments and critical values when distribution parameters are estimated. This makes Anderson–Darling attractive when forecast intervals depend on tail behavior, but it remains a marginal goodness-of-fit test. Chrono Stream preserves the critical-value form rather than inventing a p-value and displays the actual tabulated significance level used.",
        references=(ANDERSON_DARLING_1952, STEPHENS_1974),
    ),
    "lilliefors": StatisticalTestInformation(
        name="Lilliefors normality test",
        category="Residual-distribution test",
        purpose="Apply a Kolmogorov–Smirnov-type normality test when mean and variance are estimated from the same residual sample.",
        null_hypothesis="H0: The residuals come from some normal distribution whose mean and variance are unspecified and estimated from the data.",
        alternative_hypothesis="H1: The residual distribution is not normal.",
        statistic="D is the maximum absolute distance between the empirical CDF and the normal CDF fitted with sample location and scale.",
        reference_distribution="The ordinary one-sample Kolmogorov–Smirnov table is invalid after estimating mean and variance. Lilliefors critical values/p-values are obtained from the adjusted null distribution.",
        decision_rule="Reject H0 when p < alpha or D exceeds the Lilliefors critical value. Chrono Stream passes only when p > alpha.",
        chrono_stream="The app calls statsmodels.stats.diagnostic.lilliefors(residuals, dist='norm') and reports D and its approximated/table-based p-value on all usable standardized residuals.",
        interpretation="Rejection means the fitted normal CDF is too far from the empirical CDF under the Lilliefors calibration. Failure to reject does not establish normal tails or temporal independence.",
        assumptions_and_caveats="The test assumes an independent sample under H0. It can be less tail-focused than Anderson–Darling, and p-values can be capped or approximated by software tables. Estimating model parameters before forming residuals adds uncertainty beyond merely estimating normal mean and variance.",
        literature_review="Lilliefors (1967) showed that standard Kolmogorov–Smirnov critical values are incorrect when normal mean and variance are estimated, and supplied Monte Carlo critical values for the adjusted problem. Stephens (1974) situated such EDF tests within a broader comparison of goodness-of-fit statistics. The Lilliefors test therefore answers a more realistic normality question than a KS test against a completely specified N(0,1), but it still tests the marginal residual distribution rather than whether errors form a white-noise process.",
        references=(LILLIEFORS_1967, STEPHENS_1974),
    ),
    "ljung_box": StatisticalTestInformation(
        name="Ljung–Box portmanteau test",
        category="Residual white-noise test",
        purpose="Test residual autocorrelations jointly through one or more maximum lags.",
        null_hypothesis="H0: At lag h, residual autocorrelations rho_1,...,rho_h are jointly zero.",
        alternative_hypothesis="H1: At lag h, at least one residual autocorrelation through h is nonzero.",
        statistic="Q_LB = n(n+2) * sum_{k=1}^h [r_k^2/(n-k)].",
        reference_distribution="For an adequately fitted model, Q_LB is asymptotically chi-square with h - model_df degrees of freedom, where Chrono Stream uses model_df=p+q+P+Q.",
        decision_rule="Reject H0 at a displayed h when p < alpha or Q_LB exceeds the corresponding chi-square critical value. The app's multi-lag gate passes only if every displayed lag has p > alpha.",
        chrono_stream="Chrono Stream calls statsmodels.acorr_ljungbox with model_df=p+q+P+Q. Automatic lags are [10,20] for ARIMA and [10,m,2m] for SARIMA, clipped by sample size and retained only when h>model_df; users may supply lags. The table reports h, Q, p, and pass status.",
        interpretation="Rejection indicates remaining linear serial correlation somewhere through h and therefore model inadequacy for white-noise innovations. Failure to reject says the selected residual autocorrelations were not jointly detectable; it does not prove independence or rule out nonlinear dependence.",
        assumptions_and_caveats="The chi-square calibration is asymptotic, depends on parameter estimation and degrees-of-freedom adjustment, and may be inaccurate in small samples or under conditional heteroskedasticity. Results depend on h. Requiring several cumulative lags to pass is conservative and involves multiple overlapping tests.",
        literature_review="Box and Pierce (1970) derived the distribution of residual autocorrelations for fitted ARIMA models and proposed a portmanteau lack-of-fit statistic. Ljung and Box (1978) modified its finite-sample scaling and showed a substantially improved chi-square approximation. The test is central to Box–Jenkins diagnostic checking because ARIMA aims to transform data into innovations without residual autocorrelation. It is a joint zero-correlation test, not a test of Gaussianity and not a complete test of statistical independence.",
        references=(BOX_PIERCE_1970, LJUNG_BOX_1978),
    ),
    "box_pierce": StatisticalTestInformation(
        name="Box–Pierce portmanteau test",
        category="Residual white-noise test",
        purpose="Test a block of residual autocorrelations for remaining linear dependence.",
        null_hypothesis="H0: At lag h, residual autocorrelations rho_1,...,rho_h are jointly zero.",
        alternative_hypothesis="H1: At lag h, at least one residual autocorrelation through h is nonzero.",
        statistic="Q_BP = n * sum_{k=1}^h r_k^2.",
        reference_distribution="Q_BP is asymptotically chi-square with h - model_df degrees of freedom after allowing for fitted AR/MA terms.",
        decision_rule="Reject H0 when p < alpha or Q_BP exceeds its chi-square critical value. Chrono Stream passes its selected multi-lag gate only when every displayed p-value is greater than alpha.",
        chrono_stream="The app requests boxpierce=True from statsmodels.acorr_ljungbox and reads the Box–Pierce statistic and p-value, using the same automatic/user lags and model_df=p+q+P+Q correction as its Ljung–Box option.",
        interpretation="Rejection signals residual autocorrelation and lack of fit through the chosen horizon. Failure to reject is evidence of no detected linear correlation at those lags, not proof that residuals are independent.",
        assumptions_and_caveats="The original statistic's chi-square approximation can be poor in finite samples, which motivated the Ljung–Box correction. It remains useful for historical comparison, but Ljung–Box is generally preferred when sample size is limited. Lag choice and fitted-parameter correction must be reported.",
        literature_review="Box and Pierce (1970) showed that residual autocorrelations from a fitted ARIMA model do not behave exactly like autocorrelations of known innovations and derived diagnostic checks that account for fitting. Their Q statistic aggregates squared residual correlations. Ljung and Box (1978) demonstrated that a simple n- and lag-dependent rescaling substantially improves finite-sample approximation. Chrono Stream offers both so users can reproduce older analyses while seeing why the later correction is often the default.",
        references=(BOX_PIERCE_1970, LJUNG_BOX_1978),
    ),
    "arch_lm": StatisticalTestInformation(
        name="Engle ARCH Lagrange-multiplier test",
        category="Residual conditional-variance test",
        purpose="Detect whether squared residuals are predictable from their own lags, indicating autoregressive conditional heteroskedasticity.",
        null_hypothesis="H0: All coefficients on the selected lags of squared residuals are zero; there is no ARCH effect through lag L.",
        alternative_hypothesis="H1: At least one lagged-squared-residual coefficient is nonzero, so conditional variance is time dependent.",
        statistic="statsmodels computes LM = (n_aux - ddof) * R^2 from an auxiliary regression of squared residuals on a constant and L lagged squared residuals, where n_aux is the post-lag sample. The auxiliary-regression F statistic tests the same joint restriction in an alternative finite-sample form.",
        reference_distribution="LM is asymptotically chi-square with L degrees of freedom under H0; the auxiliary statistic uses an F distribution with regression restriction and residual degrees of freedom.",
        decision_rule="Reject H0 when the LM p-value < alpha (or LM exceeds chi-square critical); the F version rejects when its p-value < alpha. Chrono Stream's mandatory gate is based on LM p > alpha, while both LM and F results are reported.",
        chrono_stream="The app calls statsmodels.het_arch with L=min(10,floor(n/5)), at least one, and ddof=p+q+P+Q. This ddof argument rescales LM as (n_aux-ddof)*R^2; it does not change the chi-square reference degrees of freedom, which remain L. The app retains LM, LM p, auxiliary F, and F p, and rejects the diagnostic as unavailable when n_aux<=ddof.",
        interpretation="Rejection means residual variance remains systematically predictable even if residual levels are uncorrelated. Failure to reject means no ARCH effect was detected through L, not proof of globally constant or correctly specified variance.",
        assumptions_and_caveats="The LM calibration is asymptotic and sensitive to lag choice, outliers, mean-model misspecification, residual serial correlation, and the optional statistic rescaling for fitted parameters. The F and LM versions can disagree in small samples. Heteroskedasticity chiefly threatens Gaussian interval and likelihood assumptions; it does not automatically erase point-forecast usefulness.",
        literature_review="Engle (1982) introduced ARCH models and the associated LM test by regressing squared disturbances on their lags. The insight separates mean whiteness from variance dynamics: a series can have negligible residual ACF but clusters of large and small errors. For ARIMA diagnostics this is complementary to Ljung–Box, not redundant. Chrono Stream reports both the asymptotic chi-square LM result and the auxiliary F result. Its statsmodels ddof setting reduces the multiplier applied to R-squared for fitted AR/MA terms while retaining L chi-square degrees of freedom; the optional gate is based on that LM p-value.",
        references=(ENGLE_1982, BOX_JENKINS_1970),
    ),
    "acf_lag": StatisticalTestInformation(
        name="Approximate pointwise ACF lag test",
        category="Approximate correlation identification test",
        purpose="Flag individual autocorrelations that are large relative to a white-noise sampling approximation and guide q/Q candidates.",
        null_hypothesis="H0: At lag k, the population autocorrelation rho_k is zero under a white-noise approximation.",
        alternative_hypothesis="H1: At lag k, rho_k is nonzero (two-sided).",
        statistic="Approximate z_k = sqrt(n) * r_k.",
        reference_distribution="For white noise and fixed k, z_k is approximately standard normal, yielding a pointwise 95% band of about +/-1.96/sqrt(n). General stationary-process ACF variances follow Bartlett-type formulas instead.",
        decision_rule="At the app's fixed alpha=.05 (95% reference) level, flag lag k when |r_k| > 1.96/sqrt(n). This is a pointwise white-noise heuristic, not a simultaneous band or a universal significance test for a general autocorrelated process.",
        chrono_stream="The transformed/differenced series and fitted residuals are analyzed with statsmodels ACF. Chrono Stream records the fixed +/-1.96/sqrt(n) heuristic and uses flagged low lags to seed q and flagged multiples of m to seed Q in guided search. The visible charts and records call this a reference band, and it does not change with the diagnostic-alpha slider.",
        interpretation="A flagged lag is identification evidence of linear correlation, not proof that an MA term at that exact lag is required. A nonflagged lag may still contribute jointly or under a different model.",
        assumptions_and_caveats="Bands are approximate and pointwise; inspecting many lags creates multiplicity. Estimated trends, differencing, parameter fitting, and nonwhite data alter sampling variance. ACF cutoff/tail patterns are heuristics whose textbook forms are clearest for ideal low-order processes.",
        literature_review="Bartlett (1946) developed the sampling theory of autocorrelations for stationary time series, while Quenouille (1949) developed approximate correlation tests and autoregressive goodness-of-fit procedures. Box and Jenkins (1970) integrated the ACF into ARIMA identification and residual checking. Modern +/-1.96/sqrt(n) lines are a convenient white-noise approximation, not a universal significance theorem; Chrono Stream consequently uses them to generate candidates and still requires fitted-model diagnostics.",
        references=(BARTLETT_1946, QUENOUILLE_1949, BOX_JENKINS_1970),
    ),
    "pacf_lag": StatisticalTestInformation(
        name="Approximate pointwise PACF lag test",
        category="Approximate correlation identification test",
        purpose="Flag partial autocorrelations after controlling intervening lags and guide p/P candidates.",
        null_hypothesis="H0: At lag k, the population partial autocorrelation is zero under the identification approximation.",
        alternative_hypothesis="H1: At lag k, the partial autocorrelation is nonzero (two-sided).",
        statistic="Approximate z_k = sqrt(n) * partial_r_k, using the Yule–Walker modified PACF estimate in this app.",
        reference_distribution="A standard-normal approximation gives pointwise 95% limits around +/-1.96/sqrt(n), especially for lags beyond the order of an ideal autoregression.",
        decision_rule="At the fixed pointwise alpha=.05 level, flag lag k when |PACF_k| > 1.96/sqrt(n). This approximation is not corrected for examining many lags.",
        chrono_stream="Chrono Stream computes statsmodels.pacf(method='ywm'), records the fixed 95% bounds, and uses flagged low lags to seed p and flagged seasonal multiples to seed P. It does not force the final order to equal the last flagged lag; candidate fitting and diagnostics remain decisive.",
        interpretation="A flag suggests incremental linear association after intervening lags and is useful for AR identification. It does not establish causality or uniquely determine an AR order.",
        assumptions_and_caveats="The approximation depends on stationarity, sample size, estimator, and model context. Differencing and selection change sampling behavior, and simultaneous inspection inflates false positives. Mixed ARMA processes rarely display perfect theoretical cutoffs in finite data.",
        literature_review="Quenouille (1949) developed approximate tests of time-series correlation and large-sample autoregressive checks; Bartlett (1946) provided foundational autocorrelation sampling theory. Box and Jenkins (1970) made the contrast between ACF and PACF patterns a practical identification tool for AR and MA orders. Chrono Stream preserves that role but avoids the common overclaim that a PACF plot mechanically identifies a unique p: it guides a bounded candidate set that must survive estimation, root checks, and residual diagnostics.",
        references=(QUENOUILLE_1949, BARTLETT_1946, BOX_JENKINS_1970),
    ),
    "adf_kpss_consensus": StatisticalTestInformation(
        name="ADF + KPSS consensus protocol",
        category="Decision protocol — not a hypothesis test",
        purpose="Combine opposing null hypotheses before stopping regular differencing.",
        null_hypothesis=None,
        alternative_hypothesis=None,
        statistic="No combined statistic is calculated. The protocol retains the separate ADF tau/p and KPSS LM/p results.",
        reference_distribution="Not applicable: ADF and KPSS keep their own nonstandard null distributions.",
        decision_rule="Chrono Stream declares consensus stationarity only when ADF rejects its unit-root H0 (p<alpha) AND KPSS fails to reject its stationarity H0 (p>alpha). Other combinations are inconclusive/nonpassing and can trigger another difference up to max_d.",
        chrono_stream="Both tests run on the same seasonally and regularly differenced candidate series. Their records remain separate in the stationarity-history table; the app never merges their p-values or calls the protocol a new test.",
        interpretation="Agreement provides complementary evidence for stationarity. ADF fail/KPSS reject supports nonstationarity; ADF reject/KPSS reject may indicate breaks, deterministic misspecification, or other nonstationarity; both fail may reflect low power and a series compatible with several explanations.",
        assumptions_and_caveats="Requiring agreement is conservative and does not control a single combined Type I error. Both component tests can be wrong for the same structural break or deterministic misspecification. The result is evidence for a modeling decision, not proof of covariance stationarity.",
        literature_review="Dickey–Fuller-type tests begin from a unit-root null, whereas Kwiatkowski et al. (1992) begin from a stationarity null. Reading them together is useful precisely because 'failure to reject' is not acceptance. The four possible outcomes expose ambiguity that a single p-value hides. Chrono Stream's AND rule is an explicit conservative protocol authored for this workflow, not a procedure claimed by either original paper.",
        references=(DICKEY_FULLER_1979, SAID_DICKEY_1984, KPSS_1992),
    ),
    "seasonal_acf_rule": StatisticalTestInformation(
        name="Seasonal-lag ACF differencing rule",
        category="Heuristic — not a seasonal unit-root test",
        purpose="Offer a transparent nonparametric trigger for seasonal differencing when the user does not select OCSB or Canova–Hansen.",
        null_hypothesis=None,
        alternative_hypothesis=None,
        statistic="The ordinary sample correlation between y_t and y_(t-m) on the current series.",
        reference_distribution="No calibrated reference distribution is assigned to the combined rule.",
        decision_rule="Add one seasonal difference while r_m > max(0.30, 1.96/sqrt(n)), enough observations remain, and D<max_D. Stop otherwise.",
        chrono_stream="The rule is recomputed after each seasonal difference. Only a positive seasonal correlation triggers differencing; the fixed .30 effect-size floor is combined with the approximate pointwise 95% white-noise band.",
        interpretation="A trigger says seasonal persistence is large under this rule. It does not establish a seasonal unit root, and a stop does not establish stable deterministic seasonality.",
        assumptions_and_caveats="The .30 floor is an app heuristic, not a paper-derived critical value. Trends and nonseasonal persistence can inflate r_m; negative seasonal correlation and multiple seasonalities are not handled. OCSB/CH are preferable when formal seasonal-integration inference is required.",
        literature_review="Seasonal ACF inspection follows the Box–Jenkins identification tradition and the correlation bands draw on large-sample autocorrelation theory (Bartlett, 1946; Box & Jenkins, 1970). Chrono Stream adds a declared 0.30 floor to avoid differencing on a statistically flagged but tiny correlation. Because that composite cutoff has no unified null distribution, the UI and exported handbook label it a heuristic rather than dressing it as a formal test.",
        references=(BARTLETT_1946, BOX_JENKINS_1970),
    ),
    "roots": StatisticalTestInformation(
        name="AR stationarity and MA invertibility root check",
        category="Deterministic model-property check — not a hypothesis test",
        purpose="Verify that the fitted AR and MA polynomials satisfy stationarity and invertibility conditions.",
        null_hypothesis=None,
        alternative_hypothesis=None,
        statistic="Magnitudes of the complex roots of the fitted AR and MA lag polynomials.",
        reference_distribution="None. This is an algebraic property of the fitted parameter vector.",
        decision_rule="Pass when every AR and MA root magnitude is greater than 1 + the configured numerical tolerance. Empty AR or MA root sets pass their respective condition.",
        chrono_stream="The app reads statsmodels SARIMAX arroots and maroots, reports all magnitudes, and uses a default tolerance of 0.001 to keep borderline numerical roots from being treated as safely outside the unit circle.",
        interpretation="Passing means the estimated representation is stationary/invertible under the app's polynomial convention. It says nothing about coefficient uncertainty or future structural stability.",
        assumptions_and_caveats="A root barely outside the boundary may be practically persistent and uncertain. Root checks do not replace pre-fit unit-root assessment: one concerns the fitted ARMA polynomials, the other concerns how the observed series should be differenced.",
        literature_review="Box and Jenkins (1970) formalized stationary autoregressive and invertible moving-average representations as core conditions for identifiable ARIMA modeling. Root magnitude is the direct algebraic check. It has no p-value unless a separate uncertainty procedure is constructed, so labeling it H0/H1 would be misleading. Chrono Stream reports both the values and its tolerance instead.",
        references=(BOX_JENKINS_1970,),
    ),
    "optimizer_convergence": StatisticalTestInformation(
        name="Maximum-likelihood optimizer convergence status",
        category="Numerical status — not a hypothesis test",
        purpose="Detect whether the numerical fitting routine reported successful convergence.",
        null_hypothesis=None,
        alternative_hypothesis=None,
        statistic="The optimizer's convergence flag and return metadata, not a statistical test statistic.",
        reference_distribution="None.",
        decision_rule="Pass only when the statsmodels maximum-likelihood result reports converged=True. A converged result can still be a local optimum or a poor statistical model.",
        chrono_stream="Convergence is a mandatory strict/forecast-oriented gate and can be advisory under other policies. The maximum iteration control changes the optimizer budget, not an inferential alpha.",
        interpretation="Failure means parameter estimates and diagnostics should not be trusted as a completed optimum. Success means only that the numerical stopping criterion was satisfied.",
        assumptions_and_caveats="Convergence flags depend on starting values, scaling, parameterization, tolerance, and likelihood geometry. Multiple fits or alternative specifications may be needed for difficult seasonal models.",
        literature_review="Box–Jenkins estimation presumes that a candidate can actually be fitted before its adequacy is judged (Box & Jenkins, 1970). Modern numerical convergence is implementation metadata rather than an inferential statement about the data-generating process. Chrono Stream separates it from hypothesis tests and never translates a success flag into model validity.",
        references=(BOX_JENKINS_1970,),
    ),
    "information_criteria": StatisticalTestInformation(
        name="AIC, AICc, BIC, and HQIC model-selection criteria",
        category="Relative selection criteria — not hypothesis tests",
        purpose="Rank eligible candidate models by penalized likelihood while accounting for different parameter counts.",
        null_hypothesis=None,
        alternative_hypothesis=None,
        statistic="AIC=-2 log L+2k; AICc adds a finite-sample penalty; BIC=-2 log L+k log n; HQIC uses a penalty proportional to 2k log log n.",
        reference_distribution="None for the values themselves. Only differences among models fitted to the same response/sample and likelihood convention are meaningful.",
        decision_rule="Minimize the user-selected criterion after mandatory diagnostic gates. Chrono Stream does not attach an alpha, p-value, or 'significance' label to a criterion difference.",
        chrono_stream="The app calculates AIC, statsmodels BIC/HQIC, and AICc from the fitted likelihood, parameter count, and n; candidates with undefined AICc receive no finite score. It ranks only eligible candidates unless the user explicitly permits a closest-model override.",
        interpretation="A lower value represents a preferred estimated trade-off under that criterion, not proof that the model is true. AIC/AICc emphasize expected information loss, while BIC and HQIC impose stronger asymptotic complexity penalties.",
        assumptions_and_caveats="Candidates must use comparable data and likelihoods. Searching many models adds selection uncertainty not shown by the winning value. Criteria cannot diagnose residual autocorrelation, coefficient instability, or forecast failure after a break.",
        literature_review="Akaike (1974) connected maximum likelihood with expected information loss and produced AIC. Schwarz (1978) derived a Bayesian large-sample criterion, and Hannan and Quinn (1979) developed an intermediate consistent penalty for autoregressive order. Hurvich and Tsai (1989) derived AICc for small-sample regression and time-series selection. These are rival decision objectives, not significance tests; Chrono Stream accordingly asks the user to choose one and reports all four.",
        references=(AKAIKE_1974, HURVICH_TSAI_1989, SCHWARZ_1978, HANNAN_QUINN_1979),
    ),
    "rolling_origin_cv": StatisticalTestInformation(
        name="Expanding-window rolling-origin validation",
        category="Predictive evaluation — not a hypothesis test",
        purpose="Rank candidate orders by genuinely later forecast errors inside the training partition.",
        null_hypothesis=None,
        alternative_hypothesis=None,
        statistic="CV RMSE=sqrt(mean(error^2)) or CV MAE=mean(|error|) pooled over configured origins and horizons.",
        reference_distribution="None in this implementation. The app computes an error score and does not run an equal-predictive-accuracy test.",
        decision_rule="Minimize CV RMSE or CV MAE after diagnostic eligibility. Every fold refits both the variance transformer and SARIMA model using observations available at that origin.",
        chrono_stream="Expanding windows remain inside the outer pre-holdout training data. The user selects folds and horizon; failed or nonfinite folds make the candidate's CV score unavailable rather than being silently omitted.",
        interpretation="A lower score indicates better historical pseudo-out-of-sample performance under the chosen origins, horizon, and loss. It is an estimate with sampling variability, not proof of future superiority.",
        assumptions_and_caveats="Results can be unstable with few folds, regime changes, or an unrepresentative horizon. Candidate selection on the same folds creates selection optimism, so the untouched outer holdout remains valuable. RMSE emphasizes large errors more than MAE.",
        literature_review="Tashman (2000) analyzed fixed and rolling origins, coefficient updating, and multiple test periods, arguing that rolling-origin evaluation improves the reliability of forecast comparisons. Chrono Stream follows that design with expanding windows and complete refitting. It calls the output a criterion, not a statistical test, because no null distribution or paired loss-differential inference is computed.",
        references=(TASHMAN_2000,),
    ),
    "qq_plot": StatisticalTestInformation(
        name="Normal Q–Q residual plot",
        category="Graphical diagnostic — not a hypothesis test",
        purpose="Show where the ordered standardized residual distribution differs from theoretical normal quantiles.",
        null_hypothesis=None,
        alternative_hypothesis=None,
        statistic="No scalar statistic. Points pair ordered residuals with Phi^-1((i-0.5)/n).",
        reference_distribution="No decision distribution or app confidence envelope is supplied.",
        decision_rule="Inspect systematic curvature, slope changes, asymmetry, and isolated tail points; do not declare reject/fail-to-reject from this plot alone.",
        chrono_stream="Chrono Stream calculates theoretical standard-normal plotting positions for every usable standardized residual and renders the pairs with a reference relationship in the residual diagnostics.",
        interpretation="An approximately straight pattern supports a normal-shape approximation; tail bends or asymmetric departures explain why an omnibus normality test may reject.",
        assumptions_and_caveats="Visual judgment is subjective and the app does not add simultaneous confidence bands. Autocorrelation can make an apparently normal marginal plot compatible with a dynamically inadequate model.",
        literature_review="Empirical-distribution-function tests such as Anderson–Darling formalize scalar discrepancies between an empirical and target CDF (Anderson & Darling, 1952; Stephens, 1974). A Q–Q plot preserves the location of those discrepancies and is therefore a necessary explanatory companion, but it is not itself a calibrated hypothesis test in this app.",
        references=(ANDERSON_DARLING_1952, STEPHENS_1974),
    ),
}


ARIMA_TEST_KEYS = (
    "adf",
    "kpss",
    "pp",
    "acf_lag",
    "pacf_lag",
    "coefficient_wald",
    "residual_mean_t",
    "jarque_bera",
    "shapiro_wilk",
    "anderson_darling",
    "lilliefors",
    "ljung_box",
    "box_pierce",
    "arch_lm",
)
SARIMA_TEST_KEYS = (
    "ocsb",
    "canova_hansen",
    *ARIMA_TEST_KEYS,
)
ARIMA_DECISION_AID_KEYS = (
    "adf_kpss_consensus",
    "roots",
    "optimizer_convergence",
    "information_criteria",
    "rolling_origin_cv",
    "qq_plot",
)
SARIMA_DECISION_AID_KEYS = (
    "adf_kpss_consensus",
    "seasonal_acf_rule",
    "roots",
    "optimizer_convergence",
    "information_criteria",
    "rolling_origin_cv",
    "qq_plot",
)


def test_keys_for_model(model_id: str) -> tuple[str, ...]:
    """Return the tests and decision aids declared by the method registry."""
    from .registry import METHOD_REGISTRY

    spec = METHOD_REGISTRY.get(model_id)
    return spec.statistical_test_keys if spec is not None else ()


def copy_ready_test_note(key: str) -> str:
    """Create a complete copy-ready note for one test or decision aid."""
    item = STATISTICAL_TESTS[key]
    sections = [
        item.name,
        f"Classification: {item.category}",
        f"Purpose\n{item.purpose}",
        "Null hypothesis (H0)\n"
        + (
            item.null_hypothesis
            or "Not applicable — this item is not a hypothesis test."
        ),
        "Alternative hypothesis (H1)\n"
        + (
            item.alternative_hypothesis
            or "Not applicable — this item is not a hypothesis test."
        ),
        f"Statistic\n{item.statistic}",
        f"Reference distribution\n{item.reference_distribution}",
        f"Decision rule\n{item.decision_rule}",
        f"Interpretation\n{item.interpretation}",
        f"Assumptions and caveats\n{item.assumptions_and_caveats}",
        f"Chrono Stream implementation\n{item.chrono_stream}",
        f"Literature review\n{item.literature_review}",
        "APA 7 references\n"
        + "\n\n".join(reference.apa for reference in item.references),
    ]
    return "\n\n".join(sections)


def copy_ready_test_handbook(
    keys: Iterable[str], title: str = "Chrono Stream statistical decision handbook"
) -> str:
    """Create one downloadable handbook containing the requested catalog entries."""
    selected = tuple(keys)
    preface = (
        f"{title}\n\n"
        "Decision language\n"
        "p < alpha: reject H0. p >= alpha: fail to reject H0. "
        "Items without H0/H1 are decision aids, not hypothesis tests."
    )
    notes = [copy_ready_test_note(key) for key in selected]
    return f"{preface}\n\n" + ("\n\n" + "=" * 88 + "\n\n").join(notes)
