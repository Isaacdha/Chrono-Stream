"""Scholarly, implementation-specific notes for every forecasting method.

The prose in this module is intentionally kept separate from the Streamlit UI so it can
be reviewed, tested, copied, and reused without importing Streamlit. References favor
the original or method-defining publication. When a model combines several ideas, the
``contribution`` field states exactly which part the source establishes.
"""

from __future__ import annotations

from dataclasses import dataclass

from .literature_reviews import METHOD_LITERATURE_REVIEWS


@dataclass(frozen=True)
class MethodReference:
    """One primary or method-defining source, formatted in APA 7 style."""

    apa: str
    url: str
    contribution: str


@dataclass(frozen=True)
class MethodInformation:
    """Research and implementation note shown from a forecasting-method page."""

    origin: str
    how_it_works: str
    chrono_stream: str
    when_to_use: str
    limitations: str
    citation_ready: str
    references: tuple[MethodReference, ...]


def _reference(apa: str, url: str, contribution: str) -> MethodReference:
    return MethodReference(apa=apa, url=url, contribution=contribution)


# Smoothing foundations
POYNTING_1884 = _reference(
    "Poynting, J. H. (1884). A comparison of the fluctuations in the price "
    "of wheat and in the cotton and silk imports into Great Britain. "
    "Journal of the Statistical Society of London, 47(1), 34–64. "
    "https://doi.org/10.2307/2979211",
    "https://doi.org/10.2307/2979211",
    "An early verified published use of successive averages in time-series analysis.",
)
HOOKER_1901 = _reference(
    "Hooker, R. H. (1901). Correlation of the marriage-rate with trade. "
    "Journal of the Royal Statistical Society, 64(3), 485–492. "
    "https://doi.org/10.1111/j.2397-2335.1901.tb03810.x",
    "https://doi.org/10.1111/j.2397-2335.1901.tb03810.x",
    "Explicitly uses a nine-year moving average to separate trend from fluctuations.",
)
MACAULAY_1931 = _reference(
    "Macaulay, F. R. (1931). The smoothing of time series. National Bureau "
    "of Economic Research. https://www.nber.org/books-and-chapters/smoothing-time-series",
    "https://www.nber.org/books-and-chapters/smoothing-time-series",
    "Systematically develops simple and weighted moving-average graduation methods.",
)
SPENCER_1904 = _reference(
    "Spencer, J. (1904). On the graduation of the rates of sickness and "
    "mortality presented by the experience of the Manchester Unity of "
    "Oddfellows during the period 1893–97. Journal of the Institute of "
    "Actuaries, 38(4), 334–343. https://doi.org/10.1017/S0020268100008076",
    "https://doi.org/10.1017/S0020268100008076",
    "Introduces the weighted graduation formula later known as Spencer's moving average.",
)
HENDERSON_1916 = _reference(
    "Henderson, R. (1916). Note on graduation by adjusted average. "
    "Transactions of the Actuarial Society of America, 17, 43–48.",
    "https://books.google.com/books?id=3oXvz33-EM4C",
    "Develops adjusted finite moving-average weights for smoothing a series.",
)
BROWN_1956 = _reference(
    "Brown, R. G. (1956). Exponential smoothing for predicting demand. "
    "Arthur D. Little.",
    "https://books.google.com/books?id=Eo_rMgEACAAJ",
    "The earliest widely documented forecasting treatment of simple exponential smoothing.",
)
MUTH_1960 = _reference(
    "Muth, J. F. (1960). Optimal properties of exponentially weighted "
    "forecasts. Journal of the American Statistical Association, 55(290), "
    "299–306. https://doi.org/10.1080/01621459.1960.10482064",
    "https://doi.org/10.1080/01621459.1960.10482064",
    "Derives conditions under which an exponentially weighted forecast is optimal.",
)
HOLT_1957 = _reference(
    "Holt, C. C. (2004). Forecasting seasonals and trends by exponentially "
    "weighted moving averages. International Journal of Forecasting, 20(1), "
    "5–10. https://doi.org/10.1016/j.ijforecast.2003.09.015 "
    "(Original work published 1957)",
    "https://doi.org/10.1016/j.ijforecast.2003.09.015",
    "The complete reprint of Holt's 1957 report defining level, trend, and seasonal extensions.",
)
WINTERS_1960 = _reference(
    "Winters, P. R. (1960). Forecasting sales by exponentially weighted "
    "moving averages. Management Science, 6(3), 324–342. "
    "https://doi.org/10.1287/mnsc.6.3.324",
    "https://doi.org/10.1287/mnsc.6.3.324",
    "Presents the seasonal exponential-forecasting system now called Holt–Winters.",
)
GARDNER_MCKENZIE_1985 = _reference(
    "Gardner, E. S., Jr., & McKenzie, E. (1985). Forecasting trends in time "
    "series. Management Science, 31(10), 1237–1246. "
    "https://doi.org/10.1287/mnsc.31.10.1237",
    "https://doi.org/10.1287/mnsc.31.10.1237",
    "Defines the damped-trend exponential-smoothing extension offered by the app.",
)


# ARIMA/SARIMA foundations and the exact tests exposed by Chrono Stream
YULE_1927 = _reference(
    "Yule, G. U. (1927). On a method of investigating periodicities in "
    "disturbed series, with special reference to Wolfer's sunspot numbers. "
    "Philosophical Transactions of the Royal Society A, 226(636–646), "
    "267–298. https://doi.org/10.1098/rsta.1927.0007",
    "https://doi.org/10.1098/rsta.1927.0007",
    "Introduces the autoregressive construction used as the AR part of ARIMA.",
)
SLUTZKY_1937 = _reference(
    "Slutzky, E. (1937). The summation of random causes as the source of "
    "cyclic processes. Econometrica, 5(2), 105–146. "
    "https://doi.org/10.2307/1907241 (Original work published 1927)",
    "https://doi.org/10.2307/1907241",
    "Shows how finite moving sums of random shocks generate serial structure, a foundation of MA models.",
)
BOX_JENKINS_1970 = _reference(
    "Box, G. E. P., & Jenkins, G. M. (1970). Time series analysis: "
    "Forecasting and control. Holden-Day.",
    "https://ntrl.ntis.gov/NTRL/dashboard/searchResults/titleDetail/AD720286.xhtml",
    "Unifies AR, differencing, MA, seasonal extensions, identification, estimation, checking, and forecasting.",
)
BOX_COX_1964 = _reference(
    "Box, G. E. P., & Cox, D. R. (1964). An analysis of transformations. "
    "Journal of the Royal Statistical Society: Series B (Methodological), "
    "26(2), 211–243. https://doi.org/10.1111/j.2517-6161.1964.tb00553.x",
    "https://doi.org/10.1111/j.2517-6161.1964.tb00553.x",
    "Defines the Box–Cox power family used for optional variance stabilization.",
)
YEO_JOHNSON_2000 = _reference(
    "Yeo, I.-K., & Johnson, R. A. (2000). A new family of power "
    "transformations to improve normality or symmetry. Biometrika, 87(4), "
    "954–959. https://doi.org/10.1093/biomet/87.4.954",
    "https://doi.org/10.1093/biomet/87.4.954",
    "Defines the Yeo–Johnson family, which permits zero and negative observations.",
)
DICKEY_FULLER_1979 = _reference(
    "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators "
    "for autoregressive time series with a unit root. Journal of the "
    "American Statistical Association, 74(366a), 427–431. "
    "https://doi.org/10.1080/01621459.1979.10482531",
    "https://doi.org/10.1080/01621459.1979.10482531",
    "Establishes the unit-root test underlying the app's ADF differencing decision.",
)
PHILLIPS_PERRON_1988 = _reference(
    "Phillips, P. C. B., & Perron, P. (1988). Testing for a unit root in "
    "time series regression. Biometrika, 75(2), 335–346. "
    "https://doi.org/10.1093/biomet/75.2.335",
    "https://doi.org/10.1093/biomet/75.2.335",
    "Defines the Phillips–Perron unit-root test available for differencing decisions.",
)
KPSS_1992 = _reference(
    "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). "
    "Testing the null hypothesis of stationarity against the alternative of "
    "a unit root. Journal of Econometrics, 54(1–3), 159–178. "
    "https://doi.org/10.1016/0304-4076(92)90104-Y",
    "https://doi.org/10.1016/0304-4076(92)90104-Y",
    "Defines the complementary stationarity-null test used alone or with ADF.",
)
OCSB_1988 = _reference(
    "Osborn, D. R., Chui, A. P. L., Smith, J. P., & Birchenhall, C. R. "
    "(1988). Seasonality and the order of integration for consumption. "
    "Oxford Bulletin of Economics and Statistics, 50(4), 361–377. "
    "https://doi.org/10.1111/j.1468-0084.1988.mp50004002.x",
    "https://doi.org/10.1111/j.1468-0084.1988.mp50004002.x",
    "Develops the seasonal integration regression behind the OCSB test used to choose D.",
)
CANOVA_HANSEN_1995 = _reference(
    "Canova, F., & Hansen, B. E. (1995). Are seasonal patterns constant over "
    "time? A test for seasonal stability. Journal of Business & Economic "
    "Statistics, 13(3), 237–252. https://doi.org/10.2307/1392184",
    "https://doi.org/10.2307/1392184",
    "Defines the seasonal-stability test offered as an alternative seasonal differencing rule.",
)
HEGY_1990 = _reference(
    "Hylleberg, S., Engle, R. F., Granger, C. W. J., & Yoo, B. S. (1990). "
    "Seasonal integration and cointegration. Journal of Econometrics, "
    "44(1–2), 215–238. https://doi.org/10.1016/0304-4076(90)90080-D",
    "https://doi.org/10.1016/0304-4076(90)90080-D",
    "Formalizes unit roots at seasonal frequencies and why seasonal differencing is distinct from regular differencing.",
)
AKAIKE_1974 = _reference(
    "Akaike, H. (1974). A new look at the statistical model identification. "
    "IEEE Transactions on Automatic Control, 19(6), 716–723. "
    "https://doi.org/10.1109/TAC.1974.1100705",
    "https://doi.org/10.1109/TAC.1974.1100705",
    "Defines AIC, one of the model-ranking criteria.",
)
SCHWARZ_1978 = _reference(
    "Schwarz, G. (1978). Estimating the dimension of a model. The Annals of "
    "Statistics, 6(2), 461–464. https://doi.org/10.1214/aos/1176344136",
    "https://doi.org/10.1214/aos/1176344136",
    "Defines the Schwarz criterion, now commonly called BIC.",
)
HANNAN_QUINN_1979 = _reference(
    "Hannan, E. J., & Quinn, B. G. (1979). The determination of the order of "
    "an autoregression. Journal of the Royal Statistical Society: Series B "
    "(Methodological), 41(2), 190–195. "
    "https://doi.org/10.1111/j.2517-6161.1979.tb01072.x",
    "https://doi.org/10.1111/j.2517-6161.1979.tb01072.x",
    "Defines the Hannan–Quinn order criterion exposed as HQIC.",
)
HURVICH_TSAI_1989 = _reference(
    "Hurvich, C. M., & Tsai, C.-L. (1989). Regression and time series model "
    "selection in small samples. Biometrika, 76(2), 297–307. "
    "https://doi.org/10.1093/biomet/76.2.297",
    "https://doi.org/10.1093/biomet/76.2.297",
    "Develops the small-sample correction used by AICc.",
)
HYNDMAN_KHANDAKAR_2008 = _reference(
    "Hyndman, R. J., & Khandakar, Y. (2008). Automatic time series "
    "forecasting: The forecast package for R. Journal of Statistical "
    "Software, 27(3), 1–22. https://doi.org/10.18637/jss.v027.i03",
    "https://doi.org/10.18637/jss.v027.i03",
    "Defines the stepwise automatic ARIMA-search strategy adapted by the app.",
)
BOX_PIERCE_1970 = _reference(
    "Box, G. E. P., & Pierce, D. A. (1970). Distribution of residual "
    "autocorrelations in autoregressive-integrated moving average time series "
    "models. Journal of the American Statistical Association, 65(332), "
    "1509–1526. https://doi.org/10.1080/01621459.1970.10481180",
    "https://doi.org/10.1080/01621459.1970.10481180",
    "Defines the residual portmanteau statistic offered as the Box–Pierce test.",
)
LJUNG_BOX_1978 = _reference(
    "Ljung, G. M., & Box, G. E. P. (1978). On a measure of lack of fit in "
    "time series models. Biometrika, 65(2), 297–303. "
    "https://doi.org/10.1093/biomet/65.2.297",
    "https://doi.org/10.1093/biomet/65.2.297",
    "Defines the finite-sample-adjusted residual white-noise test used by default.",
)
JARQUE_BERA_1980 = _reference(
    "Jarque, C. M., & Bera, A. K. (1980). Efficient tests for normality, "
    "homoscedasticity and serial independence of regression residuals. "
    "Economics Letters, 6(3), 255–259. "
    "https://doi.org/10.1016/0165-1765(80)90024-5",
    "https://doi.org/10.1016/0165-1765(80)90024-5",
    "Defines the residual normality test used by the strict workflow's default.",
)
SHAPIRO_WILK_1965 = _reference(
    "Shapiro, S. S., & Wilk, M. B. (1965). An analysis of variance test for "
    "normality (complete samples). Biometrika, 52(3–4), 591–611. "
    "https://doi.org/10.1093/biomet/52.3-4.591",
    "https://doi.org/10.1093/biomet/52.3-4.591",
    "Defines one of the selectable residual normality tests.",
)
ANDERSON_DARLING_1952 = _reference(
    "Anderson, T. W., & Darling, D. A. (1952). Asymptotic theory of certain "
    "goodness of fit criteria based on stochastic processes. The Annals of "
    "Mathematical Statistics, 23(2), 193–212. "
    "https://doi.org/10.1214/aoms/1177729437",
    "https://doi.org/10.1214/aoms/1177729437",
    "Defines the tail-sensitive goodness-of-fit statistic offered for normality checking.",
)
LILLIEFORS_1967 = _reference(
    "Lilliefors, H. W. (1967). On the Kolmogorov–Smirnov test for normality "
    "with mean and variance unknown. Journal of the American Statistical "
    "Association, 62(318), 399–402. "
    "https://doi.org/10.1080/01621459.1967.10482916",
    "https://doi.org/10.1080/01621459.1967.10482916",
    "Defines the estimated-parameter normality test offered by the app.",
)
ENGLE_1982 = _reference(
    "Engle, R. F. (1982). Autoregressive conditional heteroscedasticity with "
    "estimates of the variance of United Kingdom inflation. Econometrica, "
    "50(4), 987–1007. https://doi.org/10.2307/1912773",
    "https://doi.org/10.2307/1912773",
    "Defines ARCH and its LM diagnostic, which can be made a mandatory residual-variance gate.",
)
STUDENT_1908 = _reference(
    "Student. (1908). The probable error of a mean. Biometrika, 6(1), 1–25. "
    "https://doi.org/10.1093/biomet/6.1.1",
    "https://doi.org/10.1093/biomet/6.1.1",
    "Derives the small-sample mean statistic used to test whether residual mean differs from zero.",
)


# Decomposition, machine learning, and regression foundations
SHISKIN_1967 = _reference(
    "Shiskin, J., Young, A. H., & Musgrave, J. C. (1967). The X-11 variant "
    "of the Census Method II seasonal adjustment program (Technical Paper "
    "No. 15). U.S. Bureau of the Census.",
    "https://www.census.gov/content/dam/Census/library/working-papers/1967/adrm/shiskinyoungmusgrave1967.pdf",
    "Defines official Census X-11; it is included to delimit what this app does not claim to reproduce.",
)
CLEVELAND_TIAO_1976 = _reference(
    "Cleveland, W. P., & Tiao, G. C. (1976). Decomposition of seasonal time "
    "series: A model for the Census X-11 program. Journal of the American "
    "Statistical Association, 71(355), 581–587. "
    "https://doi.org/10.1080/01621459.1976.10481532",
    "https://doi.org/10.1080/01621459.1976.10481532",
    "Gives a stochastic-component interpretation of X-11's linear filters.",
)
CLEVELAND_STL_1990 = _reference(
    "Cleveland, R. B., Cleveland, W. S., McRae, J. E., & Terpenning, I. "
    "(1990). STL: A seasonal-trend decomposition procedure based on loess. "
    "Journal of Official Statistics, 6(1), 3–73.",
    "https://www.scb.se/contentassets/ca21efb41fee47d293bbee5bf7be7fb3/stl-a-seasonal-trend-decomposition-procedure-based-on-loess.pdf",
    "Defines STL, the decomposition algorithm actually executed by Chrono Stream.",
)
TAYLOR_LETHAM_2018 = _reference(
    "Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American "
    "Statistician, 72(1), 37–45. "
    "https://doi.org/10.1080/00031305.2017.1380080",
    "https://doi.org/10.1080/00031305.2017.1380080",
    "Introduces Prophet's modular trend, seasonality, holiday, and uncertainty model.",
)
HARVEY_SHEPHARD_1993 = _reference(
    "Harvey, A. C., & Shephard, N. (1993). Structural time series models. In "
    "G. S. Maddala, C. R. Rao, & H. D. Vinod (Eds.), Handbook of statistics "
    "(Vol. 11, pp. 261–302). North-Holland.",
    "https://shephard.scholars.harvard.edu/publications/structural-time-series-models",
    "Develops interpretable trend/seasonal component models and trigonometric seasonality used as a Prophet foundation.",
)
HOCHREITER_SCHMIDHUBER_1997 = _reference(
    "Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. "
    "Neural Computation, 9(8), 1735–1780. "
    "https://doi.org/10.1162/neco.1997.9.8.1735",
    "https://doi.org/10.1162/neco.1997.9.8.1735",
    "Introduces the LSTM architecture and gated constant-error flow.",
)
GERS_FORGET_2000 = _reference(
    "Gers, F. A., Schmidhuber, J., & Cummins, F. (2000). Learning to forget: "
    "Continual prediction with LSTM. Neural Computation, 12(10), 2451–2471. "
    "https://doi.org/10.1162/089976600300015015",
    "https://doi.org/10.1162/089976600300015015",
    "Introduces the forget gate present in the modern LSTM layer used by the app.",
)
GERS_TIME_SERIES_2001 = _reference(
    "Gers, F. A., Eck, D., & Schmidhuber, J. (2001). Applying LSTM to time "
    "series predictable through time-window approaches. In G. Dorffner, H. "
    "Bischof, & K. Hornik (Eds.), Artificial neural networks—ICANN 2001 "
    "(pp. 669–676). Springer. https://doi.org/10.1007/3-540-44668-0_93",
    "https://doi.org/10.1007/3-540-44668-0_93",
    "Directly evaluates LSTM on forecasting tasks formed from time windows.",
)
FUKUSHIMA_1980 = _reference(
    "Fukushima, K. (1980). Neocognitron: A self-organizing neural network "
    "model for a mechanism of pattern recognition unaffected by shift in "
    "position. Biological Cybernetics, 36(4), 193–202. "
    "https://doi.org/10.1007/BF00344251",
    "https://doi.org/10.1007/BF00344251",
    "Introduces the hierarchical, local-receptive-field neocognitron that is an architectural precursor to CNNs.",
)
LECUN_1989 = _reference(
    "LeCun, Y., Boser, B., Denker, J. S., Henderson, D., Howard, R. E., "
    "Hubbard, W., & Jackel, L. D. (1989). Backpropagation applied to "
    "handwritten zip code recognition. Neural Computation, 1(4), 541–551. "
    "https://doi.org/10.1162/neco.1989.1.4.541",
    "https://doi.org/10.1162/neco.1989.1.4.541",
    "Demonstrates a trainable shared-weight convolutional architecture learned end to end by backpropagation.",
)
LECUN_1998 = _reference(
    "LeCun, Y., Bottou, L., Bengio, Y., & Haffner, P. (1998). Gradient-based "
    "learning applied to document recognition. Proceedings of the IEEE, "
    "86(11), 2278–2324. https://doi.org/10.1109/5.726791",
    "https://doi.org/10.1109/5.726791",
    "Provides the defining convolution, shared-weight, pooling, and gradient-training formulation for CNNs.",
)
BOROVYKH_2017 = _reference(
    "Borovykh, A., Bohte, S., & Oosterlee, C. W. (2017). Conditional time "
    "series forecasting with convolutional neural networks. arXiv. "
    "https://doi.org/10.48550/arXiv.1703.04691",
    "https://doi.org/10.48550/arXiv.1703.04691",
    "Explicitly formulates convolutional networks for time-series forecasting; its WaveNet-style architecture is deeper than the app's baseline.",
)
FRIEDMAN_2001 = _reference(
    "Friedman, J. H. (2001). Greedy function approximation: A gradient "
    "boosting machine. The Annals of Statistics, 29(5), 1189–1232. "
    "https://doi.org/10.1214/aos/1013203451",
    "https://doi.org/10.1214/aos/1013203451",
    "Develops gradient boosting, including additive regression-tree algorithms.",
)
CHEN_GUESTRIN_2016 = _reference(
    "Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting "
    "system. In Proceedings of the 22nd ACM SIGKDD International Conference "
    "on Knowledge Discovery and Data Mining (pp. 785–794). ACM. "
    "https://doi.org/10.1145/2939672.2939785",
    "https://doi.org/10.1145/2939672.2939785",
    "Defines the regularized, scalable XGBoost system used by the app.",
)
BEN_TAIEB_2012 = _reference(
    "Ben Taieb, S., Bontempi, G., Atiya, A. F., & Sorjamaa, A. (2012). A "
    "review and comparison of strategies for multi-step ahead time series "
    "forecasting based on the NN5 forecasting competition. Expert Systems "
    "with Applications, 39(8), 7067–7083. "
    "https://doi.org/10.1016/j.eswa.2012.01.039",
    "https://doi.org/10.1016/j.eswa.2012.01.039",
    "Defines and tests recursive and alternative multi-step forecasting strategies.",
)
LEGENDRE_1805 = _reference(
    "Legendre, A.-M. (1805). Nouvelles méthodes pour la détermination des "
    "orbites des comètes. Firmin Didot. "
    "https://doi.org/10.3931/e-rara-35115",
    "https://doi.org/10.3931/e-rara-35115",
    "Contains the first published statement of the method of least squares.",
)
GAUSS_1809 = _reference(
    "Gauss, C. F. (1809). Theoria motus corporum coelestium in sectionibus "
    "conicis solem ambientium. Perthes & Besser. "
    "https://doi.org/10.3931/e-rara-522",
    "https://doi.org/10.3931/e-rara-522",
    "Publishes Gauss's probabilistic development and use of least squares.",
)
GERGONNE_1815 = _reference(
    "Gergonne, J. D. (1815–1816). Analise: Application de la méthode des "
    "moindres quarrés à l'interpolation des suites. Annales de Mathématiques "
    "Pures et Appliquées, 6, 242–252.",
    "https://eudml.org/doc/79602",
    "Directly applies least squares to polynomial interpolation of a sequence.",
)
YULE_1926 = _reference(
    "Yule, G. U. (1926). Why do we sometimes get nonsense-correlations "
    "between time-series? A study in sampling and the nature of time-series. "
    "Journal of the Royal Statistical Society, 89(1), 1–64. "
    "https://doi.org/10.2307/2341482",
    "https://doi.org/10.2307/2341482",
    "Demonstrates why apparently strong trend relationships can be spurious in autocorrelated time series.",
)
MALTHUS_1798 = _reference(
    "Malthus, T. R. (1798). An essay on the principle of population. J. "
    "Johnson.",
    "https://www.gutenberg.org/files/4239/4239-h/4239-h.htm",
    "An early explicit domain model of unchecked geometric growth, the discrete analogue of exponential growth.",
)
DUAN_1983 = _reference(
    "Duan, N. (1983). Smearing estimate: A nonparametric retransformation "
    "method. Journal of the American Statistical Association, 78(383), "
    "605–610. https://doi.org/10.1080/01621459.1983.10478017",
    "https://doi.org/10.1080/01621459.1983.10478017",
    "Defines the retransformation correction used by Chrono Stream after log-linear fitting.",
)
MILLER_1984 = _reference(
    "Miller, D. M. (1984). Reducing transformation bias in curve fitting. "
    "The American Statistician, 38(2), 124–126. "
    "https://doi.org/10.1080/00031305.1984.10483180",
    "https://doi.org/10.1080/00031305.1984.10483180",
    "Explains the prediction bias caused by naively reversing a fitted transformation.",
)
FECHNER_1860 = _reference(
    "Fechner, G. T. (1860). Elemente der Psychophysik. Breitkopf und Härtel. "
    "https://doi.org/10.3931/e-rara-10879",
    "https://doi.org/10.3931/e-rara-10879",
    "Develops an explicit logarithmic response law; it is a curve-shape origin, not a forecasting paper.",
)


# Baseline-forecast foundations and benchmark literature
PEARSON_1905 = _reference(
    "Pearson, K. (1905). The problem of the random walk. Nature, 72, 294. "
    "https://doi.org/10.1038/072294b0",
    "https://doi.org/10.1038/072294b0",
    "Introduces the random-walk terminology underlying last-value persistence models.",
)
HYNDMAN_ATHANASOPOULOS_2021 = _reference(
    "Hyndman, R. J., & Athanasopoulos, G. (2021). Forecasting: Principles "
    "and practice (3rd ed.). OTexts. https://otexts.com/fpp3/",
    "https://otexts.com/fpp3/",
    "Gives the modern forecasting equations for naive, seasonal-naive, and drift benchmarks.",
)
HYNDMAN_KOEHLER_2006 = _reference(
    "Hyndman, R. J., & Koehler, A. B. (2006). Another look at measures of "
    "forecast accuracy. International Journal of Forecasting, 22(4), 679–688. "
    "https://doi.org/10.1016/j.ijforecast.2006.03.001",
    "https://doi.org/10.1016/j.ijforecast.2006.03.001",
    "Establishes naive and seasonal-naive errors as meaningful scale denominators for forecast comparison.",
)
MAKRIDAKIS_HIBON_2000 = _reference(
    "Makridakis, S., & Hibon, M. (2000). The M3-Competition: Results, "
    "conclusions and implications. International Journal of Forecasting, "
    "16(4), 451–476. https://doi.org/10.1016/S0169-2070(00)00057-1",
    "https://doi.org/10.1016/S0169-2070(00)00057-1",
    "Demonstrates the continuing role of simple benchmark methods in large empirical forecast comparisons.",
)

# Lag-regression, tree, kernel, and automatic-classical foundations
TASHMAN_2000 = _reference(
    "Tashman, L. J. (2000). Out-of-sample tests of forecasting accuracy: An "
    "analysis and review. International Journal of Forecasting, 16(4), 437–450. "
    "https://doi.org/10.1016/S0169-2070(00)00065-0",
    "https://doi.org/10.1016/S0169-2070(00)00065-0",
    "Establishes expanding and rolling forecast-origin evaluation as a predictive model-selection design.",
)
HOERL_KENNARD_1970 = _reference(
    "Hoerl, A. E., & Kennard, R. W. (1970). Ridge regression: Biased "
    "estimation for nonorthogonal problems. Technometrics, 12(1), 55–67. "
    "https://doi.org/10.1080/00401706.1970.10488634",
    "https://doi.org/10.1080/00401706.1970.10488634",
    "Introduces ridge shrinkage for unstable least-squares estimates under correlated predictors.",
)
TIBSHIRANI_1996 = _reference(
    "Tibshirani, R. (1996). Regression shrinkage and selection via the lasso. "
    "Journal of the Royal Statistical Society: Series B (Methodological), "
    "58(1), 267–288. https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",
    "https://doi.org/10.1111/j.2517-6161.1996.tb02080.x",
    "Defines the L1-constrained regression estimator that can set coefficients exactly to zero.",
)
ZOU_HASTIE_2005 = _reference(
    "Zou, H., & Hastie, T. (2005). Regularization and variable selection via "
    "the elastic net. Journal of the Royal Statistical Society: Series B "
    "(Statistical Methodology), 67(2), 301–320. "
    "https://doi.org/10.1111/j.1467-9868.2005.00503.x",
    "https://doi.org/10.1111/j.1467-9868.2005.00503.x",
    "Defines Elastic Net's combined L1/L2 penalty and grouping behavior for correlated predictors.",
)
BREIMAN_ET_AL_1984 = _reference(
    "Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). "
    "Classification and regression trees. Wadsworth. "
    "https://doi.org/10.1201/9781315139470",
    "https://doi.org/10.1201/9781315139470",
    "Provides the defining CART methodology for recursive partitioning and regression-tree pruning.",
)
BREIMAN_2001 = _reference(
    "Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5–32. "
    "https://doi.org/10.1023/A:1010933404324",
    "https://doi.org/10.1023/A:1010933404324",
    "Defines Random Forest ensembles based on randomized tree predictors and feature selection.",
)
GEURTS_ET_AL_2006 = _reference(
    "Geurts, P., Ernst, D., & Wehenkel, L. (2006). Extremely randomized "
    "trees. Machine Learning, 63(1), 3–42. "
    "https://doi.org/10.1007/s10994-006-6226-1",
    "https://doi.org/10.1007/s10994-006-6226-1",
    "Defines Extra Trees and its additional randomization of both attributes and split thresholds.",
)
DRUCKER_ET_AL_1997 = _reference(
    "Drucker, H., Burges, C. J. C., Kaufman, L., Smola, A., & Vapnik, V. "
    "(1997). Support vector regression machines. In M. C. Mozer, M. Jordan, "
    "& T. Petsche (Eds.), Advances in neural information processing systems "
    "9 (pp. 155–161). MIT Press. "
    "https://papers.nips.cc/paper/1238-support-vector-regression-machines",
    "https://papers.nips.cc/paper/1238-support-vector-regression-machines",
    "Provides an early empirical application and comparison of support-vector regression.",
)
SMOLA_SCHOLKOPF_2004 = _reference(
    "Smola, A. J., & Schölkopf, B. (2004). A tutorial on support vector "
    "regression. Statistics and Computing, 14(3), 199–222. "
    "https://doi.org/10.1023/B:STCO.0000035301.49549.88",
    "https://doi.org/10.1023/B:STCO.0000035301.49549.88",
    "Provides a later tutorial treatment of the epsilon-insensitive objective, kernels, regularization, and SVR algorithms.",
)
VAPNIK_1995 = _reference(
    "Vapnik, V. N. (1995). The nature of statistical learning theory. "
    "Springer. https://doi.org/10.1007/978-1-4757-2440-0",
    "https://doi.org/10.1007/978-1-4757-2440-0",
    "Establishes the statistical-learning and support-vector function-estimation framework underlying epsilon-SVR.",
)
VAPNIK_GOLOWICH_SMOLA_1997 = _reference(
    "Vapnik, V., Golowich, S. E., & Smola, A. J. (1997). Support vector "
    "method for function approximation, regression estimation and signal "
    "processing. In M. Mozer, M. Jordan, & T. Petsche (Eds.), Advances in "
    "neural information processing systems 9 (pp. 281–287). MIT Press. "
    "https://proceedings.neurips.cc/paper/1996/hash/4f284803bd0966cc24fa8683a34afc6e-Abstract.html",
    "https://proceedings.neurips.cc/paper/1996/hash/4f284803bd0966cc24fa8683a34afc6e-Abstract.html",
    "Presents the support-vector method for function approximation and real-valued regression estimation.",
)
STONE_1977 = _reference(
    "Stone, C. J. (1977). Consistent nonparametric regression. The Annals "
    "of Statistics, 5(4), 595–620. https://doi.org/10.1214/aos/1176343886",
    "https://doi.org/10.1214/aos/1176343886",
    "Establishes consistency conditions that include nearest-neighbor probability-weight regression rules.",
)
ALTMAN_1992 = _reference(
    "Altman, N. S. (1992). An introduction to kernel and nearest-neighbor "
    "nonparametric regression. The American Statistician, 46(3), 175–185. "
    "https://doi.org/10.1080/00031305.1992.10475879",
    "https://doi.org/10.1080/00031305.1992.10475879",
    "Explains nearest-neighbor regression as a local conditional-location estimator and its tuning tradeoffs.",
)
ASSIMAKOPOULOS_NIKOLOPOULOS_2000 = _reference(
    "Assimakopoulos, V., & Nikolopoulos, K. (2000). The Theta model: A "
    "decomposition approach to forecasting. International Journal of "
    "Forecasting, 16(4), 521–530. "
    "https://doi.org/10.1016/S0169-2070(00)00066-2",
    "https://doi.org/10.1016/S0169-2070(00)00066-2",
    "Introduces Theta lines and the conventional theta-zero/theta-two forecast combination.",
)
HYNDMAN_BILLAH_2003 = _reference(
    "Hyndman, R. J., & Billah, B. (2003). Unmasking the Theta method. "
    "International Journal of Forecasting, 19(2), 287–290. "
    "https://doi.org/10.1016/S0169-2070(01)00143-1",
    "https://doi.org/10.1016/S0169-2070(01)00143-1",
    "Derives the simpler equivalence between the classical Theta forecast and SES with drift.",
)
FIORUCCI_ET_AL_2016 = _reference(
    "Fiorucci, J. A., Pellegrini, T. R., Louzada, F., Petropoulos, F., & "
    "Koehler, A. B. (2016). Models for optimising the theta method and their "
    "relationship to state space models. International Journal of Forecasting, "
    "32(4), 1151–1161. https://doi.org/10.1016/j.ijforecast.2016.02.005",
    "https://doi.org/10.1016/j.ijforecast.2016.02.005",
    "Generalizes Theta by selecting the short-term theta line rather than fixing the classical theta-two value.",
)
HYNDMAN_ET_AL_2002 = _reference(
    "Hyndman, R. J., Koehler, A. B., Snyder, R. D., & Grose, S. (2002). A "
    "state space framework for automatic forecasting using exponential "
    "smoothing methods. International Journal of Forecasting, 18(3), "
    "439–454. https://doi.org/10.1016/S0169-2070(01)00110-8",
    "https://doi.org/10.1016/S0169-2070(01)00110-8",
    "Defines the ETS taxonomy, likelihood criteria, automatic selection, and state-space intervals.",
)
HYNDMAN_ET_AL_2008 = _reference(
    "Hyndman, R. J., Koehler, A. B., Ord, J. K., & Snyder, R. D. (2008). "
    "Forecasting with exponential smoothing: The state space approach. "
    "Springer. https://doi.org/10.1007/978-3-540-71918-2",
    "https://doi.org/10.1007/978-3-540-71918-2",
    "Develops the innovation state-space ETS family, estimation, selection, and forecast distributions in detail.",
)
DE_LIVERA_ET_AL_2011 = _reference(
    "De Livera, A. M., Hyndman, R. J., & Snyder, R. D. (2011). Forecasting "
    "time series with complex seasonal patterns using exponential smoothing. "
    "Journal of the American Statistical Association, 106(496), 1513–1527. "
    "https://doi.org/10.1198/jasa.2011.tm09771",
    "https://doi.org/10.1198/jasa.2011.tm09771",
    "Defines the TBATS framework for multiple, high-frequency, and non-integer seasonality.",
)
CROSTON_1972 = _reference(
    "Croston, J. D. (1972). Forecasting and stock control for intermittent "
    "demands. Operational Research Quarterly, 23(3), 289–303. "
    "https://doi.org/10.1057/jors.1972.50",
    "https://doi.org/10.1057/jors.1972.50",
    "Introduces separate exponential smoothing of nonzero demand sizes and inter-demand intervals.",
)
SYNTETOS_BOYLAN_2005 = _reference(
    "Syntetos, A. A., & Boylan, J. E. (2005). The accuracy of intermittent "
    "demand estimates. International Journal of Forecasting, 21(2), 303–314. "
    "https://doi.org/10.1016/j.ijforecast.2004.10.001",
    "https://doi.org/10.1016/j.ijforecast.2004.10.001",
    "Analyzes Croston's bias and establishes the correction used by the SBA variant.",
)
TEUNTER_ET_AL_2011 = _reference(
    "Teunter, R. H., Syntetos, A. A., & Babai, M. Z. (2011). Intermittent "
    "demand: Linking forecasting to inventory obsolescence. European Journal "
    "of Operational Research, 214(3), 606–615. "
    "https://doi.org/10.1016/j.ejor.2011.05.018",
    "https://doi.org/10.1016/j.ejor.2011.05.018",
    "Defines TSB's separate smoothing of demand occurrence probability and positive demand size.",
)
MSTL_2025 = _reference(
    "Bandara, K., Hyndman, R. J., & Bergmeir, C. (2025). MSTL: A "
    "seasonal-trend decomposition algorithm for time series with multiple "
    "seasonal patterns. International Journal of Operational Research, 52(1), "
    "79–98. https://doi.org/10.1504/IJOR.2025.143957",
    "https://doi.org/10.1504/IJOR.2025.143957",
    "Defines MSTL's iterative use of STL to estimate multiple seasonal components.",
)
ORESHKIN_ET_AL_2020 = _reference(
    "Oreshkin, B. N., Carpov, D., Chapados, N., & Bengio, Y. (2020). "
    "N-BEATS: Neural basis expansion analysis for interpretable time series "
    "forecasting. In International Conference on Learning Representations. "
    "https://openreview.net/forum?id=r1ecqn4YwB",
    "https://openreview.net/forum?id=r1ecqn4YwB",
    "Introduces N-BEATS backcast/forecast blocks, doubly residual stacking, and generic and interpretable variants.",
)
BAI_ET_AL_2018 = _reference(
    "Bai, S., Kolter, J. Z., & Koltun, V. (2018). An empirical evaluation of "
    "generic convolutional and recurrent networks for sequence modeling. "
    "arXiv. https://doi.org/10.48550/arXiv.1803.01271",
    "https://doi.org/10.48550/arXiv.1803.01271",
    "Defines and evaluates a generic residual TCN built from causal dilated convolutions.",
)
VAN_DEN_OORD_ET_AL_2016 = _reference(
    "van den Oord, A., Dieleman, S., Zen, H., Simonyan, K., Vinyals, O., "
    "Graves, A., Kalchbrenner, N., Senior, A., & Kavukcuoglu, K. (2016). "
    "WaveNet: A generative model for raw audio. arXiv. "
    "https://doi.org/10.48550/arXiv.1609.03499",
    "https://doi.org/10.48550/arXiv.1609.03499",
    "Establishes the causal dilated-convolution design lineage that informed later TCN architectures.",
)


ARIMA_CORE_REFERENCES = (
    YULE_1927,
    SLUTZKY_1937,
    BOX_JENKINS_1970,
    BOX_COX_1964,
    YEO_JOHNSON_2000,
    DICKEY_FULLER_1979,
    PHILLIPS_PERRON_1988,
    KPSS_1992,
    AKAIKE_1974,
    SCHWARZ_1978,
    HANNAN_QUINN_1979,
    HURVICH_TSAI_1989,
    HYNDMAN_KHANDAKAR_2008,
    BOX_PIERCE_1970,
    LJUNG_BOX_1978,
    JARQUE_BERA_1980,
    SHAPIRO_WILK_1965,
    ANDERSON_DARLING_1952,
    LILLIEFORS_1967,
    ENGLE_1982,
    STUDENT_1908,
)


METHOD_INFORMATION: dict[str, MethodInformation] = {
    "naive": MethodInformation(
        origin=(
            "The naive forecast is the forecasting form of a random walk: the latest "
            "observation is the best point forecast for every future horizon. Pearson "
            "(1905) introduced random-walk terminology, although no single publication "
            "should be credited as the inventor of last-observation persistence. Modern "
            "forecasting texts define it explicitly as an indispensable benchmark."
        ),
        how_it_works=(
            "At forecast origin T, the rule is y-hat(T+h|T) = y(T) for every positive "
            "horizon h. Historical one-step fitted values are therefore the immediately "
            "preceding observations. The point forecast has no estimated trend, seasonal "
            "state, or fitted coefficient and cannot change until a new value is observed."
        ),
        chrono_stream=(
            "Chrono Stream fits the rule independently on the pre-holdout partition and "
            "on the complete series, so the holdout cannot affect its benchmark forecast. "
            "It reports the last training observation, uses direct persistence for all "
            "horizons, and uses Gaussian random-walk intervals whose standard deviation "
            "accumulates in proportion to the square root of the horizon."
        ),
        when_to_use=(
            "Use it for level-like or random-walk series and as the minimum non-seasonal "
            "standard against which every more elaborate method should be compared."
        ),
        limitations=(
            "Persistence ignores systematic drift, seasonality, interventions, external "
            "drivers, and mean reversion. A low error can still be unhelpful when the "
            "decision requires anticipating turning points. Its intervals additionally assume "
            "independent, constant-variance Gaussian first differences."
        ),
        citation_ready=(
            "The naive forecast sets every future point forecast equal to the final "
            "observed value and corresponds to the conditional mean or median forecast "
            "of a random walk under suitable innovation assumptions (Hyndman & "
            "Athanasopoulos, 2021). Pearson (1905) supplies the historical random-walk "
            "term rather than this forecasting equation. Its main practical role is as a transparent "
            "benchmark that more complex methods should outperform out of sample."
        ),
        references=(
            PEARSON_1905,
            HYNDMAN_ATHANASOPOULOS_2021,
            HYNDMAN_KOEHLER_2006,
        ),
    ),
    "seasonal_naive": MethodInformation(
        origin=(
            "Seasonal naive forecasting extends persistence to a repeating cycle: each "
            "future phase inherits the most recent observation from that same phase. It "
            "is a standard benchmark rather than a technique attributable to one unique "
            "origin paper. Hyndman and Athanasopoulos (2021) give its modern equation, "
            "while forecast-accuracy research formalizes its benchmark role."
        ),
        how_it_works=(
            "For seasonal period m, the h-step forecast is the last available observation "
            "whose index has the same phase modulo m. Forecasts beyond one cycle repeat "
            "the same final observed seasonal cycle. Historical fitted values at time t "
            "use y(t-m), so no centered or future observation enters the predictor."
        ),
        chrono_stream=(
            "Chrono Stream requires an integer seasonal period of at least two and at "
            "least two complete cycles. The evaluation fit uses only pre-holdout values, "
            "the final fit repeats the last complete observed cycle, and the selected "
            "period plus direct seasonal-persistence strategy are recorded in model details."
        ),
        when_to_use=(
            "Use it as the essential comparator for regularly spaced series whose level "
            "may vary by season but whose phase pattern is reasonably stable from cycle to cycle."
        ),
        limitations=(
            "The method assumes an unchanged seasonal pattern and does not estimate trend, "
            "calendar anomalies, multiple periods, evolving seasonal amplitude, or external "
            "effects. An incorrect period gives systematically misaligned forecasts, and "
            "residual-based intervals are only approximate rather than a calibrated seasonal model."
        ),
        citation_ready=(
            "A seasonal naive forecast assigns each future observation the latest value "
            "recorded in the same seasonal phase (Hyndman & Athanasopoulos, 2021). It is "
            "therefore a causal seasonal-persistence benchmark. Naive and seasonal-naive "
            "errors also underpin scaled accuracy measures designed for comparison across "
            "series (Hyndman & Koehler, 2006)."
        ),
        references=(
            HYNDMAN_ATHANASOPOULOS_2021,
            HYNDMAN_KOEHLER_2006,
            MAKRIDAKIS_HIBON_2000,
        ),
    ),
    "drift": MethodInformation(
        origin=(
            "The drift method augments random-walk persistence with a constant mean change. "
            "It belongs to the random-walk-with-drift family rather than deterministic "
            "least-squares trend regression. Pearson (1905) supplies the random-walk lineage, "
            "and Hyndman and Athanasopoulos (2021) state the familiar forecasting rule as "
            "the average change between the first and latest observations."
        ),
        how_it_works=(
            "For n observations, the estimated drift is [y(n)-y(1)]/(n-1), which is also "
            "the arithmetic mean of all one-period changes because the differences telescope. "
            "The h-step forecast adds h times this drift to y(n). It is a direct multi-step "
            "rule: forecasts are not recursively re-estimated from earlier forecasts."
        ),
        chrono_stream=(
            "Chrono Stream estimates drift separately inside the outer training partition "
            "and after refitting on all observations. It records the first and last values, "
            "the per-step drift, and its direct extrapolation strategy. Historical fitted "
            "values are one-step random-walk-with-drift forecasts, and Gaussian intervals "
            "accumulate innovation and drift-estimation uncertainty with the horizon."
        ),
        when_to_use=(
            "Use it as a low-cost trend-sensitive baseline when a roughly constant average "
            "change is plausible and a naive flat forecast is too restrictive."
        ),
        limitations=(
            "Only the endpoints determine the estimated average change, so an unusual first "
            "or last observation can strongly alter every forecast. Drift ignores seasonality, "
            "curvature, structural breaks, mean reversion, and external causes; long-horizon "
            "linear extrapolation can become implausible even when short-horizon error is competitive."
        ),
        citation_ready=(
            "The drift method forecasts from a random walk with a constant mean increment: "
            "the latest observation is advanced by the horizon times the average historical "
            "change between the first and last observations (Hyndman & Athanasopoulos, "
            "2021). Pearson (1905) supplies random-walk lineage, not the drift forecast rule. "
            "It is a direct benchmark and should not be confused with "
            "a least-squares line fitted to all levels."
        ),
        references=(
            PEARSON_1905,
            HYNDMAN_ATHANASOPOULOS_2021,
            MAKRIDAKIS_HIBON_2000,
        ),
    ),
    "moving_average": MethodInformation(
        origin=(
            "Moving averages were used before modern forecast theory. The earliest "
            "published statistical use verified for this project is Poynting (1884); "
            "Hooker (1901) then explicitly separated trend and fluctuation with a "
            "nine-year moving average, and Macaulay (1931) systematized the method. "
            "Those historical smoothers were usually centered. A forecasting moving "
            "average must instead be trailing so it never uses future observations."
        ),
        how_it_works=(
            "For window length w, the next forecast is the arithmetic mean of the "
            "latest w available observations. Equal weights make the method transparent "
            "and damp short-run noise. Multi-step forecasts are recursive: each newly "
            "predicted value becomes available to the next window."
        ),
        chrono_stream=(
            "Chrono Stream uses past-only one-step fitted values and recursive future "
            "forecasts. The user can set w or let the app choose it by minimum in-sample "
            "one-step RMSE within the model's training partition. Holdout observations "
            "are not used to select the evaluation window."
        ),
        when_to_use=(
            "Use it as an interpretable baseline for a locally level series whose recent "
            "history is more relevant than its distant history."
        ),
        limitations=(
            "It has no explicit trend or seasonal state. Recursive forecasts converge "
            "toward a constant, structural breaks are averaged rather than modeled, and "
            "the automatic window optimizes training fit rather than guaranteeing future accuracy."
        ),
        citation_ready=(
            "A moving average suppresses short-run variation by replacing the current "
            "level with the arithmetic mean of a fixed number of consecutive "
            "observations. Early published time-series use can be traced at least to "
            "Poynting (1884); Hooker (1901) explicitly used a moving average to separate "
            "trend from fluctuation, and Macaulay (1931) later systematized moving-average "
            "smoothing. For genuine forecasting, the average must be trailing rather "
            "than centered so that only information available at the forecast origin is used."
        ),
        references=(POYNTING_1884, HOOKER_1901, MACAULAY_1931),
    ),
    "weighted_moving_average": MethodInformation(
        origin=(
            "Weighted moving averages grew from actuarial graduation. Spencer (1904) "
            "published an explicit weighted smoothing formula, Henderson (1916) developed "
            "adjusted-average weights, and Macaulay (1931) compared these systems. Their "
            "classical filters were symmetric smoothers, not the exact trailing forecast "
            "rule implemented here."
        ),
        how_it_works=(
            "The forecast is a normalized weighted sum of the latest w observations. "
            "Larger weights on recent values make it adapt faster than an equal-weight "
            "average. Linear weighting uses weights 1 through w; exponential weighting "
            "uses powers of a user-selected decay factor within the finite window."
        ),
        chrono_stream=(
            "The newest observation always receives the largest weight. Fitted values "
            "are past-only, future values are generated recursively, and the window can "
            "be manual or selected by one-step training RMSE. This is a finite weighted "
            "moving average, not recursive simple exponential smoothing."
        ),
        when_to_use=(
            "Use it as a baseline when the recent level should respond faster than an "
            "unweighted average but an explicit trend or seasonal model is unnecessary."
        ),
        limitations=(
            "The weight shape is imposed rather than learned, forecast errors can "
            "accumulate recursively, and finite recent weighting does not by itself "
            "represent sustained trend or seasonality."
        ),
        citation_ready=(
            "A weighted moving average estimates the current level as a normalized linear "
            "combination of consecutive observations, allowing selected observations to "
            "have greater influence than others. Spencer (1904) and Henderson (1916) "
            "developed influential weighted graduation formulas, which Macaulay (1931) "
            "treated systematically. Chrono Stream adapts this finite-filter idea to "
            "past-only forecasting by assigning the largest weight to the newest value; "
            "it does not claim to reproduce the symmetric Spencer or Henderson filters."
        ),
        references=(SPENCER_1904, HENDERSON_1916, MACAULAY_1931),
    ),
    "single_exponential_smoothing": MethodInformation(
        origin=(
            "Brown (1956) gave the earliest widely documented demand-forecasting "
            "treatment. Holt's 1957 report developed the exponentially weighted framework "
            "systematically, and Muth (1960) established an optimality result for a "
            "random-walk-plus-noise process."
        ),
        how_it_works=(
            "The level update is l_t = alpha*y_t + (1-alpha)*l_(t-1), where 0 < alpha <= 1. "
            "This recursion is an average of all past observations with geometrically "
            "declining weights. Every future point forecast equals the final estimated level."
        ),
        chrono_stream=(
            "Chrono Stream fits statsmodels SimpleExpSmoothing. Automatic mode jointly "
            "estimates alpha and the initial level; manual-alpha mode uses statsmodels' "
            "heuristic initial level rather than optimizing that state."
        ),
        when_to_use=(
            "Use it for a series with a locally changing level but no sustained trend and "
            "no stable repeating seasonality."
        ),
        limitations=(
            "It produces a flat point forecast, reacts slowly when alpha is small, and "
            "cannot separately represent trend or seasonality. Muth's optimality result "
            "depends on a particular stochastic process and is not universal."
        ),
        citation_ready=(
            "Simple exponential smoothing recursively updates a latent level by combining "
            "the newest observation with the previous level estimate, which is equivalent "
            "to assigning geometrically declining weights to older observations. Brown "
            "(1956) gave an early widely documented demand-forecasting treatment, Holt (1957/2004) "
            "developed the forecasting equations systematically, and Muth (1960) showed "
            "that the exponentially weighted forecast is optimal for a random walk observed with noise."
        ),
        references=(BROWN_1956, HOLT_1957, MUTH_1960),
    ),
    "double_exponential_smoothing": MethodInformation(
        origin=(
            "Holt's 1957 Office of Naval Research report introduced separate exponentially "
            "smoothed level and trend recursions. Gardner and McKenzie (1985) later "
            "introduced the damped-trend extension used to prevent indefinite linear extrapolation."
        ),
        how_it_works=(
            "Holt's additive method updates a local level and a local slope, then forecasts "
            "h steps ahead as level plus h times slope. In the damped version, successive "
            "trend contributions are multiplied by powers of phi, so their long-horizon "
            "effect gradually decreases."
        ),
        chrono_stream=(
            "The app fits additive-trend statsmodels ExponentialSmoothing. Automatic mode "
            "jointly estimates smoothing parameters and initial states; manual alpha/beta "
            "uses statsmodels' heuristic initial states. Damping is optional, with phi "
            "optimized automatically or supplied in manual mode."
        ),
        when_to_use=(
            "Use it when a nonseasonal series has a changing level and an approximately "
            "linear local trend. Damping is often safer for longer horizons."
        ),
        limitations=(
            "It has no seasonal component, assumes a smooth local trend, and an undamped "
            "slope can create implausibly large long-range forecasts."
        ),
        citation_ready=(
            "Holt's trend method extends exponential smoothing by updating separate level "
            "and slope states and extrapolating their sum (Holt, 1957/2004). The damped "
            "extension reduces the contribution of the slope as the forecast horizon grows, "
            "which can improve long-range robustness when a recent trend is unlikely to "
            "continue indefinitely (Gardner & McKenzie, 1985)."
        ),
        references=(HOLT_1957, GARDNER_MCKENZIE_1985),
    ),
    "triple_exponential_smoothing": MethodInformation(
        origin=(
            "Holt's 1957 report developed exponentially weighted level, trend, and seasonal "
            "components. Winters (1960) published the operational seasonal sales-forecasting "
            "system now called Holt–Winters; Gardner and McKenzie (1985) supplied the "
            "optional damped-trend extension."
        ),
        how_it_works=(
            "Holt–Winters updates level, trend, and one seasonal state for each phase of a "
            "known cycle. Additive seasonality models a roughly constant seasonal amplitude; "
            "multiplicative seasonality scales that amplitude with the level."
        ),
        chrono_stream=(
            "The app supports additive or multiplicative trend and seasonality, optional "
            "trend damping, and automatic or manual alpha, beta, and gamma. Automatic mode "
            "jointly estimates smoothing parameters and initial states; manual mode uses "
            "statsmodels' heuristic initial states. It requires at least two complete cycles; multiplicative components "
            "require strictly positive values."
        ),
        when_to_use=(
            "Use it for regularly spaced data with a persistent trend and a single stable "
            "seasonal period supported by at least two full cycles."
        ),
        limitations=(
            "The seasonal period must be known, rapidly changing seasonality is difficult, "
            "and multiplicative states cannot handle nonpositive observations. Two cycles "
            "are a computational minimum, not a guarantee of reliable seasonal estimation."
        ),
        citation_ready=(
            "Holt–Winters forecasting represents a series with recursively updated level, "
            "trend, and seasonal states. Holt (1957/2004) developed the general exponentially "
            "weighted treatment of trends and seasonals, while Winters (1960) presented the "
            "seasonal forecasting system in operational form. Additive seasonality is "
            "appropriate when seasonal effects are roughly constant in size, whereas "
            "multiplicative seasonality lets them vary with the level; a trend may also be "
            "damped following Gardner and McKenzie (1985)."
        ),
        references=(HOLT_1957, WINTERS_1960, GARDNER_MCKENZIE_1985),
    ),
    "arima": MethodInformation(
        origin=(
            "ARIMA combines two earlier stochastic-model strands: Yule's (1927) "
            "autoregression and Slutzky's (1927/1937) moving sums of shocks. Box and "
            "Jenkins (1970) unified autoregression, integration by differencing, and "
            "moving-average errors into the iterative identification–estimation–checking "
            "workflow now called Box–Jenkins modeling. The app's transformations, tests, "
            "selection criteria, and diagnostics are later, separately cited additions."
        ),
        how_it_works=(
            "After any variance transformation, differencing order d removes stochastic "
            "level nonstationarity. An ARIMA(p,d,q) then explains the stationary result "
            "with p lagged values and q lagged innovations. Forecasts are integrated and "
            "inverse-transformed to the original scale. ACF/PACF patterns are diagnostic "
            "guides, not proofs of a unique order."
        ),
        chrono_stream=(
            "Chrono Stream follows the requested variance-before-mean sequence. It fits a "
            "reversible identity, Box–Cox, Yeo–Johnson, log, or square-root transform on "
            "training data only; chooses or accepts d using ADF, KPSS, Phillips–Perron, or "
            "ADF+KPSS consensus; records ACF/PACF; and searches guided, exhaustive, "
            "stepwise, or manual p/q candidates. Eligible candidates can be ranked by "
            "AICc, AIC, BIC, HQIC, or internal rolling-validation error. Strict selection "
            "requires stationarity, optimizer convergence, valid AR/MA roots, significance "
            "of every evaluated structural coefficient, the selected residual-normality "
            "test, degrees-of-freedom-corrected Ljung–Box or Box–Pierce white-noise tests "
            "at every selected lag, and a zero residual mean. The ARCH LM test is reported and can also be "
            "mandatory. The final model is independently retuned on all observations, and "
            "simulation can correct inverse-transformation mean bias."
        ),
        when_to_use=(
            "Use ARIMA for a single regularly spaced, nonseasonal series whose transformed "
            "and differenced dynamics can be represented by a parsimonious linear model. "
            "Use SARIMA instead when dependence repeats at a known seasonal period."
        ),
        limitations=(
            "Unit-root and residual tests have limited power in short samples, candidate "
            "search creates multiple-testing risk, and passing diagnostics does not prove "
            "the data-generating process is correct. Normal residuals are useful for "
            "Gaussian likelihood inference and intervals but are not universally required "
            "for useful point forecasts; Strict mode deliberately enforces the user's more "
            "conservative rule. An explicit near-match override is therefore labeled as an override."
        ),
        citation_ready=(
            "An ARIMA(p,d,q) model differences a series d times to address mean "
            "nonstationarity and represents the resulting process with p autoregressive "
            "terms and q moving-average innovation terms. The autoregressive construction "
            "has an early defining treatment in Yule (1927), while Slutzky (1927/1937) "
            "showed how moving sums of random shocks can produce serial patterns; Box and "
            "Jenkins (1970) unified these components with differencing and an iterative "
            "identification, estimation, diagnostic-checking, and forecasting procedure.\n\n"
            "In a comprehensive workflow, variance stabilization precedes differencing: "
            "Box–Cox or Yeo–Johnson transformations can stabilize scale-dependent variance "
            "before ADF, Phillips–Perron, or KPSS evidence is used to assess mean "
            "stationarity (Box & Cox, 1964; Dickey & Fuller, 1979; Phillips & Perron, "
            "1988; Kwiatkowski et al., 1992; Yeo & Johnson, 2000). Candidate orders may be "
            "guided by ACF/PACF and ranked with information criteria or rolling forecast "
            "validation (Akaike, 1974; Hannan & Quinn, 1979; Hurvich & Tsai, 1989; "
            "Hyndman & Khandakar, 2008; Schwarz, 1978). Adequacy checking should examine "
            "coefficient estimates, roots, residual plots and ACF/PACF, residual normality, "
            "a zero-mean test, and portmanteau white-noise tests such as Box–Pierce or "
            "Ljung–Box; conditional variance can additionally be checked with an ARCH LM "
            "test (Box & Pierce, 1970; Engle, 1982; Jarque & Bera, 1980; Ljung & Box, "
            "1978; Student, 1908)."
        ),
        references=ARIMA_CORE_REFERENCES,
    ),
    "sarima": MethodInformation(
        origin=(
            "Seasonal ARIMA is the multiplicative seasonal extension of the Box–Jenkins "
            "ARIMA framework (Box & Jenkins, 1970). It adds AR, differencing, and MA "
            "operators at a known seasonal lag m. Later seasonal-unit-root work explains "
            "why the seasonal difference D is a distinct decision (Hylleberg et al., 1990), "
            "while Osborn et al. (1988) and Canova and Hansen (1995) define the two formal "
            "seasonal tests exposed by the app."
        ),
        how_it_works=(
            "SARIMA(p,d,q)(P,D,Q)[m] combines ordinary ARIMA behavior with P seasonal AR "
            "terms, D differences y_t-y_(t-m), and Q seasonal innovation terms. The "
            "multiplicative polynomial includes interactions between ordinary and seasonal "
            "operators. Forecasts repeat learned dependence at the specified cycle length."
        ),
        chrono_stream=(
            "The complete ARIMA pipeline is retained, but seasonal differencing is decided "
            "before regular differencing. D can be automatic via OCSB, Canova–Hansen, or a "
            "documented seasonal-lag ACF heuristic, manual, or disabled. The configured m "
            "is used consistently in differencing, P/Q candidates, diagnostic lags, fitting, "
            "and forecasts. Strict selection also gates seasonal coefficients and seasonal "
            "roots; the white-noise table normally includes m and 2m when sample size permits."
        ),
        when_to_use=(
            "Use it when a regular series has autocorrelation that recurs at a known fixed "
            "cycle, such as 12 monthly observations per year, and enough complete cycles "
            "exist to estimate seasonal behavior."
        ),
        limitations=(
            "A wrong seasonal period invalidates the interpretation. Seasonal tests and "
            "parameters need substantially more data than nonseasonal ARIMA, multiple or "
            "changing seasonalities are not represented, and excessive d or D can overdifference. "
            "As with ARIMA, strict diagnostic passing is evidence of adequacy, not proof."
        ),
        citation_ready=(
            "A SARIMA(p,d,q)(P,D,Q)[m] model extends ARIMA by multiplying ordinary and "
            "seasonal autoregressive and moving-average polynomials and by permitting both "
            "ordinary differencing d and seasonal differencing D at period m. Box and "
            "Jenkins (1970) established this seasonal identification, estimation, checking, "
            "and forecasting framework. Seasonal integration is distinct from ordinary "
            "unit-root behavior (Hylleberg et al., 1990), so D should be assessed explicitly; "
            "the OCSB procedure derives from Osborn et al. (1988), while Canova and Hansen "
            "(1995) developed a complementary seasonal-stability test.\n\n"
            "Before SARIMA order selection, variance may be stabilized with a reversible "
            "power transformation and mean stationarity assessed after seasonal differencing "
            "(Box & Cox, 1964; Dickey & Fuller, 1979; Kwiatkowski et al., 1992; Yeo & "
            "Johnson, 2000). Candidate models should then be compared only after inspecting "
            "coefficient significance, stationary/invertible roots, residual ACF/PACF, a "
            "zero mean, normality, and multi-lag residual white-noise tests (Box & Pierce, "
            "1970; Jarque & Bera, 1980; Ljung & Box, 1978; Student, 1908)."
        ),
        references=(
            *ARIMA_CORE_REFERENCES,
            HEGY_1990,
            OCSB_1988,
            CANOVA_HANSEN_1995,
        ),
    ),
    "stl": MethodInformation(
        origin=(
            "Official X-11 is the Census Method II seasonal-adjustment program documented "
            "by Shiskin, Young, and Musgrave (1967), based on iterative moving-average "
            "filters, trading-day handling, extreme-value treatment, and diagnostics. "
            "Cleveland and Tiao (1976) supplied a stochastic-component interpretation. "
            "STL is a different loess-based decomposition introduced by Cleveland et al. (1990)."
        ),
        how_it_works=(
            "STL alternates loess smoothers to decompose an additive series into trend, "
            "seasonal, and remainder components, with optional robust weights for outliers. "
            "Decomposition alone does not define a future forecast, so a separate component "
            "extrapolation rule is required."
        ),
        chrono_stream=(
            "This page executes robust statsmodels STL—not Census X-11. Chrono Stream fits "
            "trend plus seasonal as in-sample values, extrapolates a straight line from the "
            "last min(n, max(2m, 8)) trend observations, averages the historical STL seasonal "
            "component by cycle phase, and recombines them. The method is therefore labeled "
            "'STL Decomposition Forecast (X-11-inspired)'."
        ),
        when_to_use=(
            "Use it as a transparent decomposition-based benchmark when there is one stable "
            "additive seasonal cycle and a locally linear terminal trend."
        ),
        limitations=(
            "It is not official X-11 or X-13ARIMA-SEATS: it has no trading-day/regARIMA "
            "stage, X-11 filter sequence, official diagnostics, or revision analysis. The "
            "linear trend and repeated seasonal forecast are Chrono Stream extensions, not "
            "part of the STL paper, and uncertainty intervals are residual approximations."
        ),
        citation_ready=(
            "Census X-11 is an iterative moving-average seasonal-adjustment system with "
            "special handling of trend-cycle, seasonal, trading-day, irregular, and extreme "
            "components (Shiskin et al., 1967); Cleveland and Tiao (1976) later interpreted "
            "its linear filters through a stochastic component model. STL is a separate "
            "procedure that uses repeated loess smoothing to decompose a series into trend, "
            "seasonal, and remainder components and can use robust weights to reduce outlier "
            "influence (Cleveland et al., 1990). Chrono Stream implements STL followed by a "
            "custom linear-trend and repeated-seasonal extrapolation. It should be cited "
            "as an STL decomposition forecast inspired by X-11 and is not an implementation of Census X-11."
        ),
        references=(SHISKIN_1967, CLEVELAND_TIAO_1976, CLEVELAND_STL_1990),
    ),
    "mstl_ets": MethodInformation(
        origin=(
            "Bandara, Hyndman, and Bergmeir (2025) introduced Multiple Seasonal-Trend "
            "decomposition using Loess (MSTL) to extend STL to series containing more than "
            "one seasonal pattern. The downstream ETS family follows the innovation "
            "state-space taxonomy of Hyndman et al. (2002)."
        ),
        how_it_works=(
            "MSTL iteratively applies STL smoothers so that each requested period receives "
            "its own seasonal component while a shared trend and remainder are estimated. "
            "Decomposition does not itself define a future forecast, so the adjusted series "
            "and every seasonal component require explicit extrapolation rules."
        ),
        chrono_stream=(
            "Chrono Stream subtracts all statsmodels MSTL seasonal components, fits a "
            "nonseasonal additive-error ETS model to the adjusted series, and selects no, "
            "additive, or damped-additive trend by AICc/AIC/BIC unless manually fixed. Each "
            "final decomposed seasonal cycle is repeated and added back. Reported intervals "
            "condition on those repeated seasonal paths and include ETS uncertainty only."
        ),
        when_to_use=(
            "Use it when regularly spaced data contain at least two credible and repeatedly "
            "observed seasonal periods, such as hourly data with daily and weekly cycles, "
            "and an additive decomposition is substantively reasonable."
        ),
        limitations=(
            "Periods are supplied rather than discovered, long cycles are data hungry, and "
            "repeating the terminal decomposed cycles assumes their shape remains stable. "
            "The ETS recombination rule is a transparent Chrono Stream forecasting choice, "
            "not a forecasting algorithm prescribed by the MSTL paper. Decomposition and "
            "period-estimation uncertainty are excluded from the conditional intervals."
        ),
        citation_ready=(
            "MSTL iteratively estimates multiple seasonal components through repeated STL "
            "decompositions (Bandara et al., 2025). Because MSTL is a decomposition rather "
            "than a complete forecast rule, Chrono Stream forecasts the seasonally adjusted "
            "series with nonseasonal additive-error ETS (Hyndman et al., 2002), repeats each "
            "last seasonal cycle, and adds the components."
        ),
        references=(MSTL_2025, CLEVELAND_STL_1990, HYNDMAN_ET_AL_2002),
    ),
    "prophet": MethodInformation(
        origin=(
            "Taylor and Letham (2018) introduced Prophet as an analyst-adjustable modular "
            "regression system for forecasting many business series. Its interpretable "
            "decomposition into trend, seasonal, event, and noise components follows the "
            "broader structural-component tradition described by Harvey and Shephard (1993)."
        ),
        how_it_works=(
            "Prophet models y(t) as a trend g(t), periodic seasonal terms s(t), optional "
            "holiday/event effects h(t), and an error. A piecewise-linear trend changes "
            "slope at candidate changepoints, with a sparsity prior controlling how many "
            "changes matter. Smooth seasonalities are Fourier regressions and components "
            "can combine additively or multiplicatively."
        ),
        chrono_stream=(
            "Chrono Stream calls the official Prophet package with a piecewise-linear trend, "
            "automatic candidate changepoints, adjustable changepoint-prior scale, automatic "
            "or disabled weekly/yearly seasonalities, and additive or multiplicative "
            "seasonality. Daily seasonality and holiday regressors are disabled on this page. "
            "The plotted interval is Prophet's native 95% uncertainty interval."
        ),
        when_to_use=(
            "Use it for regular calendar data with interpretable recurring weekly or yearly "
            "patterns, trend changes, enough history, and possible missing timestamps or outliers."
        ),
        limitations=(
            "Automatic calendar seasonalities do not discover arbitrary periods, abrupt "
            "future regime changes remain unknowable, and default uncertainty relies on "
            "assumptions about future changepoint frequency. This page exposes only a "
            "subset of Prophet, with no holidays, logistic capacity, or custom regressors."
        ),
        citation_ready=(
            "Prophet is a modular regression forecasting procedure in which a piecewise "
            "linear or logistic trend is combined with Fourier-series seasonalities, holiday "
            "effects, and an error term. Candidate trend changepoints are regularized so that "
            "only supported rate changes have substantial effect, while components remain "
            "interpretable and adjustable by an analyst (Taylor & Letham, 2018). This "
            "trend-plus-seasonal construction belongs to the broader structural time-series "
            "tradition of modeling directly interpretable components (Harvey & Shephard, 1993). "
            "Chrono Stream uses Prophet's linear trend and weekly/yearly seasonalities but "
            "does not enable holiday terms or logistic growth."
        ),
        references=(TAYLOR_LETHAM_2018, HARVEY_SHEPHARD_1993),
    ),
    "lstm": MethodInformation(
        origin=(
            "Hochreiter and Schmidhuber (1997) introduced Long Short-Term Memory to address "
            "vanishing error signals in recurrent learning. Gers, Schmidhuber, and Cummins "
            "(2000) added the forget gate used by modern LSTM layers, and Gers, Eck, and "
            "Schmidhuber (2001) directly studied LSTM on window-predictable time series."
        ),
        how_it_works=(
            "An LSTM recurrently updates a memory cell through learned input, forget, and "
            "output gates. Given a lag window, it can retain or discard information at "
            "different delays and map the final hidden representation to the next value."
        ),
        chrono_stream=(
            "The app min–max scales one univariate series, creates supervised samples from "
            "fixed past-only lookback windows, trains one Keras LSTM layer followed by a "
            "Dense output using Adam and mean-squared error, and inverse-scales predictions. "
            "Multi-step forecasts are recursive. A fixed seed and shuffle=False improve, "
            "but do not guarantee, reproducibility."
        ),
        when_to_use=(
            "Use it experimentally when nonlinear temporal dependence is plausible and "
            "substantially more data are available than model parameters. Compare it against "
            "simpler holdout baselines before trusting the added complexity."
        ),
        limitations=(
            "Small univariate datasets make neural estimates unstable, hyperparameters are "
            "not automatically tuned, recursive errors compound, and the residual-based "
            "interval is not a calibrated neural predictive distribution. Scaling and model "
            "training are correctly confined to each fitting partition, but the final seed "
            "does not remove platform-level nondeterminism."
        ),
        citation_ready=(
            "Long Short-Term Memory is a gated recurrent neural architecture designed to "
            "preserve trainable error flow across long delays (Hochreiter & Schmidhuber, "
            "1997). The modern forget gate lets a cell reset obsolete state during continual "
            "prediction (Gers et al., 2000), and early direct time-series experiments showed "
            "that LSTM should be compared with simpler fixed-window approaches rather than "
            "assumed superior (Gers et al., 2001). Chrono Stream trains a compact, single-layer "
            "LSTM on normalized lag windows and recursively feeds each one-step prediction "
            "back as input for multi-step forecasting."
        ),
        references=(
            HOCHREITER_SCHMIDHUBER_1997,
            GERS_FORGET_2000,
            GERS_TIME_SERIES_2001,
        ),
    ),
    "cnn": MethodInformation(
        origin=(
            "Fukushima's neocognitron (1980) is an explicit hierarchical local-receptive-field "
            "precursor to CNNs. LeCun et al. (1989) made shared convolutional weights trainable "
            "end to end with backpropagation, and LeCun et al. (1998) consolidated the modern "
            "CNN formulation. Borovykh, Bohte, and Oosterlee (2017) explicitly adapted a "
            "convolutional architecture to time-series forecasting. Their deep dilated "
            "WaveNet-style network is related research, not the app's exact compact network."
        ),
        how_it_works=(
            "A one-dimensional convolution slides learned kernels across a lag window so the "
            "same local pattern detector is used at every position. Pooling compresses the "
            "result, and dense layers map extracted features to the next numeric value."
        ),
        chrono_stream=(
            "Chrono Stream min–max scales the series, forms past-only windows, and trains one "
            "Conv1D ReLU layer, global-average pooling, a 16-unit ReLU layer, and one scalar "
            "output with Adam/MSE. It then inverse-scales and recursively forecasts. It is a "
            "compact 1D CNN baseline—not a causal dilated TCN or WaveNet reproduction."
        ),
        when_to_use=(
            "Use it experimentally when short local motifs inside a fixed lookback window may "
            "be predictive and enough examples exist to learn convolutional filters."
        ),
        limitations=(
            "Global-average pooling discards exact position after feature extraction, the "
            "single convolution has a limited receptive field, hyperparameters are not tuned, "
            "and recursive multi-step error accumulates. Its approximate residual interval "
            "does not quantify neural parameter uncertainty."
        ),
        citation_ready=(
            "A one-dimensional convolutional neural network applies shared learned filters "
            "across positions in a time window, allowing local motifs to be detected with far "
            "fewer parameters than separate position-specific connections. The defining CNN "
            "architectural roots include Fukushima's (1980) neocognitron, while shared "
            "convolutional weights were trained end to end by backpropagation in LeCun et al. "
            "(1989) and the modern CNN formulation was consolidated by LeCun et al. (1998). "
            "Convolutional architectures were later formulated explicitly for time-series "
            "forecasting by Borovykh et al. (2017). Chrono Stream uses a smaller "
            "Conv1D-plus-global-pooling baseline and "
            "recursive forecasting, so results should not be attributed to the deeper dilated "
            "architecture evaluated in that forecasting paper."
        ),
        references=(FUKUSHIMA_1980, LECUN_1989, LECUN_1998, BOROVYKH_2017),
    ),
    "nbeats": MethodInformation(
        origin=(
            "Oreshkin, Carpov, Chapados, and Bengio (2020) introduced N-BEATS as a "
            "deep fully connected forecasting architecture based on basis-expansion blocks "
            "and backward and forward residual links. Their paper distinguishes a generic "
            "learned-basis form from constrained interpretable trend and seasonality stacks."
        ),
        how_it_works=(
            "Each block maps a fixed historical backcast window through dense nonlinear "
            "layers to a backcast vector and a multi-step forecast vector. The backcast is "
            "subtracted from the block input, later blocks explain the residual, and all "
            "block forecast vectors are added to produce the complete requested horizon."
        ),
        chrono_stream=(
            "Chrono Stream standardizes only the current fitting partition and trains a "
            "compact generic N-BEATS with learned backcast and forecast heads, Adam/MSE, "
            "ordered windows, and seed 42. It predicts the requested horizon directly, a "
            "multi-step strategy reviewed by Ben Taieb et al. (2012). This "
            "is neither the paper's constrained trend/seasonality-stack variant nor its large "
            "global ensemble, and its displayed residual intervals are approximate."
        ),
        when_to_use=(
            "Use it experimentally when ample history supports a nonlinear direct-horizon "
            "model and fixed-window patterns may not be summarized well by a small classical "
            "model. Compare it with naive, ETS, and simpler neural baselines on the holdout."
        ),
        limitations=(
            "Direct training loses complete examples as lookback or horizon grows, neural "
            "optimization is stochastic and compute intensive, and a local univariate fit "
            "cannot inherit the scale and diversity advantages of the paper's cross-series "
            "experiments. Learned generic bases are not automatically interpretable, while "
            "residual bands do not represent neural parameter or model uncertainty."
        ),
        citation_ready=(
            "N-BEATS uses fully connected basis-expansion blocks with doubly residual "
            "connections: each block removes a backcast from its input and contributes an "
            "additive multi-horizon forecast (Oreshkin et al., 2020). Chrono Stream implements "
            "the generic learned-basis form as one compact local model; it does not reproduce "
            "the paper's interpretable constrained stacks or forecasting ensemble."
        ),
        references=(ORESHKIN_ET_AL_2020, BEN_TAIEB_2012),
    ),
    "tcn": MethodInformation(
        origin=(
            "Bai, Kolter, and Koltun (2018) evaluated a generic Temporal Convolutional "
            "Network built from causal dilated convolutions and residual blocks against "
            "recurrent sequence models. Its causal dilation lineage includes WaveNet, which "
            "van den Oord et al. (2016) introduced for autoregressive audio generation; "
            "Borovykh et al. (2017) applied convolutional architectures to conditional "
            "time-series forecasting."
        ),
        how_it_works=(
            "A causal convolution uses only the current and earlier positions. Exponentially "
            "increasing dilation spaces kernel taps farther apart, expanding the receptive "
            "field without a recurrent state, while residual shortcuts support optimization "
            "through stacked convolutional blocks."
        ),
        chrono_stream=(
            "Chrono Stream standardizes the current fitting partition, constructs past-only "
            "univariate windows, and trains residual blocks containing two causal Conv1D "
            "layers at dilations 1, 2, 4, and onward. The final causal state predicts one next "
            "value, which is fed back recursively. Seed 42 and ordered batches improve "
            "repeatability; intervals remain residual approximations."
        ),
        when_to_use=(
            "Use it experimentally when dependencies may span several lag scales and enough "
            "ordered examples exist to learn convolutional filters. It is especially useful "
            "as a causal-dilation comparator to a plain CNN or recurrent LSTM."
        ),
        limitations=(
            "A theoretical receptive field larger than the supplied lookback sees causal zero "
            "padding rather than older observations. Recursive output compounds errors, the "
            "small local network is not the benchmark suite or exact architecture of Bai et "
            "al., and it is not WaveNet's probabilistic audio model. Hyperparameters are not "
            "automatically tuned and residual bands are not calibrated neural uncertainty."
        ),
        citation_ready=(
            "A generic TCN combines causal convolutions, exponentially increasing dilations, "
            "and residual connections to process sequences without recurrent state (Bai et "
            "al., 2018), drawing on the causal dilated-convolution lineage of WaveNet (van den "
            "Oord et al., 2016). Chrono Stream fits a compact past-only version and generates "
            "multi-step forecasts recursively, rather than reproducing either cited system exactly."
        ),
        references=(BAI_ET_AL_2018, VAN_DEN_OORD_ET_AL_2016, BOROVYKH_2017),
    ),
    "xgboost": MethodInformation(
        origin=(
            "Friedman (2001) developed gradient boosting as stagewise optimization in "
            "function space, including regression-tree learners. Chen and Guestrin (2016) "
            "introduced XGBoost's regularized objective and scalable tree-building system. "
            "Turning a series into lagged supervised examples and recursively forecasting "
            "multiple steps follows the strategy literature evaluated by Ben Taieb et al. (2012)."
        ),
        how_it_works=(
            "Boosted trees are added sequentially so each new tree reduces the current loss, "
            "while regularization penalizes model complexity. For a univariate series, the "
            "last w observations become input features and the next observation is the target."
        ),
        chrono_stream=(
            "The app trains XGBRegressor with squared-error loss on past-only lag vectors. "
            "Users control lookback, tree count, maximum depth, and learning rate; row and "
            "column subsampling are 0.9 and the seed is fixed. Each future prediction is "
            "appended to the lag history, making this a recursive strategy."
        ),
        when_to_use=(
            "Use it when lag effects may be nonlinear or interactive and there are enough "
            "windows for tree learning. It is especially useful as a nonlinear comparator "
            "to ARIMA and smoothing baselines."
        ),
        limitations=(
            "It knows time only through lag positions, so no calendar or seasonal features "
            "are included beyond what the lookback contains. Recursive forecast errors "
            "accumulate, extrapolation outside learned target patterns is weak, and the "
            "displayed interval is a residual approximation rather than an XGBoost distribution."
        ),
        citation_ready=(
            "XGBoost is a regularized implementation of gradient-boosted decision trees. "
            "Gradient boosting builds an additive predictor stage by stage to descend a "
            "chosen loss in function space (Friedman, 2001), while XGBoost adds a regularized "
            "objective and scalable algorithms for tree construction (Chen & Guestrin, "
            "2016). Chrono Stream converts a univariate series into lagged supervised "
            "examples and uses the recursive multi-step strategy: each predicted value is "
            "fed into the next lag vector, a strategy whose error-propagation tradeoffs are "
            "examined by Ben Taieb et al. (2012)."
        ),
        references=(FRIEDMAN_2001, CHEN_GUESTRIN_2016, BEN_TAIEB_2012),
    ),
    "lagged_linear": MethodInformation(
        origin=(
            "Lagged linear forecasting joins ordinary least squares with autoregressive "
            "prediction. Yule (1927) established the autoregressive idea of explaining a "
            "series from its own past, while least-squares estimation has a much earlier "
            "lineage in Legendre (1805). The implementation is supervised regression on "
            "causal time-indexed rows rather than a full stochastic AR error model."
        ),
        how_it_works=(
            "Each target y_t is paired with y_(t-1) through y_(t-p), optionally augmented "
            "by shifted rolling summaries, a longer seasonal lag, and sine/cosine calendar "
            "values known for time t. OLS minimizes squared one-step residuals. Multi-step "
            "forecasts recursively append each prediction before constructing the next row."
        ),
        chrono_stream=(
            "Chrono Stream rejects irregular dates, creates every rolling statistic from "
            "values strictly before its target, and confines automatic lag selection to "
            "expanding-window folds inside the current training partition. The chosen lags, "
            "feature names, coefficients, fold scores, and recursive strategy are recorded; "
            "displayed intervals are explicitly residual approximations."
        ),
        when_to_use=(
            "Use it as an interpretable autoregressive benchmark when dependence is broadly "
            "linear and lag coefficients or known calendar cycles are substantively useful."
        ),
        limitations=(
            "Adjacent lags are often highly correlated, making individual OLS coefficients "
            "unstable even when predictions are adequate. The equation does not automatically "
            "difference nonstationary data, model innovation autocorrelation, or protect against "
            "breaks; recursive errors compound and residual intervals are not model-native."
        ),
        citation_ready=(
            "Lagged linear regression represents the next observation as a least-squares "
            "function of earlier observations, connecting autoregression (Yule, 1927) with "
            "causal supervised forecasting. Chrono Stream chooses lags by expanding-window "
            "validation (Tashman, 2000) and generates multiple horizons recursively, whose "
            "error-propagation tradeoff is documented by Ben Taieb et al. (2012)."
        ),
        references=(YULE_1927, LEGENDRE_1805, TASHMAN_2000, BEN_TAIEB_2012),
    ),
    "regularized_regression": MethodInformation(
        origin=(
            "Regularized lag regression applies penalized linear estimation to correlated "
            "time-lag predictors. Hoerl and Kennard (1970) introduced ridge regression, "
            "Tibshirani (1996) introduced the Lasso, and Zou and Hastie (2005) combined L1 "
            "and L2 penalties in Elastic Net. These are related estimators exposed as one "
            "conceptual method with an explicit penalty selector."
        ),
        how_it_works=(
            "Ridge shrinks squared coefficient magnitude, Lasso penalizes absolute magnitude "
            "and can set coefficients to zero, and Elastic Net combines both penalties. The "
            "lag matrix is standardized so penalties compare predictors on a common scale. "
            "Alpha controls total shrinkage; Elastic Net's L1 ratio controls the mixture."
        ),
        chrono_stream=(
            "Chrono Stream wraps StandardScaler and the selected estimator in one pipeline, "
            "so scaling is refitted separately within every expanding training fold and again "
            "on the complete fit partition. Automatic mode searches a bounded alpha grid and "
            "Elastic Net ratios; manual mode preserves user values. Forecasting remains recursive "
            "and all standardized coefficients and candidate scores are reported."
        ),
        when_to_use=(
            "Use it when many overlapping lags make OLS unstable, when shrinkage may improve "
            "out-of-sample prediction, or when sparse lag selection is desirable."
        ),
        limitations=(
            "A selected zero coefficient is not proof that a lag has no effect, especially "
            "when predictors are correlated. Results depend on scaling, alpha, L1 ratio, fold "
            "origins, and structural stability. The method still assumes one global linear "
            "response and its residual interval does not incorporate tuning uncertainty."
        ),
        citation_ready=(
            "Regularized lag regression stabilizes a linear autoregression by penalizing its "
            "coefficients: ridge uses L2 shrinkage (Hoerl & Kennard, 1970), Lasso uses L1 "
            "shrinkage and selection (Tibshirani, 1996), and Elastic Net combines the two to "
            "encourage grouped behavior among correlated predictors (Zou & Hastie, 2005)."
        ),
        references=(HOERL_KENNARD_1970, TIBSHIRANI_1996, ZOU_HASTIE_2005, TASHMAN_2000),
    ),
    "cart": MethodInformation(
        origin=(
            "Breiman, Friedman, Olshen, and Stone (1984) systematized Classification and "
            "Regression Trees (CART), including binary recursive partitioning for continuous "
            "responses and complexity control. CART and decision-tree regression are therefore "
            "not separate forecast methods here: one page applies the regression-tree method "
            "to lag-derived time-series examples."
        ),
        how_it_works=(
            "A regression tree repeatedly chooses a feature and threshold that reduce squared "
            "error in child nodes. A terminal leaf predicts the average training target within "
            "its partition. Maximum depth and minimum leaf size control the bias–variance tradeoff, "
            "while past-only lags translate temporal history into predictors the tree can consume."
        ),
        chrono_stream=(
            "Chrono Stream fits one seeded sklearn DecisionTreeRegressor. Automatic mode chooses "
            "depth and leaf size by bounded expanding-window validation; manual mode exposes both. "
            "It reports realized depth, leaves, impurity importances, causal features, and the "
            "recursive strategy. The residual bands are approximations, not ordinary CART 95% intervals."
        ),
        when_to_use=(
            "Use CART when a compact set of nonlinear thresholds or interactions is plausible "
            "and the resulting tree structure is more valuable than a smooth equation."
        ),
        limitations=(
            "One tree is unstable to modest data changes, leaf predictions are piecewise constant, "
            "and extrapolation beyond learned target regions is weak. Impurity importance is descriptive "
            "rather than inferential, recursive errors accumulate, and small time series offer few "
            "independent regimes from which to learn reliable splits."
        ),
        citation_ready=(
            "CART recursively partitions predictor space and assigns a constant response estimate "
            "within each terminal region (Breiman et al., 1984). Chrono Stream uses lagged causal "
            "rows, tunes complexity with ordered forecast-origin folds (Tashman, 2000), and warns "
            "that recursive multi-step use can propagate early prediction errors."
        ),
        references=(BREIMAN_ET_AL_1984, TASHMAN_2000, BEN_TAIEB_2012),
    ),
    "random_forest": MethodInformation(
        origin=(
            "Breiman (2001) defined Random Forests as ensembles of randomized tree predictors. "
            "They extend the CART regression-tree foundation of Breiman et al. (1984) by combining "
            "bootstrap resampling with random feature subsets, reducing the dependence among trees "
            "so their average can be more stable than a single tree."
        ),
        how_it_works=(
            "Each regression tree is trained on a bootstrap sample and considers a random subset "
            "of features at a split. The forest prediction averages terminal-leaf predictions across "
            "trees. Tree count controls Monte Carlo stability; depth, leaf size, and feature fraction "
            "control complexity and diversity. Time enters only through constructed lag/calendar rows."
        ),
        chrono_stream=(
            "Chrono Stream uses a fixed seed and one worker for reproducible sklearn forests. Automatic "
            "mode performs a bounded expanding-window search over depth, leaf size, and feature fraction "
            "with 100 trees; manual mode exposes these plus tree count. It logs feature importances, "
            "bootstrap use, candidate errors, and recursive forecasting. Intervals remain residual approximations."
        ),
        when_to_use=(
            "Use it for nonlinear lag interactions when sufficient supervised rows exist and a more "
            "stable ensemble is preferred to one highly variable decision tree."
        ),
        limitations=(
            "A forest is less transparent than one tree and still predicts by averaging learned leaf "
            "targets, so it generally cannot extend a new linear trend beyond observed response regions. "
            "Impurity importance can be biased, temporal regimes can change, recursive inputs drift from "
            "training states, and the app does not claim native or calibrated forest intervals."
        ),
        citation_ready=(
            "Random Forest regression averages many CART-style predictors built with bootstrap samples "
            "and randomized feature selection, trading one tree's instability for ensemble diversity "
            "(Breiman, 2001; Breiman et al., 1984). Chrono Stream applies this learner to past-only lag "
            "states and selects bounded complexity using expanding forecast origins (Tashman, 2000)."
        ),
        references=(BREIMAN_2001, BREIMAN_ET_AL_1984, TASHMAN_2000),
    ),
    "support_vector_regression": MethodInformation(
        origin=(
            "Vapnik (1995) established the support-vector function-estimation framework, and "
            "Vapnik, Golowich, and Smola (1997) presented it for regression estimation. "
            "Drucker et al. (1997) supplied an early regression application and comparison, while "
            "Smola and Schölkopf (2004) later consolidated its optimization, kernel, sparsity, "
            "and regularization interpretation. Chrono Stream applies epsilon-SVR to "
            "lag-derived states rather than claiming a specialized stochastic time-series likelihood."
        ),
        how_it_works=(
            "Epsilon-insensitive loss ignores errors inside a tube of width epsilon and penalizes deviations "
            "outside it, while C balances flatness against violations. A linear kernel estimates a regularized "
            "linear surface; the radial-basis kernel represents smooth nonlinear similarity. Only support vectors "
            "with active constraints determine the fitted function."
        ),
        chrono_stream=(
            "Chrono Stream standardizes both causal features and the target inside a leakage-safe "
            "transformed-target pipeline, then inverse-transforms predictions. Automatic mode "
            "searches bounded linear/RBF, C, epsilon, and gamma settings with expanding folds; manual mode exposes "
            "the principal controls, with C and epsilon interpreted on the standardized target scale. "
            "The selected support-vector count and every candidate score are recorded. "
            "Multi-step forecasts recurse and uncertainty bands are residual approximations."
        ),
        when_to_use=(
            "Use SVR when a smooth nonlinear relationship among a moderate number of lag features is plausible "
            "and the dataset is large enough to tune regularization without an excessive search."
        ),
        limitations=(
            "SVR is sensitive to scale and tuning, RBF behavior is difficult to explain, and fitting cost grows "
            "quickly with sample size. Kernel regression does not guarantee trend extrapolation, recursive inputs "
            "may move away from observed states, and epsilon-insensitive fitting supplies no native predictive "
            "distribution for the app's displayed interval."
        ),
        citation_ready=(
            "Support Vector Regression estimates a regularized function under epsilon-insensitive loss, using "
            "kernels to represent nonlinear relations when required (Vapnik, 1995; Vapnik et al., 1997). "
            "Drucker et al. (1997) provided an early application, and Smola and Schölkopf (2004) a later tutorial. "
            "Chrono Stream standardizes only within training folds and tunes candidates with ordered "
            "expanding validation rather than shuffled cross-validation (Tashman, 2000)."
        ),
        references=(
            VAPNIK_1995,
            VAPNIK_GOLOWICH_SMOLA_1997,
            DRUCKER_ET_AL_1997,
            SMOLA_SCHOLKOPF_2004,
            TASHMAN_2000,
        ),
    ),
    "knn_regression": MethodInformation(
        origin=(
            "Nearest-neighbor regression belongs to nonparametric conditional-mean estimation. Stone (1977) "
            "established broad consistency results including nearest-neighbor weight rules, and Altman (1992) "
            "provided a clear treatment of kernel and nearest-neighbor regression. The forecasting adaptation "
            "treats each lag window as a state whose historical successors are candidate outcomes."
        ),
        how_it_works=(
            "After standardization, the method finds k training rows closest to the current lag/calendar vector "
            "under Manhattan or Euclidean distance. It predicts the uniform or inverse-distance average of their "
            "targets. Small k is locally adaptive but variable; large k is smoother but can average distinct regimes."
        ),
        chrono_stream=(
            "Chrono Stream fits StandardScaler only inside each training pipeline, limits automatic k values to the "
            "smallest expanding fold, and searches neighbor count, weighting, and distance power without shuffling. "
            "Manual settings are also available. Future lag states contain earlier predictions, the chosen distance "
            "contract is reported, and intervals are residual approximations."
        ),
        when_to_use=(
            "Use it when historically similar lag patterns recur and a local analog forecast is easier to justify "
            "than one global parametric response function."
        ),
        limitations=(
            "Distance becomes less informative as feature dimension grows, duplicate or sparse states can dominate, "
            "and scaling choices materially change neighbors. kNN averages observed target values rather than "
            "extrapolating new levels; recursive forecasts can enter regions without good analogs, and consistency "
            "theory does not guarantee accuracy for a short dependent series."
        ),
        citation_ready=(
            "k-nearest-neighbor regression estimates a response from the targets attached to nearby predictor states, "
            "with locality controlled by k and optional distance weights (Stone, 1977; Altman, 1992). Chrono Stream "
            "uses standardized, past-only lag states and expanding-window tuning (Tashman, 2000), then recursively "
            "queries predicted future states."
        ),
        references=(STONE_1977, ALTMAN_1992, TASHMAN_2000),
    ),
    "extra_trees": MethodInformation(
        origin=(
            "Geurts, Ernst, and Wehenkel (2006) introduced Extremely Randomized Trees, increasing ensemble diversity "
            "by randomizing split thresholds as well as candidate attributes. The method shares regression-tree roots "
            "with CART and ensemble motivation with Random Forests, but it is not simply a renamed Random Forest."
        ),
        how_it_works=(
            "Many regression trees are grown from the full training sample by default. At each node, random candidate "
            "features and random cut points are proposed, and the best proposal under the split criterion is used. "
            "Averaging the trees reduces variance introduced by this strong randomization and yields piecewise-constant "
            "predictions over lag-feature space."
        ),
        chrono_stream=(
            "Chrono Stream uses sklearn ExtraTreesRegressor without bootstrap sampling, a fixed seed, and one worker. "
            "Automatic expanding-window validation tunes depth, leaf size, and feature fraction with a bounded 100-tree "
            "search; manual mode exposes those controls and tree count. It reports threshold-randomization semantics, "
            "feature importances, candidate scores, and recursive forecasts."
        ),
        when_to_use=(
            "Use Extra Trees as a fast, strongly randomized nonlinear comparator when a Random Forest may retain too "
            "much split correlation and enough lag rows exist for ensemble learning."
        ),
        limitations=(
            "Additional randomization can increase bias, leaf averages remain poor extrapolators, and ensemble feature "
            "importance is not a significance test. Results still depend on lag construction and regimes represented "
            "in the sample. Recursive errors compound, while the residual interval does not measure variation across "
            "trees or provide calibrated coverage."
        ),
        citation_ready=(
            "Extra Trees strongly randomizes both feature and cut-point selection before averaging regression trees "
            "(Geurts et al., 2006), distinguishing it from Breiman's bootstrap-oriented Random Forest formulation "
            "(Breiman, 2001). Chrono Stream fits it to causal lag rows and tunes only inside expanding training folds."
        ),
        references=(GEURTS_ET_AL_2006, BREIMAN_2001, TASHMAN_2000),
    ),
    "theta": MethodInformation(
        origin=(
            "Assimakopoulos and Nikolopoulos (2000) introduced the Theta model and its combination of transformed "
            "curvature lines, which performed strongly in the M3 competition. Hyndman and Billah (2003) subsequently "
            "showed that the conventional forecast has a much simpler representation closely related to simple "
            "exponential smoothing with drift."
        ),
        how_it_works=(
            "The classical theta-two forecast combines a long-run slope contribution with a simple-exponential-smoothing "
            "level. The general theta coefficient changes the trend weight. Optional classical seasonal decomposition "
            "is applied before fitting and reversed afterward; estimation can use two-step OLS plus SES or the equivalent "
            "ARIMA(0,1,1)-with-drift maximum likelihood form."
        ),
        chrono_stream=(
            "Chrono Stream uses statsmodels ThetaModel, supports automatic seasonality testing, forced or disabled "
            "seasonal adjustment, additive/multiplicative/automatic decomposition, theta, and two-step or MLE estimation. "
            "Historical fitted values are expanding-prefix one-step forecasts wherever a prefix has enough history; "
            "at the exact two-cycle seasonal minimum none are available. The integrated-MA Gaussian interval variance "
            "is estimated from the same deseasonalized series used for fitting rather than the raw seasonal series."
        ),
        when_to_use=(
            "Use Theta as a parsimonious general-purpose benchmark for level-and-drift series, especially when more "
            "complex models need a strong lightweight comparator."
        ),
        limitations=(
            "Seasonal decomposition requires a defensible period and at least two cycles, structural breaks can invalidate "
            "the drift, and one smoothed level cannot represent rich dynamics. The conventional success of theta=2 does "
            "not make it universally optimal. Native intervals rely on Gaussian and integrated-MA assumptions, and forced "
            "multiplicative adjustment requires positive observations."
        ),
        citation_ready=(
            "The Theta method combines extrapolations of series whose curvature is transformed while mean and slope are "
            "preserved (Assimakopoulos & Nikolopoulos, 2000). Its conventional two-line forecast is algebraically "
            "equivalent to a simple-exponential-smoothing forecast with drift (Hyndman & Billah, 2003), which is the "
            "computational interpretation used by statsmodels and Chrono Stream. Theta values other than the conventional "
            "two belong to the later generalized/optimized family (Fiorucci et al., 2016)."
        ),
        references=(
            ASSIMAKOPOULOS_NIKOLOPOULOS_2000,
            HYNDMAN_BILLAH_2003,
            FIORUCCI_ET_AL_2016,
            MAKRIDAKIS_HIBON_2000,
        ),
    ),
    "automatic_ets": MethodInformation(
        origin=(
            "Hyndman, Koehler, Snyder, and Grose (2002) organized exponential smoothing as an innovation state-space "
            "taxonomy indexed by Error, Trend, and Seasonality—ETS—and showed how likelihood and information criteria "
            "enable automatic forecasting. Hyndman et al. (2008) developed the framework, estimation, admissibility, "
            "simulation, and intervals comprehensively."
        ),
        how_it_works=(
            "An ETS structure specifies additive or multiplicative observation errors, absent/additive/multiplicative "
            "trend, optional damping, and absent/additive/multiplicative seasonality. Each candidate is fitted by maximum "
            "likelihood with estimated states. Automatic selection minimizes AICc, AIC, or BIC among converged finite "
            "candidates fitted to the same partition."
        ),
        chrono_stream=(
            "Chrono Stream searches a bounded, disclosed statsmodels ETS grid. Multiplicative candidates are omitted for "
            "nonpositive data, seasonal candidates require two cycles, and every attempted success or failure is retained. "
            "Manual mode selects an exact structure. The winning error/trend/damping/seasonal tuple, parameters, criteria, "
            "convergence, fitted states, and native state-space intervals are saved."
        ),
        when_to_use=(
            "Use Automatic ETS when evolving level, trend, or one stable seasonal cycle is plausible but the appropriate "
            "additive/multiplicative structure is not known in advance."
        ),
        limitations=(
            "Information criteria rank fitted candidates but do not prove adequacy or future accuracy. Multiplicative "
            "forms require positive data, seasonal estimation needs repeated cycles, and one ETS seasonal period cannot "
            "represent complex multiple calendars. A large candidate family adds selection uncertainty, while optimizer "
            "convergence and structural stability remain separate concerns."
        ),
        citation_ready=(
            "Automatic ETS treats exponential smoothing methods as innovation state-space models indexed by error, trend, "
            "and seasonal components, permitting likelihood-based selection and predictive intervals (Hyndman et al., "
            "2002, 2008). Chrono Stream ranks only converged candidates by the selected information criterion and preserves "
            "the full attempted-candidate record."
        ),
        references=(HYNDMAN_ET_AL_2002, HYNDMAN_ET_AL_2008, HOLT_1957, WINTERS_1960),
    ),
    "tbats": MethodInformation(
        origin=(
            "De Livera, Hyndman, and Snyder (2011) introduced the framework now called TBATS for complex seasonality. "
            "Its acronym summarizes Trigonometric seasonality, Box–Cox transformation, ARMA errors, Trend, and Seasonal "
            "components. It extends innovation state-space smoothing to periods that are multiple, long, high-frequency, "
            "or non-integer."
        ),
        how_it_works=(
            "Seasonal cycles are represented with trigonometric harmonics whose states evolve over time, avoiding one "
            "separate state for every position of a long period. A Box–Cox transform can stabilize positive-scale variance, "
            "a local trend may be damped, and ARMA terms can capture remaining short-run errors. Candidate component "
            "configurations are evaluated by AIC in the installed implementation."
        ),
        chrono_stream=(
            "Chrono Stream uses the pinned tbats package lazily, fixes one worker, accepts comma-separated floating-point "
            "periods, and requires two repetitions of the longest period. Automatic mode searches Box–Cox when positivity "
            "allows it plus trend and damping, with optional ARMA errors; manual mode fixes switches. Selected harmonics, "
            "lambda, damping, ARMA orders, AIC, warnings, and native Gaussian intervals are reported."
        ),
        when_to_use=(
            "Use TBATS when a regularly spaced series has credible multiple or non-integer seasonal cycles that simpler "
            "Holt–Winters or one-period ETS cannot represent."
        ),
        limitations=(
            "TBATS can be computationally costly and data hungry, especially with long periods and ARMA search. User-supplied "
            "periods remain substantive assumptions, Box–Cox requires positivity, and evolving events or structural breaks "
            "are not explained merely by Fourier flexibility. Gaussian intervals and AIC comparisons inherit distributional "
            "and candidate-set assumptions."
        ),
        citation_ready=(
            "TBATS embeds trigonometric multiple-seasonal states, optional Box–Cox transformation, damped trend, and ARMA "
            "errors in an innovation state-space forecast (De Livera et al., 2011). Chrono Stream exposes those component "
            "choices and uses native interval output; the Box–Cox option follows the transformation family of Box and Cox (1964)."
        ),
        references=(DE_LIVERA_ET_AL_2011, BOX_COX_1964, HYNDMAN_ET_AL_2008),
    ),
    "croston_family": MethodInformation(
        origin=(
            "Croston (1972) proposed separately smoothing positive demand sizes and the "
            "intervals between them for intermittent demand. Syntetos and Boylan (2005) "
            "analyzed the estimator's bias and supplied the approximation now called SBA. "
            "Teunter, Syntetos, and Babai (2011) replaced interval smoothing with an "
            "occurrence-probability update designed to react to obsolescence."
        ),
        how_it_works=(
            "Croston and SBA update a demand-size state and an inter-demand-interval state "
            "only when positive demand occurs, then divide size by interval; SBA multiplies "
            "that ratio by 1-alpha/2. TSB smooths a binary occurrence indicator every period "
            "and multiplies its probability state by the smoothed positive size."
        ),
        chrono_stream=(
            "Exact zero means no demand and negative observations are rejected. The first "
            "positive demand initializes size and its one-based location initializes timing. "
            "Automatic mode evaluates bounded alpha—and beta for TSB—using causal one-step "
            "MAE or RMSE on only the fitting partition, retaining zero periods. Every future "
            "point is the terminal expected demand per period; intervals are clipped-at-zero "
            "residual approximations rather than native inventory distributions."
        ),
        when_to_use=(
            "Use the family for nonnegative, regularly spaced demand with many genuine "
            "zero-demand periods. SBA is a bias-corrected Croston comparator; TSB is useful "
            "when the probability of demand can decay, including possible obsolescence."
        ),
        limitations=(
            "A zero caused by missing collection is not a no-demand event, so preprocessing "
            "semantics are decisive. The methods forecast expected demand per period rather "
            "than the timing or size of the next transaction. Constant terminal paths ignore "
            "future covariates and changing positive sizes; automatic grid scores are not an "
            "inventory cost optimization, and approximate intervals are not calibrated for "
            "zero-inflated or discrete demand."
        ),
        citation_ready=(
            "Croston's intermittent-demand method separately smooths nonzero demand size and "
            "inter-demand timing (Croston, 1972). SBA applies the Syntetos–Boylan bias "
            "correction (Syntetos & Boylan, 2005), whereas TSB smooths occurrence probability "
            "each period so expected demand can decay after sustained zeros (Teunter et al., "
            "2011). Chrono Stream exposes all three recursions with explicit zero semantics."
        ),
        references=(CROSTON_1972, SYNTETOS_BOYLAN_2005, TEUNTER_ET_AL_2011),
    ),
    "linear": MethodInformation(
        origin=(
            "Linear trend projection is not a separately invented forecasting algorithm; "
            "it applies ordinary least squares to value versus an index of time. Legendre "
            "(1805) first published least squares and Gauss (1809) supplied an influential "
            "probabilistic development. Yule (1926) later demonstrated why regressions on "
            "serially dependent time series can appear convincing even when relationships are spurious."
        ),
        how_it_works=(
            "The model y_t = a*t + b chooses slope a and intercept b to minimize the sum of "
            "squared in-sample residuals. Forecasts evaluate the fitted line at future time indices."
        ),
        chrono_stream=(
            "Chrono Stream uses NumPy least squares through polyfit on observation indices "
            "1 through n. It fits each backtest model only on its training partition and "
            "refits on the full series for future projection."
        ),
        when_to_use=(
            "Use it as a deterministic benchmark when the level changes by an approximately "
            "constant amount per equally spaced observation."
        ),
        limitations=(
            "The time index is positional, so irregular spacing must be regularized first. "
            "A line ignores autocorrelation, seasonality, breaks, and stochastic trend; "
            "usual regression inference is not computed, and extrapolation assumes the same slope continues."
        ),
        citation_ready=(
            "Linear trend projection fits y_t = a t + b by least squares and evaluates the "
            "fitted line at future time indices. The method of least squares was first "
            "published by Legendre (1805) and was subsequently developed probabilistically "
            "by Gauss (1809). Applied to a time series, this is a deterministic curve "
            "projection rather than a complete stochastic time-series model; serial "
            "dependence, nonstationarity, and structural change can produce misleading trend "
            "evidence, as the classic warning by Yule (1926) illustrates."
        ),
        references=(LEGENDRE_1805, GAUSS_1809, YULE_1926),
    ),
    "quadratic": MethodInformation(
        origin=(
            "Quadratic trend projection combines least squares, first published by Legendre "
            "(1805), with polynomial interpolation/regression. Gergonne (1815–1816) gave an "
            "early explicit application of least squares to polynomial interpolation of sequences."
        ),
        how_it_works=(
            "The curve y_t = a*t^2 + b*t + c minimizes squared residuals. Its changing slope "
            "can represent one smooth acceleration or deceleration over the observed range."
        ),
        chrono_stream=(
            "The app uses NumPy polyfit with degree two on observation indices, requiring at "
            "least five values. The same leakage-free holdout/refit contract used by other models applies."
        ),
        when_to_use=(
            "Use it mainly as a benchmark when one smooth curvature is visually plausible "
            "and the forecast horizon is short relative to the observed history."
        ),
        limitations=(
            "A quadratic is unbounded outside the data and can reverse, explode, or become "
            "negative quickly. It ignores seasonality and residual dependence, and choosing "
            "a curved trend after viewing the entire evaluation period would leak information."
        ),
        citation_ready=(
            "Quadratic trend projection fits y_t = a t^2 + b t + c by minimizing squared "
            "residuals and extrapolates that polynomial beyond the sample. It combines the "
            "least-squares principle first published by Legendre (1805) with polynomial "
            "least-squares interpolation explicitly developed by Gergonne (1815–1816). It "
            "is a deterministic curve benchmark, not a stochastic forecasting process; "
            "because polynomial terms dominate outside the observed range, forecasts can "
            "become implausible rapidly, and time-series dependence can make apparent fit misleading (Yule, 1926)."
        ),
        references=(LEGENDRE_1805, GERGONNE_1815, YULE_1926),
    ),
    "exponential": MethodInformation(
        origin=(
            "Geometric growth was stated explicitly in Malthus's population model (1798), "
            "but exponential trend forecasting has no single originating paper. The fitted "
            "form here combines least squares (Legendre, 1805) with a log transformation. "
            "Duan (1983) and Miller (1984) established why simply exponentiating a fitted "
            "log mean is biased for the original-scale conditional mean."
        ),
        how_it_works=(
            "For positive data, log(y_t) = a*t + b + e_t is fitted by least squares, which "
            "implies multiplicative growth y_t = exp(b)*exp(a*t)*exp(e_t). Chrono Stream "
            "multiplies exp(a*t+b) by the nonparametric smearing factor mean(exp(e_t)) to "
            "target the original-scale conditional mean rather than the lognormal median."
        ),
        chrono_stream=(
            "The app rejects zero or negative values, fits the log-linear relation on each "
            "training partition, calculates Duan's smearing factor from its log residuals, "
            "and applies the same factor to fitted and future values. The factor is reported "
            "in model details."
        ),
        when_to_use=(
            "Use it as a benchmark for strictly positive data with a roughly constant "
            "percentage rate of growth or decay and approximately stable log-scale errors."
        ),
        limitations=(
            "A constant percentage rate rarely persists indefinitely, the model has no "
            "seasonal or autoregressive structure, and one global smearing factor assumes "
            "the residual distribution is stable across time and fitted level."
        ),
        citation_ready=(
            "Exponential trend projection assumes that the logarithm of a positive response "
            "changes linearly with time, log(y_t) = a t + b + e_t, so the original-scale "
            "trend changes at a constant percentage rate. Geometric growth has an early "
            "explicit domain formulation in Malthus (1798), while the fitted curve applies "
            "the least-squares principle published by Legendre (1805). Directly exponentiating "
            "the fitted log mean generally targets a median and can underestimate the "
            "original-scale conditional mean; Chrono Stream therefore applies Duan's (1983) "
            "nonparametric smearing estimate, addressing the retransformation-bias problem "
            "also analyzed by Miller (1984)."
        ),
        references=(MALTHUS_1798, LEGENDRE_1805, DUAN_1983, MILLER_1984),
    ),
    "logarithmic": MethodInformation(
        origin=(
            "The curve y = a*log(t) + b is not a uniquely invented time-series forecast "
            "method. It combines least-squares curve fitting (Legendre, 1805) with an imposed "
            "logarithmic response shape. Fechner (1860) is an early defining scientific use "
            "of an explicit logarithmic response law, but it concerns psychophysics rather "
            "than forecasting and is cited only for the curve form."
        ),
        how_it_works=(
            "The response is linear in log time: equal multiplicative changes in t imply "
            "equal additive changes in the fitted response. Its slope a/t decreases in "
            "magnitude, producing growth or decline that gradually flattens."
        ),
        chrono_stream=(
            "Chrono Stream fits value against log(1), ..., log(n) by NumPy least squares and "
            "evaluates the fitted curve at future observation indices. The shift to indices "
            "starting at one avoids log(0) but makes the chosen time origin part of the model."
        ),
        when_to_use=(
            "Use it as a descriptive benchmark for a series whose changes are large early "
            "and steadily diminish, with no recurring seasonal pattern."
        ),
        limitations=(
            "The result changes if the arbitrary time origin changes, it cannot represent an "
            "S-shaped saturation level, and it ignores autocorrelation, breaks, and seasonality. "
            "There is no defensible single 'inventor of logarithmic time forecasting.'"
        ),
        citation_ready=(
            "A logarithmic trend fits y_t = a log(t) + b by least squares, producing a rate "
            "of change a/t whose magnitude decreases as the time index grows. This is best "
            "described as a curve-projection benchmark rather than a uniquely invented "
            "time-series method: it combines the least-squares principle first published by "
            "Legendre (1805) with an imposed logarithmic functional form. Fechner (1860) "
            "provided an early explicit scientific logarithmic response law, although that "
            "work concerns psychophysics rather than forecasting. As with other deterministic "
            "time trends, serial dependence can make apparent fit misleading (Yule, 1926)."
        ),
        references=(LEGENDRE_1805, FECHNER_1860, YULE_1926),
    ),
}


def copy_ready_method_note(model_id: str, title: str) -> str:
    """Return a plain-text research note with in-text and APA references."""
    information = METHOD_INFORMATION[model_id]
    sections = [
        title,
        f"Overview\n{information.citation_ready}",
        f"Literature review\n{METHOD_LITERATURE_REVIEWS[model_id]}",
        f"How the method works\n{information.how_it_works}",
        f"How Chrono Stream implements it\n{information.chrono_stream}",
        f"Appropriate use\n{information.when_to_use}",
        f"Limitations\n{information.limitations}",
        "References (APA 7)\n"
        + "\n\n".join(reference.apa for reference in information.references),
    ]
    return "\n\n".join(sections)
