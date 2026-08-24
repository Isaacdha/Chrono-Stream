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
            "Chrono Stream fits statsmodels SimpleExpSmoothing with an estimated initial "
            "level. Alpha can be optimized from the training data or supplied manually."
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
            "(1956) introduced the approach for demand forecasting, Holt (1957/2004) "
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
            "The app fits additive-trend statsmodels ExponentialSmoothing with estimated "
            "initial states. Alpha and beta can be optimized or manual; damping is "
            "optional, with phi optimized automatically or supplied in manual mode."
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
        references=(HOLT_1957, GARDNER_MCKENZIE_1985, WINTERS_1960),
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
            "trend damping, estimated initial states, and automatic or manual alpha, beta, "
            "and gamma. It requires at least two complete cycles; multiplicative components "
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
    "x11": MethodInformation(
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
            "last max(2m, 8) trend observations, averages the historical STL seasonal "
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
