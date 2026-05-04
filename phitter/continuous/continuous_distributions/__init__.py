from .alpha import Alpha
from .arcsine import Arcsine
from .argus import Argus
from .bates import Bates
from .beta import Beta
from .beta_prime import BetaPrime
from .beta_prime_4p import BetaPrime4P
from .bradford import Bradford
from .burr import Burr
from .burr_4p import Burr4P
from .cauchy import Cauchy
from .chi_square import ChiSquare
from .chi_square_3p import ChiSquare3P
from .dagum import Dagum
from .dagum_4p import Dagum4P
from .erlang import Erlang
from .erlang_3p import Erlang3P
from .error_function import ErrorFunction
from .exponential import Exponential
from .exponential_2p import Exponential2P
from .exponentiated_kumaraswamy import ExponentiatedKumaraswamy
from .f import F
from .f_4p import F4P
from .fatigue_life import FatigueLife
from .folded_normal import FoldedNormal
from .frechet import Frechet
from .gamma import Gamma
from .gamma_3p import Gamma3P
from .generalized_extreme_value import GeneralizedExtremeValue
from .generalized_gamma import GeneralizedGamma
from .generalized_gamma_4p import GeneralizedGamma4P
from .generalized_logistic import GeneralizedLogistic
from .generalized_normal import GeneralizedNormal
from .generalized_pareto import GeneralizedPareto
from .gibrat import Gibrat
from .gumbel_left import GumbelLeft
from .gumbel_right import GumbelRight
from .half_normal import HalfNormal
from .hyperbolic_secant import HyperbolicSecant
from .inverse_gamma import InverseGamma
from .inverse_gamma_3p import InverseGamma3P
from .inverse_gaussian import InverseGaussian
from .inverse_gaussian_3p import InverseGaussian3P
from .johnson_sb import JohnsonSB
from .johnson_su import JohnsonSU
from .kumaraswamy import Kumaraswamy
from .laplace import Laplace
from .levy import Levy
from .loggamma import LogGamma
from .logistic import Logistic
from .loglogistic import LogLogistic
from .loglogistic_3p import LogLogistic3P
from .lognormal import LogNormal
from .maxwell import Maxwell
from .mcdonald import McDonald
from .moyal import Moyal
from .nakagami import Nakagami
from .non_central_chi_square import NonCentralChiSquare
from .non_central_f import NonCentralF
from .non_central_t_student import NonCentralTStudent
from .normal import Normal
from .pareto_first_kind import ParetoFirstKind
from .pareto_second_kind import ParetoSecondKind
from .pert import Pert
from .power_function import PowerFunction
from .rayleigh import Rayleigh
from .reciprocal import Reciprocal
from .rice import Rice
from .semicircular import Semicircular
from .topp_leone import ToppLeone
from .t_student import TStudent
from .t_student_3p import TStudent3P
from .trapezoidal import Trapezoidal
from .triangular import Triangular
from .uniform import Uniform
from .weibull import Weibull
from .weibull_3p import Weibull3P

CONTINUOUS_DISTRIBUTIONS = {
    "alpha": Alpha,
    "arcsine": Arcsine,
    "argus": Argus,
    "bates": Bates,
    "beta": Beta,
    "beta_prime": BetaPrime,
    "beta_prime_4p": BetaPrime4P,
    "bradford": Bradford,
    "burr": Burr,
    "burr_4p": Burr4P,
    "cauchy": Cauchy,
    "chi_square": ChiSquare,
    "chi_square_3p": ChiSquare3P,
    "dagum": Dagum,
    "dagum_4p": Dagum4P,
    "erlang": Erlang,
    "erlang_3p": Erlang3P,
    "error_function": ErrorFunction,
    "exponential": Exponential,
    "exponential_2p": Exponential2P,
    "exponentiated_kumaraswamy": ExponentiatedKumaraswamy,
    "f": F,
    "fatigue_life": FatigueLife,
    "folded_normal": FoldedNormal,
    "frechet": Frechet,
    "f_4p": F4P,
    "gamma": Gamma,
    "gamma_3p": Gamma3P,
    "generalized_extreme_value": GeneralizedExtremeValue,
    "generalized_gamma": GeneralizedGamma,
    "generalized_gamma_4p": GeneralizedGamma4P,
    "generalized_logistic": GeneralizedLogistic,
    "generalized_normal": GeneralizedNormal,
    "generalized_pareto": GeneralizedPareto,
    "gibrat": Gibrat,
    "gumbel_left": GumbelLeft,
    "gumbel_right": GumbelRight,
    "half_normal": HalfNormal,
    "hyperbolic_secant": HyperbolicSecant,
    "inverse_gamma": InverseGamma,
    "inverse_gamma_3p": InverseGamma3P,
    "inverse_gaussian": InverseGaussian,
    "inverse_gaussian_3p": InverseGaussian3P,
    "johnson_sb": JohnsonSB,
    "johnson_su": JohnsonSU,
    "kumaraswamy": Kumaraswamy,
    "laplace": Laplace,
    "levy": Levy,
    "loggamma": LogGamma,
    "logistic": Logistic,
    "loglogistic": LogLogistic,
    "loglogistic_3p": LogLogistic3P,
    "lognormal": LogNormal,
    "maxwell": Maxwell,
    "mcdonald": McDonald,
    "moyal": Moyal,
    "nakagami": Nakagami,
    "non_central_chi_square": NonCentralChiSquare,
    "non_central_f": NonCentralF,
    "non_central_t_student": NonCentralTStudent,
    "normal": Normal,
    "pareto_first_kind": ParetoFirstKind,
    "pareto_second_kind": ParetoSecondKind,
    "pert": Pert,
    "power_function": PowerFunction,
    "rayleigh": Rayleigh,
    "reciprocal": Reciprocal,
    "rice": Rice,
    "semicircular": Semicircular,
    "topp_leone": ToppLeone,
    "trapezoidal": Trapezoidal,
    "triangular": Triangular,
    "t_student": TStudent,
    "t_student_3p": TStudent3P,
    "uniform": Uniform,
    "weibull": Weibull,
    "weibull_3p": Weibull3P,
}
