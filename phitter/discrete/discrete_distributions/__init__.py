from .bernoulli import Bernoulli
from .binomial import Binomial
from .discrete_laplace import DiscreteLaplace
from .geometric import Geometric
from .hypergeometric import Hypergeometric
from .logarithmic import Logarithmic
from .negative_binomial import NegativeBinomial
from .poisson import Poisson
from .skellam import Skellam
from .uniform import Uniform

DISCRETE_DISTRIBUTIONS = {
    "bernoulli": Bernoulli,
    "binomial": Binomial,
    "discrete_laplace": DiscreteLaplace,
    "geometric": Geometric,
    "hypergeometric": Hypergeometric,
    "logarithmic": Logarithmic,
    "negative_binomial": NegativeBinomial,
    "poisson": Poisson,
    "skellam": Skellam,
    "uniform": Uniform,
}
