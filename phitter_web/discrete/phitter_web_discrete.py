import concurrent.futures
import numpy
import scipy.optimize
import scipy.stats
import typing


class Bernoulli:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "bernoulli"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"p": 0.7006}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.bernoulli.cdf(x, self.p)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = (self.p**x) * (1 - self.p) ** (1 - x)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.bernoulli.ppf(u, self.p)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.p

    @property
    def variance(self) -> float:
        return self.p * (1 - self.p)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (1 - 2 * self.p) / numpy.sqrt(self.p * (1 - self.p))

    @property
    def kurtosis(self) -> float:
        return (6 * self.p * self.p - 6 * self.p + 1) / (self.p * (1 - self.p)) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 0 if self.p < 0.5 else 1

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = 0 < self.p < 1
        return v1

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        p = discrete_measures.mean
        parameters = {"p": p}
        return parameters


class Binomial:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.n = self.parameters["n"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "binomial"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"n": 10, "p": 0.6994}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.binom.cdf(x, self.n, self.p)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.binom.pmf(x, self.n, self.p)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.binom.ppf(u, self.n, self.p)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.n * self.p

    @property
    def variance(self) -> float:
        return self.n * self.p * (1 - self.p)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (1 - self.p - self.p) / numpy.sqrt(self.n * self.p * (1 - self.p))

    @property
    def kurtosis(self) -> float:
        return (1 - 6 * self.p * (1 - self.p)) / (self.n * self.p * (1 - self.p)) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.floor(self.p * (self.n + 1))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = 0 < self.p < 1
        v2 = self.n > 0
        v3 = type(self.n) == int
        return v1 and v2 and v3

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        p = 1 - discrete_measures.variance / discrete_measures.mean
        n = int(round(discrete_measures.mean / p, 0))
        parameters = {"n": n, "p": p}
        return parameters


class Geometric:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "geometric"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"p": 0.2973}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.geom.cdf(x, self.p)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.geom.pmf(x, self.p)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.geom.ppf(u, self.p)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return 1 / self.p

    @property
    def variance(self) -> float:
        return (1 - self.p) / (self.p * self.p)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 - self.p) / numpy.sqrt(1 - self.p)

    @property
    def kurtosis(self) -> float:
        return 3 + 6 + (self.p * self.p) / (1 - self.p)

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 1.0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = 0 < self.p < 1
        return v1

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        p = 1 / discrete_measures.mean
        parameters = {"p": p}
        return parameters


class Hypergeometric:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.N = self.parameters["N"]
        self.K = self.parameters["K"]
        self.n = self.parameters["n"]

    @property
    def name(self):
        return "hypergeometric"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"N": 120, "K": 66, "n": 27}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.hypergeom.cdf(x, self.N, self.n, self.K)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.hypergeom.pmf(x, self.N, self.n, self.K)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.hypergeom.ppf(u, self.N, self.n, self.K)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (self.n * self.K) / self.N

    @property
    def variance(self) -> float:
        return ((self.n * self.K) / self.N) * ((self.N - self.K) / self.N) * ((self.N - self.n) / (self.N - 1))

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return ((self.N - 2 * self.K) * numpy.sqrt(self.N - 1) * (self.N - 2 * self.n)) / (numpy.sqrt(self.n * self.K * (self.N - self.K) * (self.N - self.n)) * (self.N - 2))

    @property
    def kurtosis(self) -> float:
        return 3 + (1 / (self.n * self.K * (self.N - self.K) * (self.N - self.n) * (self.N - 2) * (self.N - 3))) * (
            (self.N - 1) * self.N * self.N * (self.N * (self.N + 1) - 6 * self.K * (self.N - self.K) - 6 * self.n * (self.N - self.n)) + 6 * self.n * self.K * (self.N - self.K) * (self.N - self.n) * (5 * self.N - 6)
        )

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.floor(((self.n + 1) * (self.K + 1)) / (self.N + 2))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.N > 0 and type(self.N) == int
        v2 = self.K > 0 and type(self.K) == int
        v3 = self.n > 0 and type(self.n) == int
        return v1 and v2 and v3

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], discrete_measures) -> tuple[float]:
            N, K, n = initial_solution
            parametric_mean = n * K / N
            parametric_variance = (n * K / N) * ((N - K) / N) * ((N - n) / (N - 1))
            parametric_mode = numpy.floor((n + 1) * (K + 1) / (N + 2))
            eq1 = parametric_mean - discrete_measures.mean
            eq2 = parametric_variance - discrete_measures.variance
            eq3 = parametric_mode - discrete_measures.mode
            return (eq1, eq2, eq3)

        bounds = ((discrete_measures.max, discrete_measures.max, 1), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (discrete_measures.max * 5, discrete_measures.max * 3, discrete_measures.max)
        args = [discrete_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"N": round(solution.x[0]), "K": round(solution.x[1]), "n": round(solution.x[2])}
        return parameters


class Logarithmic:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "logarithmic"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"p": 0.81}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.logser.cdf(x, self.p)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.logser.pmf(x, self.p)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.logser.ppf(u, self.p)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return -self.p / ((1 - self.p) * numpy.log(1 - self.p))

    @property
    def variance(self) -> float:
        return (-self.p * (self.p + numpy.log(1 - self.p))) / ((1 - self.p) ** 2 * numpy.log(1 - self.p) ** 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (-(2 * self.p**2 + 3 * self.p * numpy.log(1 - self.p) + (1 + self.p) * numpy.log(1 - self.p) ** 2) / (numpy.log(1 - self.p) * (self.p + numpy.log(1 - self.p)) * numpy.sqrt(-self.p * (self.p + numpy.log(1 - self.p))))) * numpy.log(1 - self.p)

    @property
    def kurtosis(self) -> float:
        return -(6 * self.p**3 + 12 * self.p**2 * numpy.log(1 - self.p) + self.p * (4 * self.p + 7) * numpy.log(1 - self.p) ** 2 + (self.p**2 + 4 * self.p + 1) * numpy.log(1 - self.p) ** 3) / (self.p * (self.p + numpy.log(1 - self.p)) ** 2)

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 1

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = 0 < self.p < 1
        return v1

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], discrete_measures) -> tuple[float]:
            p = initial_solution
            parametric_mean = -p / ((1 - p) * numpy.log(1 - p))
            eq1 = parametric_mean - discrete_measures.mean
            return eq1

        solution = scipy.optimize.least_squares(equations, 0.5, bounds=(0, 1), args=([discrete_measures]))
        parameters = {"p": solution.x[0]}
        return parameters


class NegativeBinomial:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.r = self.parameters["r"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "negative_binomial"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"r": 96, "p": 0.6893}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.beta.cdf(self.p, self.r, x + 1)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nbinom.pmf(x, self.r, self.p)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nbinom.ppf(u, self.r, self.p)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (self.r * (1 - self.p)) / self.p

    @property
    def variance(self) -> float:
        return (self.r * (1 - self.p)) / (self.p * self.p)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 - self.p) / numpy.sqrt(self.r * (1 - self.p))

    @property
    def kurtosis(self) -> float:
        return 6 / self.r + (self.p * self.p) / (self.r * (1 - self.p)) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.floor(((self.r - 1) * (1 - self.p)) / self.p)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0 and self.p < 1
        v2 = self.r > 0
        v3 = type(self.r) == int
        return v1 and v2 and v3

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        p = discrete_measures.mean / discrete_measures.variance
        r = round(discrete_measures.mean * p / (1 - p))
        parameters = {"r": r, "p": p}
        return parameters


class Poisson:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.lambda_ = self.parameters["lambda"]

    @property
    def name(self):
        return "poisson"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 4.969}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.poisson.cdf(x, self.lambda_)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.poisson.pmf(x, self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.poisson.ppf(u, self.lambda_)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return self.lambda_

    @property
    def variance(self) -> float:
        return self.lambda_

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return self.lambda_**-0.5

    @property
    def kurtosis(self) -> float:
        return 1 / self.lambda_ + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.floor(self.lambda_)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        lambda_ = discrete_measures.mean
        parameters = {"lambda": lambda_}
        return parameters


class Uniform:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "uniform"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 3, "b": 10}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        return (x - self.a + 1) / (self.b - self.a + 1)

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        if type(x) == int:
            return 1 / (self.b - self.a + 1)
        return numpy.full(len(x), 1 / (self.b - self.a + 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.ceil(u * (self.b - self.a + 1) + self.a - 1)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        return None

    @property
    def mean(self) -> float:
        return (self.a + self.b) / 2

    @property
    def variance(self) -> float:
        return ((self.b - self.a + 1) * (self.b - self.a + 1) - 1) / 12

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return ((-6 / 5) * ((self.b - self.a + 1) * (self.b - self.a + 1) + 1)) / ((self.b - self.a + 1) * (self.b - self.a + 1) - 1) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return None

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        v2 = type(self.b) == int
        v3 = type(self.a) == int
        return v1 and v2 and v3

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        a = round(discrete_measures.min)
        b = round(discrete_measures.max)
        parameters = {"a": a, "b": b}
        return parameters


DISCRETE_DISTRIBUTIONS = {
    "bernoulli": Bernoulli,
    "binomial": Binomial,
    "geometric": Geometric,
    "hypergeometric": Hypergeometric,
    "logarithmic": Logarithmic,
    "negative_binomial": NegativeBinomial,
    "poisson": Poisson,
    "uniform": Uniform,
}


class DiscreteMeasures:
    def __init__(
        self,
        data: list[int],
        confidence_level: float = 0.95,
        subsample_size: int | None = None,
        subsample_estimation_size: int | None = None,
    ):
        self.data = numpy.sort(data) if subsample_size == None else numpy.sort(numpy.random.choice(data, size=subsample_size, replace=False))
        self.size = self.data.size
        self.data_to_fit = self.data if subsample_estimation_size == None else numpy.random.choice(self.data, size=min(self.size, subsample_estimation_size), replace=False)
        self.min = self.data[0]
        self.max = self.data[-1]
        self.mean = numpy.mean(data)
        self.variance = numpy.var(data, ddof=1)
        self.std = numpy.std(data, ddof=1)
        self.skewness = scipy.stats.moment(data, 3) / pow(self.std, 3)
        self.kurtosis = scipy.stats.moment(data, 4) / pow(self.std, 4)
        self.median = int(numpy.median(self.data))
        self.mode = int(scipy.stats.mode(data, keepdims=True)[0][0])
        self.domain = numpy.arange(self.min, self.max + 1)
        self.absolutes_frequencies, _ = numpy.histogram(self.data, bins=numpy.arange(self.min, self.max + 2) - 0.5, density=False)
        self.densities_frequencies, _ = numpy.histogram(self.data, bins=numpy.arange(self.min, self.max + 2) - 0.5, density=True)
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.size - 1]])
        self.Sn_ks_before = (numpy.arange(self.size)) / self.size
        self.Sn_ks_after = (numpy.arange(self.size) + 1) / self.size
        self.confidence_level = confidence_level
        self.critical_value_ks = scipy.stats.kstwo.ppf(self.confidence_level, self.size)
        self.ecdf_frequencies = numpy.cumsum(self.densities_frequencies)
        self.qq_arr = (numpy.arange(1, self.size + 1) - 0.5) / self.size

    def __str__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def critical_value_chi2(self, freedom_degrees):
        return scipy.stats.chi2.ppf(self.confidence_level, freedom_degrees)


def evaluate_discrete_test_chi_square(distribution, discrete_measures):
    N = discrete_measures.size
    freedom_degrees = max(len(discrete_measures.domain) - 1 - distribution.num_parameters, 1)
    expected_values = numpy.ceil(N * (distribution.pmf(discrete_measures.domain)))
    errors = ((discrete_measures.absolutes_frequencies - expected_values) ** 2) / expected_values
    statistic_chi2 = numpy.sum(errors)
    critical_value = discrete_measures.critical_value_chi2(freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {
        "test_statistic": statistic_chi2,
        "critical_value": critical_value,
        "p-value": p_value,
        "rejected": rejected,
    }
    return result_test_chi2


def evaluate_discrete_test_kolmogorov_smirnov(distribution, discrete_measures):
    Fn = distribution.cdf(discrete_measures.data)
    errors_before = numpy.abs(discrete_measures.Sn_ks_before[discrete_measures.idx_ks] - Fn[discrete_measures.idx_ks])
    errors_after = numpy.abs(discrete_measures.Sn_ks_after[discrete_measures.idx_ks] - Fn[discrete_measures.idx_ks])
    statistic_ks = max(numpy.max(errors_before), numpy.max(errors_after))
    critical_value = discrete_measures.critical_value_ks
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, discrete_measures.size)
    rejected = statistic_ks >= critical_value
    result_test_ks = {"test_statistic": statistic_ks, "critical_value": critical_value, "p-value": p_value, "rejected": rejected}
    return result_test_ks


class PhitterDiscrete:
    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        confidence_level=0.95,
        minimum_sse=numpy.inf,
        subsample_size: int | None = None,
        subsample_estimation_size: int | None = None,
        distributions_to_fit: list[str] | typing.Literal["all"] = "all",
        exclude_distributions: list[str] | typing.Literal["any"] = "any",
    ):
        if distributions_to_fit != "all" and exclude_distributions != "any":
            raise ValueError("Specify either 'distributions_to_fit' or 'exclude_distributions', but not both.")
        if distributions_to_fit == "all" and exclude_distributions == "any":
            self.distributions_to_fit = list(DISCRETE_DISTRIBUTIONS.keys())
        if distributions_to_fit != "all" and exclude_distributions == "any":
            not_distributions_ids = [id_distribution for id_distribution in distributions_to_fit if id_distribution not in DISCRETE_DISTRIBUTIONS.keys()]
            if len(not_distributions_ids) > 0:
                raise ValueError(f"The following distributions are not found in the discrete distributions list: {not_distributions_ids}")
            self.distributions_to_fit = distributions_to_fit
        if distributions_to_fit == "all" and exclude_distributions != "any":
            not_distributions_ids = [id_distribution for id_distribution in exclude_distributions if id_distribution not in DISCRETE_DISTRIBUTIONS.keys()]
            if len(not_distributions_ids) > 0:
                raise ValueError(f"The following distributions to exclude are not found in the discrete distributions list: {not_distributions_ids}")
            self.distributions_to_fit = [dist for dist in DISCRETE_DISTRIBUTIONS.keys() if dist not in exclude_distributions]
        self.discrete_measures = DiscreteMeasures(
            data=data,
            confidence_level=confidence_level,
            subsample_size=subsample_size,
            subsample_estimation_size=subsample_estimation_size,
        )
        self.minimum_sse = minimum_sse
        self.distribution_results = {}
        self.none_results = {"test_statistic": None, "critical_value": None, "p_value": None, "rejected": None}
        self.sorted_distributions = None
        self.not_rejected_distributions = None
        self.distribution_instances = None

    def log_likelihood(self, distribution, eps=1e-300) -> float:
        x = self.discrete_measures.data
        p = distribution.pmf(x)
        bad = ~numpy.isfinite(p) | (p <= 0)
        if bad.mean() > 0.01:
            domain = self.discrete_measures.domain
            counts = self.discrete_measures.absolutes_frequencies
            probs = distribution.pmf(domain)
            probs = numpy.clip(numpy.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0), eps, 1.0)
            return float(numpy.sum(counts * numpy.log(probs)))
        p = numpy.clip(numpy.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), eps, None)
        return float(numpy.sum(numpy.log(p)))

    def get_aic_and_bic(self, distribution) -> tuple[float, float]:
        ll = self.log_likelihood(distribution)
        k = distribution.num_parameters
        n = self.discrete_measures.size
        aic = 2.0 * k - 2.0 * ll
        bic = k * numpy.log(n) - 2.0 * ll
        return aic, bic

    def test(self, test_function, label: str, distribution):
        validation_test = False
        try:
            test = test_function(distribution, self.discrete_measures)
            if numpy.isnan(test["test_statistic"]) == False and numpy.isinf(test["test_statistic"]) == False and test["test_statistic"] >= 0:
                self.distribution_results[label] = {
                    "test_statistic": test["test_statistic"],
                    "critical_value": test["critical_value"],
                    "p_value": test["p-value"],
                    "rejected": test["rejected"],
                }
                validation_test = True
            else:
                self.distribution_results[label] = self.none_results
        except:
            self.distribution_results[label] = self.none_results
        return validation_test

    def process_distribution(self, id_distribution: str) -> tuple[str, dict, typing.Any] | None:
        distribution_class = DISCRETE_DISTRIBUTIONS[id_distribution]
        validate_estimation = True
        sse = 0
        try:
            distribution = distribution_class(discrete_measures=self.discrete_measures)
            pmf_values = distribution.pmf(self.discrete_measures.domain)
            sse = numpy.sum(numpy.power(pmf_values - self.discrete_measures.densities_frequencies, 2))
        except:
            validate_estimation = False
        self.distribution_results = {}
        if validate_estimation and distribution.parameter_restrictions() and not numpy.isnan(sse) and not numpy.isinf(sse) and sse < self.minimum_sse:
            v1 = self.test(evaluate_discrete_test_chi_square, "chi_square", distribution)
            v2 = self.test(evaluate_discrete_test_kolmogorov_smirnov, "kolmogorov_smirnov", distribution)
            if v1 or v2:
                aic, bic = self.get_aic_and_bic(distribution)
                self.distribution_results["aic"] = aic
                self.distribution_results["bic"] = bic
                self.distribution_results["sse"] = sse
                self.distribution_results["parameters"] = distribution.parameters
                self.distribution_results["n_test_passed"] = int(self.distribution_results["chi_square"]["rejected"] == False) + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == False)
                self.distribution_results["n_test_null"] = int(self.distribution_results["chi_square"]["rejected"] == None) + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == None)
                return id_distribution, self.distribution_results, distribution
        return None

    def fit(self, n_workers: int = 1):
        if n_workers <= 0:
            raise Exception("n_workers must be greater than 1")
        if n_workers == 1:
            processing_results = [self.process_distribution(id_distribution) for id_distribution in self.distributions_to_fit]
        else:
            executor = concurrent.futures.ProcessPoolExecutor(max_workers=n_workers)
            processing_results = list(executor.map(self.process_distribution, self.distributions_to_fit))
        processing_results = [r for r in processing_results if r is not None]
        self.sorted_distributions = {distribution: results for distribution, results, _ in sorted(processing_results, key=lambda x: (-x[1]["n_test_passed"], x[1]["sse"]))}
        self.not_rejected_distributions = {distribution: results for distribution, results in self.sorted_distributions.items() if results["n_test_passed"] > 0}
        self.distribution_instances = {distribution: instance for distribution, _, instance in processing_results}

    def parse_rgba_color(self, rgba_string):
        rgba = re.match(r"rgba\((\d+),(\d+),(\d+),(\d*(?:\.\d+)?)\)", rgba_string)
        r, g, b, a = map(float, rgba.groups())
        return (r / 255, g / 255, b / 255, a)

    def plot_histogram_plotly(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        domain = self.discrete_measures.domain
        densities_frequencies = self.discrete_measures.densities_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=domain, y=densities_frequencies, marker_color=plot_bar_color, name="Data", showlegend=True))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=plot_title,
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=11),
            bargap=plot_bargap,
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_histogram_matplotlib(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
    ):
        plt.style.use("ggplot")
        domain = self.discrete_measures.domain
        densities_frequencies = self.discrete_measures.densities_frequencies
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.bar(domain, densities_frequencies, label="Data", color=self.parse_rgba_color(plot_bar_color))
        plt.title(plot_title)
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def plot_histogram_distributions_pmf_plotly(
        self,
        n_distributions: int,
        n_distributions_visible: int,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        domain = self.discrete_measures.domain
        densities_frequencies = self.discrete_measures.densities_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=domain, y=densities_frequencies, marker_color=plot_bar_color, name="Data", showlegend=False))
        for idx, (id_distribution, result) in enumerate(list(self.sorted_distributions.items())[:n_distributions]):
            y_plot = self.distribution_instances[id_distribution].pmf(domain)
            distribution_sse = result["sse"]
            is_visible = True if idx + 1 <= n_distributions_visible else "legendonly"
            is_rejected = "✅" if id_distribution in self.not_rejected_distributions else ""
            scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
            scatter_line = dict(color=px.colors.qualitative.G10[idx], width=2) if idx < len(px.colors.qualitative.G10) else dict(width=2)
            try:
                fig.add_trace(go.Scatter(x=domain, y=y_plot, mode="lines+markers", visible=is_visible, name=scatter_name, line=scatter_line))
            except Exception:
                fig.add_trace(go.Scatter(x=domain, y=numpy.zeros(len(domain)), mode="lines+markers", visible=is_visible, name=scatter_name, line=scatter_line))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} - PMF DISTRIBUTIONS",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10)),
            bargap=plot_bargap,
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_histogram_distributions_pmf_matplotlib(
        self,
        n_distributions: int,
        n_distributions_visible: int,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
    ):
        plt.style.use("ggplot")
        domain = self.discrete_measures.domain
        densities_frequencies = self.discrete_measures.densities_frequencies
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.bar(domain, densities_frequencies, color=self.parse_rgba_color(plot_bar_color))
        for idx, (id_distribution, result) in enumerate(list(self.sorted_distributions.items())[:n_distributions]):
            y_plot = self.distribution_instances[id_distribution].pmf(domain)
            distribution_sse = result["sse"]
            is_rejected = "✓" if id_distribution in self.not_rejected_distributions else ""
            scatter_name = f"{idx+1:02d}. {id_distribution}: {distribution_sse:.4E}{is_rejected}"
            try:
                plt.plot(domain, y_plot, label=scatter_name, color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)), marker=".")
            except Exception:
                plt.plot(domain, numpy.zeros(len(domain)), label=scatter_name, color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)), marker=".")
        plt.title(f"{plot_title} - PMF DISTRIBUTIONS")
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def plot_distribution_pmf_plotly(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
        plot_line_color: str,
        plot_line_width: int,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        domain = self.discrete_measures.domain
        densities_frequencies = self.discrete_measures.densities_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=domain, y=densities_frequencies, marker_color=plot_bar_color, name="Data", showlegend=True))
        y_plot = self.distribution_instances[id_distribution].pmf(domain)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✅" if id_distribution in self.not_rejected_distributions else ""
        scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
        scatter_line = dict(color=plot_line_color, width=plot_line_width)
        try:
            fig.add_trace(go.Scatter(x=domain, y=y_plot, mode="lines+markers", name=scatter_name, line=scatter_line))
        except Exception:
            fig.add_trace(go.Scatter(x=domain, y=numpy.zeros(len(domain)), mode="lines+markers", name=scatter_name, line=scatter_line))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} - PMF {id_distribution.upper().replace('_', ' ')} DISTRIBUTION",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10)),
            bargap=plot_bargap,
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_distribution_pmf_matplotlib(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
        plot_line_color: str,
        plot_line_width: int,
    ):
        plt.style.use("ggplot")
        domain = self.discrete_measures.domain
        densities_frequencies = self.discrete_measures.densities_frequencies
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.bar(domain, densities_frequencies, label="Data", color=self.parse_rgba_color(plot_bar_color))
        y_plot = self.distribution_instances[id_distribution].pmf(domain)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✓" if id_distribution in self.not_rejected_distributions else ""
        scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
        try:
            plt.plot(domain, y_plot, label=scatter_name, color=self.parse_rgba_color(plot_line_color), linewidth=plot_line_width, marker="o")
        except Exception:
            plt.plot(domain, numpy.zeros(len(domain)), label=scatter_name, color=self.parse_rgba_color(plot_line_color), linewidth=plot_line_width, marker="o")
        plt.title(f"{plot_title} - PMF {id_distribution.upper().replace('_', ' ')} DISTRIBUTION")
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def plot_ecdf_plotly(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
        plot_bargap: float,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        domain = self.discrete_measures.domain
        ecdf_frequencies = self.discrete_measures.ecdf_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=domain, y=ecdf_frequencies, marker_color=plot_bar_color, name="Empirical Distribution", showlegend=True))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=plot_title,
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10)),
            bargap=plot_bargap,
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_ecdf_matplotlib(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_bar_color: str,
    ):
        plt.style.use("ggplot")
        plt.style.use("ggplot")
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.hist(
            self.discrete_measures.data,
            density=True,
            cumulative=True,
            label="Data",
            ec="white",
            color=self.parse_rgba_color(plot_bar_color),
        )
        plt.title(plot_title)
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def plot_ecdf_distribution_plotly(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_empirical_bar_color: str,
        plot_empirical_bargap: float,
        plot_distribution_line_color: str,
        plot_distribution_line_width: int,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        domain = self.discrete_measures.domain
        ecdf_frequencies = self.discrete_measures.ecdf_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=domain, y=ecdf_frequencies, marker_color=plot_empirical_bar_color, name="Empirical Distribution", showlegend=True))
        y_plot = self.distribution_instances[id_distribution].cdf(domain)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✅" if id_distribution in self.not_rejected_distributions else ""
        try:
            fig.add_trace(
                go.Scatter(
                    x=domain,
                    y=y_plot,
                    mode="lines+markers",
                    name=f"{id_distribution}: {distribution_sse:.4E}{is_rejected}",
                    line=dict(color=plot_distribution_line_color, width=plot_distribution_line_width),
                )
            )
        except Exception:
            fig.add_trace(go.Scatter(x=domain, y=numpy.zeros(len(domain)), mode="lines+markers", name=f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} - CDF {id_distribution.upper().replace('_', ' ')} DISTRIBUTION",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10)),
            bargap=plot_empirical_bargap,
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_ecdf_distribution_matplotlib(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_empirical_bar_color: str,
        plot_empirical_bargap: float,
        plot_distribution_line_color: str,
        plot_distribution_line_width: int,
    ):
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        plt.style.use("ggplot")
        domain = self.discrete_measures.domain
        ecdf_frequencies = self.discrete_measures.ecdf_frequencies
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.bar(domain, ecdf_frequencies, label="Data", color=self.parse_rgba_color(plot_empirical_bar_color))
        domain = self.discrete_measures.domain
        y_plot = self.distribution_instances[id_distribution].cdf(domain)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✓" if id_distribution in self.not_rejected_distributions else ""
        scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
        try:
            plt.plot(domain, y_plot, label=scatter_name, color=self.parse_rgba_color(plot_distribution_line_color), linewidth=plot_distribution_line_width, marker=".")
        except Exception:
            plt.plot(domain, numpy.zeros(len(domain)), label=scatter_name, color=self.parse_rgba_color(plot_distribution_line_color), linewidth=plot_distribution_line_width)
        plt.title(plot_title)
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def qq_plot_plotly(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        qq_marker_name: str,
        qq_marker_color: str,
        qq_marker_size: int,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        x = self.distribution_instances[id_distribution].ppf(self.discrete_measures.qq_arr)
        y = self.discrete_measures.data
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=qq_marker_name, marker=dict(color=qq_marker_color, size=qq_marker_size), showlegend=True))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} {id_distribution.upper().replace('_', ' ')} DISTRIBUTION",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10)),
        )
        fig.show(renderer=plotly_plot_renderer)

    def qq_plot_matplotlib(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        qq_marker_name: str,
        qq_marker_color: str,
        qq_marker_size: int,
    ):
        plt.style.use("ggplot")
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not found")
        x = self.distribution_instances[id_distribution].ppf(self.discrete_measures.qq_arr)
        y = self.discrete_measures.data
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.scatter(x, y, label=qq_marker_name, color=self.parse_rgba_color(qq_marker_color), s=qq_marker_size)
        plt.title(f"{plot_title} {id_distribution.upper().replace('_', ' ')} DISTRIBUTION")
        plt.xlabel(plot_xaxis_title)
        plt.ylabel(plot_yaxis_title)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def qq_plot_regression_plotly(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        qq_marker_name: str,
        qq_marker_color: str,
        qq_marker_size: int,
        regression_line_name: str,
        regression_line_color: str,
        regression_line_width: int,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        x = self.distribution_instances[id_distribution].ppf(self.discrete_measures.qq_arr)
        y = self.discrete_measures.data
        linear_regression = scipy.stats.linregress(x, y)
        y_reg = linear_regression.intercept + x * linear_regression.slope
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y_reg, mode="lines", name=regression_line_name, line=dict(color=regression_line_color, width=regression_line_width)))
        fig.add_trace(go.Scatter(x=x, y=y, mode="markers", name=qq_marker_name, marker=dict(color=qq_marker_color, size=qq_marker_size)))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} {id_distribution.upper().replace('_', ' ')} DISTRIBUTION <br><br><sup>Regression: {linear_regression.intercept:.4f} + x * {linear_regression.slope:.4f} • r = {linear_regression.rvalue:.4f}</sup>",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10)),
        )
        fig.show(renderer=plotly_plot_renderer)

    def qq_plot_regression_matplotlib(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        qq_marker_name: str,
        qq_marker_color: str,
        qq_marker_size: int,
        regression_line_name: str,
        regression_line_color: str,
        regression_line_width: int,
    ):
        plt.style.use("ggplot")
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not found")
        x = self.distribution_instances[id_distribution].ppf(self.discrete_measures.qq_arr)
        y = self.discrete_measures.data
        linear_regression = scipy.stats.linregress(x, y)
        y_reg = linear_regression.intercept + x * linear_regression.slope
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.plot(x, y_reg, label=regression_line_name, color=self.parse_rgba_color(regression_line_color), linewidth=regression_line_width)
        plt.scatter(x, y, label=qq_marker_name, color=self.parse_rgba_color(qq_marker_color), s=qq_marker_size)
        plt.title(f"{plot_title} {id_distribution.upper().replace('_', ' ')} DISTRIBUTION\nRegression: {linear_regression.intercept:.4f} + x * {linear_regression.slope:.4f} • r = {linear_regression.rvalue:.4f}")
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()
