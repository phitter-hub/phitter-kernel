import concurrent.futures
import numpy
import scipy.integrate
import scipy.optimize
import scipy.special
import scipy.stats
import typing
import warnings

warnings.filterwarnings("ignore")


class Alpha:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "alpha"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 5, "loc": 9, "scale": 57}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.stats.norm.cdf(self.alpha - (1 / z(x))) / scipy.stats.norm.cdf(self.alpha)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.alpha.pdf(x, self.alpha, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.scale / (self.alpha - scipy.stats.norm.ppf(u * scipy.stats.norm.cdf(self.alpha)))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        f = lambda x: x**k * scipy.stats.alpha.pdf(x, self.alpha, loc=0, scale=1)
        return scipy.integrate.quad(f, 0, 2)[0]

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale * self.scale * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / (µ2 - µ1**2) ** 1.5

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / (µ2 - µ1**2) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (self.scale * (numpy.sqrt(self.alpha * self.alpha + 8) - self.alpha)) / 4 + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.alpha.fit(continuous_measures.data_to_fit)
        parameters = {"alpha": scipy_parameters[0], "loc": scipy_parameters[1], "scale": scipy_parameters[2]}
        return parameters


class Arcsine:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "arcsine"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 77, "b": 89}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.a) / (self.b - self.a)
        return 2 * numpy.arcsin(numpy.sqrt(z(x))) / numpy.pi

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 / (numpy.pi * numpy.sqrt((x - self.a) * (self.b - x)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a + (self.b - self.a) * numpy.sin((u * numpy.pi) / 2) ** 2
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (scipy.special.gamma(0.5) * scipy.special.gamma(k + 0.5)) / (numpy.pi * scipy.special.gamma(k + 1))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1 * (self.b - self.a) + self.a

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (µ2 - µ1**2) * (self.b - self.a) ** 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

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
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-3
        b = continuous_measures.max + 1e-3
        parameters = {"a": a, "b": b}
        return parameters


class Argus:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.chi = self.parameters["chi"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "argus"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"chi": 3, "loc": 102, "scale": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.argus.cdf(x, self.chi, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.argus.pdf(x, self.chi, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        y1 = (1 - u) * scipy.special.gammainc(1.5, (self.chi * self.chi) / 2)
        y2 = (2 * scipy.special.gammaincinv(1.5, y1)) / (self.chi * self.chi)
        result = self.loc + self.scale * numpy.sqrt(1 - y2)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        f = lambda x: x**k * self.pdf(x)
        return scipy.integrate.quad(f, self.loc, self.loc + self.scale)[0]

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        return self.loc + self.scale * numpy.sqrt(numpy.pi / 8) * ((self.chi * numpy.exp((-self.chi * self.chi) / 4) * scipy.special.iv(1, (self.chi * self.chi) / 4)) / Ψ(self.chi))

    @property
    def variance(self) -> float:
        Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
        return self.scale * self.scale * (1 - 3 / (self.chi * self.chi) + (self.chi * scipy.stats.norm.pdf(self.chi)) / Ψ(self.chi)) - (self.mean - self.loc) ** 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / (µ2 - µ1**2) ** 1.5

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / (µ2 - µ1**2) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.scale * (1 / (numpy.sqrt(2) * self.chi)) * numpy.sqrt(self.chi * self.chi - 2 + numpy.sqrt(self.chi * self.chi * self.chi * self.chi + 4))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.chi > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            chi, loc, scale = initial_solution
            Ψ = lambda t: scipy.stats.norm.cdf(t) - t * scipy.stats.norm.pdf(t) - 0.5
            parametric_mean = loc + scale * numpy.sqrt(numpy.pi / 8) * ((chi * numpy.exp((-chi * chi) / 4) * scipy.special.iv(1, (chi * chi) / 4)) / Ψ(chi))
            parametric_variance = scale * scale * (1 - 3 / (chi * chi) + (chi * scipy.stats.norm.pdf(chi)) / Ψ(chi)) - (parametric_mean - loc) ** 2
            parametric_median = loc + scale * numpy.sqrt(1 - (2 * scipy.special.gammaincinv(1.5, (1 - 0.5) * scipy.special.gammainc(1.5, (chi * chi) / 2))) / (chi * chi))
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3)

        bounds = ((0, -numpy.inf, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, continuous_measures.min, 1)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"chi": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}
        return parameters


class Bates:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.n = self.parameters["n"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "bates"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"n": 5, "min": 2, "max": 10}

    def _n_int(self) -> int:
        return max(1, int(round(self.n)))

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        n = self._n_int()
        z = (numpy.asarray(x, dtype=float) - self.min) / (self.max - self.min)
        y = numpy.clip(z, 0.0, 1.0) * n
        result = numpy.zeros_like(y)
        flat_y = y.ravel()
        flat_res = numpy.zeros_like(flat_y)
        for idx, yi in enumerate(flat_y):
            kmax = int(numpy.floor(yi))
            total = 0.0
            for k in range(kmax + 1):
                total += ((-1) ** k) * scipy.special.comb(n, k) * (yi - k) ** n
            flat_res[idx] = total / scipy.special.factorial(n)
        result = flat_res.reshape(y.shape)
        result = numpy.where(z <= 0, 0.0, result)
        result = numpy.where(z >= 1, 1.0, result)
        if result.ndim == 0:
            return float(result)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        n = self._n_int()
        z = (numpy.asarray(x, dtype=float) - self.min) / (self.max - self.min)
        y = z * n
        flat_y = y.ravel()
        flat_res = numpy.zeros_like(flat_y)
        for idx, yi in enumerate(flat_y):
            if yi <= 0 or yi >= n:
                flat_res[idx] = 0.0
                continue
            kmax = int(numpy.floor(yi))
            total = 0.0
            for k in range(kmax + 1):
                total += ((-1) ** k) * scipy.special.comb(n, k) * (yi - k) ** (n - 1)
            flat_res[idx] = total / scipy.special.factorial(n - 1)
        result_0_1 = flat_res.reshape(y.shape)
        result = (n / (self.max - self.min)) * result_0_1
        if result.ndim == 0:
            return float(result)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        u = numpy.asarray(u, dtype=float)
        flat_u = u.ravel()
        flat_res = numpy.zeros_like(flat_u)
        for idx, ui in enumerate(flat_u):
            if ui <= 0:
                flat_res[idx] = self.min
                continue
            if ui >= 1:
                flat_res[idx] = self.max
                continue
            f = lambda xv: float(self.cdf(xv)) - ui
            flat_res[idx] = scipy.optimize.brentq(f, self.min, self.max)
        result = flat_res.reshape(u.shape)
        if result.ndim == 0:
            return float(result)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        n_int = self._n_int()
        u = numpy.random.rand(n, n_int).mean(axis=1)
        return self.min + (self.max - self.min) * u

    def non_central_moments(self, k: int) -> float | None:
        return None

    def central_moments(self, k: int) -> float | None:
        n = self.n
        if k == 1:
            return 0
        if k == 2:
            return 1 / (12 * n)
        if k == 3:
            return 0
        if k == 4:
            return (3 * n + 2) / (80 * n**3)
        return None

    @property
    def mean(self) -> float:
        return (self.min + self.max) / 2

    @property
    def variance(self) -> float:
        return (self.max - self.min) ** 2 / (12 * self.n)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 3 - 6 / (5 * self.n)

    @property
    def median(self) -> float:
        return (self.min + self.max) / 2

    @property
    def mode(self) -> float:
        return (self.min + self.max) / 2

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.n >= 1
        v2 = self.min < self.max
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        kurt = continuous_measures.kurtosis
        if kurt < 3 and kurt > 3 - 6 / 5:
            n_est = 6 / (5 * (3 - kurt))
        else:
            n_est = max(1.0, (continuous_measures.max - continuous_measures.min) ** 2 / (12 * continuous_measures.variance))
        n_est = max(1.0, n_est)
        half_range = numpy.sqrt(12 * n_est * continuous_measures.variance) / 2
        min_ = continuous_measures.mean - half_range
        max_ = continuous_measures.mean + half_range
        parameters = {"n": n_est, "min": min_, "max": max_}
        return parameters


class Beta:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]

    @property
    def name(self):
        return "beta"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 42, "beta": 10, "A": 518, "B": 969}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.beta.cdf(x, self.alpha, self.beta, loc=self.A, scale=self.B - self.A)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.beta.pdf(x, self.alpha, self.beta, loc=self.A, scale=self.B - self.A)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.A + (self.B - self.A) * scipy.special.betaincinv(self.alpha, self.beta, u)
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
        return self.A + (self.alpha / (self.alpha + self.beta)) * (self.B - self.A)

    @property
    def variance(self) -> float:
        return ((self.alpha * self.beta) / ((self.alpha + self.beta + 1) * (self.alpha + self.beta) ** 2)) * (self.B - self.A) ** 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 2 * ((self.beta - self.alpha) / (self.alpha + self.beta + 2)) * numpy.sqrt((self.alpha + self.beta + 1) / (self.alpha * self.beta))

    @property
    def kurtosis(self) -> float:
        return 3 + (6 * ((self.alpha + self.beta + 1) * (self.alpha - self.beta) ** 2 - self.alpha * self.beta * (self.alpha + self.beta + 2))) / (self.alpha * self.beta * (self.alpha + self.beta + 2) * (self.alpha + self.beta + 3))

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.A + ((self.alpha - 1) / (self.alpha + self.beta - 2)) * (self.B - self.A)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.A < self.B
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, A, B = initial_solution
            parametric_mean = A + (alpha / (alpha + beta)) * (B - A)
            parametric_variance = ((alpha * beta) / ((alpha + beta) ** 2 * (alpha + beta + 1))) * (B - A) ** 2
            parametric_skewness = 2 * ((beta - alpha) / (alpha + beta + 2)) * numpy.sqrt((alpha + beta + 1) / (alpha * beta))
            parametric_kurtosis = 3 * (((alpha + beta + 1) * (2 * (alpha + beta) ** 2 + (alpha * beta) * (alpha + beta - 6))) / ((alpha * beta) * (alpha + beta + 2) * (alpha + beta + 3)))
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        bounds = ((0, 0, -numpy.inf, continuous_measures.mean), (numpy.inf, numpy.inf, continuous_measures.mean, numpy.inf))
        x0 = (1, 1, continuous_measures.min, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "A": solution.x[2], "B": solution.x[3]}
        v1 = parameters["alpha"] > 0
        v2 = parameters["beta"] > 0
        v3 = parameters["A"] < parameters["B"]
        if (v1 and v2 and v3) == False:
            scipy_parameters = scipy.stats.beta.fit(continuous_measures.data_to_fit)
            parameters = {"alpha": scipy_parameters[0], "beta": scipy_parameters[1], "A": scipy_parameters[2], "B": scipy_parameters[3]}
        return parameters


warnings.filterwarnings("ignore")


class BetaPrime:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "beta_prime"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 101, "beta": 54}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.betainc(self.alpha, self.beta, x / (1 + x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (x ** (self.alpha - 1) * (1 + x) ** (-self.alpha - self.beta)) / (scipy.special.beta(self.alpha, self.beta))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.betaincinv(self.alpha, self.beta, u) / (1 - scipy.special.betaincinv(self.alpha, self.beta, u))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (scipy.special.gamma(k + self.alpha) * scipy.special.gamma(self.beta - k)) / (scipy.special.gamma(self.alpha) * scipy.special.gamma(self.beta))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (self.alpha - 1) / (self.beta + 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta = initial_solution
            parametric_mean = alpha / (beta - 1)
            parametric_variance = alpha * (alpha + beta - 1) / ((beta - 1) ** 2 * (beta - 2))
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        scipy_parameters = scipy.stats.betaprime.fit(continuous_measures.data_to_fit)
        try:
            x0 = (scipy_parameters[0], scipy_parameters[1])
            bounds = ((0, 0), (numpy.inf, numpy.inf))
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
        except:
            parameters = {"alpha": scipy_parameters[0], "beta": scipy_parameters[1]}
        return parameters


warnings.filterwarnings("ignore")


class BetaPrime4P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "beta_prime_4p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 911, "beta": 937, "loc": -7, "scale": 125}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.betaprime.cdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.betaprime.pdf(x, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.betaprime.ppf(u, self.alpha, self.beta, loc=self.loc, scale=self.scale)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (scipy.special.gamma(k + self.alpha) * scipy.special.gamma(self.beta - k)) / (scipy.special.gamma(self.alpha) * scipy.special.gamma(self.beta))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + (self.scale * (self.alpha - 1)) / (self.beta + 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.scale > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, loc, scale = initial_solution
            parametric_mean = scale * alpha / (beta - 1) + loc
            parametric_variance = (scale**2) * alpha * (alpha + beta - 1) / ((beta - 1) ** 2 * (beta - 2))
            parametric_median = loc + scale * scipy.stats.beta.ppf(0.5, alpha, beta) / (1 - scipy.stats.beta.ppf(0.5, alpha, beta))
            parametric_mode = scale * (alpha - 1) / (beta + 1) + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3, eq4)

        scipy_parameters = scipy.stats.betaprime.fit(continuous_measures.data_to_fit)
        try:
            x0 = (continuous_measures.mean, continuous_measures.mean, continuous_measures.min, scipy_parameters[3])
            bounds = ((0, 0, -numpy.inf, 0), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
            parameters = {"alpha": solution.x[0], "beta": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        except:
            parameters = {"alpha": scipy_parameters[0], "beta": scipy_parameters[1], "loc": scipy_parameters[2], "scale": scipy_parameters[3]}
        return parameters


class Bradford:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.c = self.parameters["c"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "bradford"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"c": 4, "min": 19, "max": 50}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.log(1 + self.c * (x - self.min) / (self.max - self.min)) / numpy.log(self.c + 1)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.c / ((self.c * (x - self.min) + self.max - self.min) * numpy.log(self.c + 1))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.min + ((numpy.exp(u * numpy.log(1 + self.c)) - 1) * (self.max - self.min)) / self.c
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
        return (self.c * (self.max - self.min) + numpy.log(1 + self.c) * (self.min * (self.c + 1) - self.max)) / (numpy.log(1 + self.c) * self.c)

    @property
    def variance(self) -> float:
        return ((self.max - self.min) ** 2 * ((self.c + 2) * numpy.log(1 + self.c) - 2 * self.c)) / (2 * self.c * numpy.log(1 + self.c) ** 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.sqrt(2) * (12 * self.c * self.c - 9 * numpy.log(1 + self.c) * self.c * (self.c + 2) + 2 * numpy.log(1 + self.c) * numpy.log(1 + self.c) * (self.c * (self.c + 3) + 3))) / (
            numpy.sqrt(self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c))) * (3 * self.c * (numpy.log(1 + self.c) - 2) + 6 * numpy.log(1 + self.c))
        )

    @property
    def kurtosis(self) -> float:
        return (
            self.c**3 * (numpy.log(1 + self.c) - 3) * (numpy.log(1 + self.c) * (3 * numpy.log(1 + self.c) - 16) + 24)
            + 12 * numpy.log(1 + self.c) * self.c * self.c * (numpy.log(1 + self.c) - 4) * (numpy.log(1 + self.c) - 3)
            + 6 * self.c * numpy.log(1 + self.c) ** 2 * (3 * numpy.log(1 + self.c) - 14)
            + 12 * numpy.log(1 + self.c) ** 3
        ) / (3 * self.c * (self.c * (numpy.log(1 + self.c) - 2) + 2 * numpy.log(1 + self.c)) ** 2) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.min

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.max > self.min
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(
            initial_solution: tuple[float],
            continuous_measures,
            precalculated_parameters: dict[str, int | float],
        ) -> tuple[float]:
            min_ = precalculated_parameters["min"]
            max_ = precalculated_parameters["max"]
            c = initial_solution
            parametric_mean = (c * (max_ - min_) + numpy.log(c + 1) * (min_ * (c + 1) - max_)) / (c * numpy.log(c + 1))
            eq1 = parametric_mean - continuous_measures.mean
            return eq1

        min_ = continuous_measures.min - 1e-3
        max_ = continuous_measures.max + 1e-3
        bounds = ((-numpy.inf), (numpy.inf))
        x0 = 1
        args = [continuous_measures, {"min": min_, "max": max_}]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"c": solution.x[0], "min": min_, "max": max_}
        return parameters


warnings.filterwarnings("ignore")


class Burr:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]

    @property
    def name(self):
        return "burr"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"A": 1, "B": 10, "C": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.cdf(x, self.B, self.C, scale=self.A)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.pdf(x, self.B, self.C, scale=self.A)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.A * ((1 - u) ** (-1 / self.C) - 1) ** (1 / self.B)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.A**k * self.C * ((scipy.special.gamma((self.B * self.C - k) / self.B) * scipy.special.gamma((self.B + k) / self.B)) / scipy.special.gamma((self.B * self.C - k) / self.B + (self.B + k) / self.B))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.A * ((self.B - 1) / (self.B * self.C + 1)) ** (1 / self.B)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.burr12.fit(continuous_measures.data_to_fit)
        parameters = {"A": scipy_parameters[3], "B": scipy_parameters[0], "C": scipy_parameters[1]}
        return parameters


warnings.filterwarnings("ignore")


class Burr4P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.A = self.parameters["A"]
        self.B = self.parameters["B"]
        self.C = self.parameters["C"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "burr_4p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"A": 108, "B": 114, "C": 1, "loc": 0}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.cdf(x, self.B, self.C, loc=self.loc, scale=self.A)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.burr12.pdf(x, self.B, self.C, loc=self.loc, scale=self.A)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.A * ((1 - u) ** (-1 / self.C) - 1) ** (1 / self.B) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.A**k * self.C * ((scipy.special.gamma((self.B * self.C - k) / self.B) * scipy.special.gamma((self.B + k) / self.B)) / scipy.special.gamma((self.B * self.C - k) / self.B + (self.B + k) / self.B))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.A * ((self.B - 1) / (self.B * self.C + 1)) ** (1 / self.B)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.A > 0
        v2 = self.C > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.burr12.fit(continuous_measures.data_to_fit)
        parameters = {"A": scipy_parameters[3], "B": scipy_parameters[0], "C": scipy_parameters[1], "loc": scipy_parameters[2]}
        return parameters


class Cauchy:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.x0 = self.parameters["x0"]
        self.gamma = self.parameters["gamma"]

    @property
    def name(self):
        return "cauchy"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"x0": 10, "gamma": 19}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 / numpy.pi) * numpy.arctan(((x - self.x0) / self.gamma)) + (1 / 2)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 / (numpy.pi * self.gamma * (1 + ((x - self.x0) / self.gamma) ** 2))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.x0 + self.gamma * numpy.tan(numpy.pi * (u - 0.5))
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
        return None

    @property
    def variance(self) -> float:
        return None

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return None

    @property
    def kurtosis(self) -> float:
        return None

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.x0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.gamma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.cauchy.fit(continuous_measures.data_to_fit)
        parameters = {"x0": scipy_parameters[0], "gamma": scipy_parameters[1]}
        return parameters


class ChiSquare:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.df = self.parameters["df"]

    @property
    def name(self):
        return "chi_square"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df": 7}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.df / 2, x / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.chi2.pdf(x, self.df)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = 2 * scipy.special.gammaincinv(self.df / 2, u)
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
        return self.df

    @property
    def variance(self) -> float:
        return self.df * 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return numpy.sqrt(8 / self.df)

    @property
    def kurtosis(self) -> float:
        return 12 / self.df + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return max(self.df - 2, 0)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = type(self.df) == int
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        parameters = {"df": round(continuous_measures.mean)}
        return parameters


class ChiSquare3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "chi_square_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df": 4, "loc": 10, "scale": 2}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = scipy.special.gammainc(self.df / 2, z(x) / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        result = (1 / self.scale) * (1 / (2 ** (self.df / 2) * scipy.special.gamma(self.df / 2))) * (z(x) ** ((self.df / 2) - 1)) * (numpy.exp(-z(x) / 2))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = 2 * self.scale * scipy.special.gammaincinv(self.df / 2, u) + self.loc
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
        return self.df * self.scale + self.loc

    @property
    def variance(self) -> float:
        return self.df * 2 * (self.scale * self.scale)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return numpy.sqrt(8 / self.df)

    @property
    def kurtosis(self) -> float:
        return 12 / self.df + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return max(self.df - 2, 0) * self.scale + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = type(self.df) == int
        v2 = True
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            df, loc, scale = initial_solution
            parametric_mean = df * scale + loc
            parametric_variance = 2 * df * (scale**2)
            parametric_skewness = numpy.sqrt(8 / df)
            parametric_median = 2 * scale * scipy.special.gammaincinv(df / 2, 0.5) + loc
            parametric_mode = max(df - 2, 0) * scale + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3)

        x0 = (1, continuous_measures.mean, 1)
        bounds = ((0, -numpy.inf, 0), (numpy.inf, numpy.inf, numpy.inf))
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"df": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}
        return parameters


class Dagum:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "dagum"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 5, "b": 56, "p": 1}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 + (x / self.b) ** (-self.a)) ** (-self.p)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.a * self.p / x) * (((x / self.b) ** (self.a * self.p)) / ((((x / self.b) ** (self.a)) + 1) ** (self.p + 1)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.b * (u ** (-1 / self.p) - 1) ** (-1 / self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.b**k * self.p * ((scipy.special.gamma((self.a * self.p + k) / self.a) * scipy.special.gamma((self.a - k) / self.a)) / scipy.special.gamma((self.a * self.p + k) / self.a + (self.a - k) / self.a))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.b * ((self.a * self.p - 1) / (self.a + 1)) ** (1 / self.a)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def sse(parameters: dict) -> float:
            def __pdf(x: float, params: dict) -> float:
                return (params["a"] * params["p"] / x) * (((x / params["b"]) ** (params["a"] * params["p"])) / ((((x / params["b"]) ** (params["a"])) + 1) ** (params["p"] + 1)))

            pdf_values = __pdf(continuous_measures.central_values, parameters)
            sse = numpy.sum(numpy.power(continuous_measures.densities_frequencies - pdf_values, 2))
            return sse

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, p = initial_solution
            mu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = mu(1)
            parametric_variance = -(mu(1) ** 2) + mu(2)
            parametric_mode = b * ((a * p - 1) / (a + 1)) ** (1 / a)
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3)

        s0_burr3_sc = scipy.stats.burr.fit(continuous_measures.data_to_fit)
        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1]}
        a0 = s0_burr3_sc[0]
        b0 = s0_burr3_sc[3]
        x0 = [a0, b0, 1]
        bounds = ((1e-5, 1e-5, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
        parameters_ls = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2]}
        sse_sc = sse(parameters_sc)
        sse_ls = sse(parameters_ls)
        if a0 <= 2:
            return parameters_sc
        else:
            if sse_sc < sse_ls:
                return parameters_sc
            else:
                return parameters_ls


warnings.filterwarnings("ignore")


class Dagum4P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "dagum_4p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 6, "b": 1, "p": 3, "loc": 100}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 + ((x - self.loc) / self.b) ** (-self.a)) ** (-self.p)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.a * self.p / x) * ((((x - self.loc) / self.b) ** (self.a * self.p)) / (((((x - self.loc) / self.b) ** (self.a)) + 1) ** (self.p + 1)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.b * (u ** (-1 / self.p) - 1) ** (-1 / self.a) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.b**k * self.p * ((scipy.special.gamma((self.a * self.p + k) / self.a) * scipy.special.gamma((self.a - k) / self.a)) / scipy.special.gamma((self.a * self.p + k) / self.a + (self.a - k) / self.a))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.b * ((self.a * self.p - 1) / (self.a + 1)) ** (1 / self.a)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.p > 0
        v2 = self.a > 0
        v3 = self.b > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def sse(parameters: dict) -> float:
            def __pdf(x: float, params: dict) -> float:
                return (params["a"] * params["p"] / (x - params["loc"])) * ((((x - params["loc"]) / params["b"]) ** (params["a"] * params["p"])) / ((((x / params["b"]) ** (params["a"])) + 1) ** (params["p"] + 1)))

            frequencies, bin_edges = numpy.histogram(continuous_measures.data, density=True)
            central_values = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
            pdf_values = [__pdf(c, parameters) for c in central_values]
            sse = numpy.sum(numpy.power(frequencies - pdf_values, 2))
            return sse

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, p, loc = initial_solution
            mu = lambda k: (b**k) * p * scipy.special.beta((a * p + k) / a, (a - k) / a)
            parametric_mean = mu(1) + loc
            parametric_variance = -(mu(1) ** 2) + mu(2)
            parametric_median = b * ((2 ** (1 / p)) - 1) ** (-1 / a) + loc
            parametric_mode = b * ((a * p - 1) / (a + 1)) ** (1 / a) + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3, eq4)

        s0_burr3_sc = scipy.stats.burr.fit(continuous_measures.data_to_fit)
        parameters_sc = {"a": s0_burr3_sc[0], "b": s0_burr3_sc[3], "p": s0_burr3_sc[1], "loc": s0_burr3_sc[2]}
        if s0_burr3_sc[0] <= 2:
            return parameters_sc
        else:
            a0 = s0_burr3_sc[0]
            x0 = [a0, 1, 1, continuous_measures.min]
            bounds = ((1e-5, 1e-5, 1e-5, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
            parameters_ls = {"a": solution.x[0], "b": solution.x[1], "p": solution.x[2], "loc": solution.x[3]}
            sse_sc = sse(parameters_sc)
            sse_ls = sse(parameters_ls)
            if sse_sc < sse_ls:
                return parameters_sc
            else:
                return parameters_ls


class Erlang:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "erlang"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"k": 48, "beta": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.k, x / self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.erlang.pdf(x, self.k, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.k, u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * (scipy.special.gamma(self.k + k) / scipy.special.gamma(self.k))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.k - 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.k > 0
        v2 = self.beta > 0
        v3 = type(self.k) == int
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        k = round(continuous_measures.mean**2 / continuous_measures.variance)
        beta = continuous_measures.variance / continuous_measures.mean
        parameters = {"k": k, "beta": beta}
        return parameters


class Erlang3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.k = self.parameters["k"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "erlang_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"k": 54, "beta": 5, "loc": 981}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.erlang.cdf(x, self.k, scale=self.beta, loc=self.loc)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.erlang.pdf(x, self.k, scale=self.beta, loc=self.loc)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.k, u) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * (scipy.special.gamma(k + self.k) / scipy.special.gamma(self.k))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.k - 1) + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.k > 0
        v2 = self.beta > 0
        v3 = type(self.k) == int
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        k = round((2 / continuous_measures.skewness) ** 2)
        beta = numpy.sqrt(continuous_measures.variance / ((2 / continuous_measures.skewness) ** 2))
        loc = continuous_measures.mean - ((2 / continuous_measures.skewness) ** 2) * beta
        parameters = {"k": k, "beta": beta, "loc": loc}
        return parameters


class ErrorFunction:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.h = self.parameters["h"]

    @property
    def name(self):
        return "error_function"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"h": 9}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return scipy.stats.norm.cdf((2**0.5) * self.h * x)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.h * numpy.exp(-((self.h * x) ** 2)) / numpy.sqrt(numpy.pi)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.norm.ppf(u) / (self.h * numpy.sqrt(2))
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
        return 0

    @property
    def variance(self) -> float:
        return 1 / (2 * self.h**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.h > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        h = numpy.sqrt(1 / (2 * continuous_measures.variance))
        parameters = {"h": h}
        return parameters


class Exponential:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.lambda_ = self.parameters["lambda"]

    @property
    def name(self):
        return "exponential"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 0.05}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 - numpy.exp(-self.lambda_ * x)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.lambda_ * numpy.exp(-self.lambda_ * x)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = -numpy.log(1 - u) / self.lambda_
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
        return 1 / self.lambda_

    @property
    def variance(self) -> float:
        return 1 / (self.lambda_ * self.lambda_)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 2

    @property
    def kurtosis(self) -> float:
        return 9

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        lambda_ = 1 / continuous_measures.mean
        parameters = {"lambda": lambda_}
        return parameters


class Exponential2P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "exponential_2p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 0.01, "loc": 50}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 - numpy.exp(-self.lambda_ * (x - self.loc))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.lambda_ * numpy.exp(-self.lambda_ * (x - self.loc))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc - numpy.log(1 - u) / self.lambda_
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
        return 1 / self.lambda_ + self.loc

    @property
    def variance(self) -> float:
        return 1 / (self.lambda_ * self.lambda_)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 2

    @property
    def kurtosis(self) -> float:
        return 9

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.lambda_ > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        lambda_ = (1 - numpy.log(2)) / (continuous_measures.mean - continuous_measures.median)
        loc = continuous_measures.min - 1e-4
        parameters = {"lambda": lambda_, "loc": loc}
        return parameters


class ExponentiatedKumaraswamy:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.lambda_ = self.parameters["lambda"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "exponentiated_kumaraswamy"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 3, "beta": 4, "lambda": 2, "min": 5, "max": 15}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (1 - (1 - z(x) ** self.alpha) ** self.beta) ** self.lambda_

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        zx = z(x)
        result = (self.alpha * self.beta * self.lambda_ * zx ** (self.alpha - 1) * (1 - zx**self.alpha) ** (self.beta - 1) * (1 - (1 - zx**self.alpha) ** self.beta) ** (self.lambda_ - 1)) / (self.max - self.min)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.min + (self.max - self.min) * (1 - (1 - u ** (1 / self.lambda_)) ** (1 / self.beta)) ** (1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        total = 0.0
        for j in range(200):
            term = ((-1) ** j) * scipy.special.binom(k / self.alpha, j) * scipy.special.beta(j / self.beta + 1, self.lambda_)
            total += term
            if j > 20 and abs(term) < 1e-16 * max(1.0, abs(total)):
                break
        return self.lambda_ * total

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.min + (self.max - self.min) * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.max - self.min) ** 2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        standard_pdf = lambda z: -(self.alpha * self.beta * self.lambda_ * z ** (self.alpha - 1) * (1 - z**self.alpha) ** (self.beta - 1) * (1 - (1 - z**self.alpha) ** self.beta) ** (self.lambda_ - 1))
        res = scipy.optimize.minimize_scalar(standard_pdf, bounds=(1e-6, 1 - 1e-6), method="bounded")
        return self.min + (self.max - self.min) * res.x

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.lambda_ > 0
        v4 = self.min < self.max
        return v1 and v2 and v3 and v4

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def E(k, alpha, beta, lambda_):
            total = 0.0
            for j in range(200):
                term = ((-1) ** j) * scipy.special.binom(k / alpha, j) * scipy.special.beta(j / beta + 1, lambda_)
                total += term
                if j > 20 and abs(term) < 1e-16 * max(1.0, abs(total)):
                    break
            return lambda_ * total

        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, lambda_, min_, max_ = initial_solution
            e1 = E(1, alpha, beta, lambda_)
            e2 = E(2, alpha, beta, lambda_)
            e3 = E(3, alpha, beta, lambda_)
            e4 = E(4, alpha, beta, lambda_)
            parametric_mean = e1 * (max_ - min_) + min_
            parametric_variance = (e2 - e1**2) * (max_ - min_) ** 2
            parametric_skewness = (e3 - 3 * e2 * e1 + 2 * e1**3) / ((e2 - e1**2)) ** 1.5
            parametric_kurtosis = (e4 - 4 * e1 * e3 + 6 * e1**2 * e2 - 3 * e1**4) / ((e2 - e1**2)) ** 2
            z_med = (1 - (1 - 0.5 ** (1 / lambda_)) ** (1 / beta)) ** (1 / alpha)
            parametric_median = min_ + (max_ - min_) * z_med
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            eq5 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3, eq4, eq5)

        bounds = (
            (1e-5, 1e-5, 1e-5, -numpy.inf, continuous_measures.mean),
            (numpy.inf, numpy.inf, numpy.inf, continuous_measures.mean, numpy.inf),
        )
        x0 = (1, 1, 1, continuous_measures.min - 1e-3, continuous_measures.max + 1e-3)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {
            "alpha": solution.x[0],
            "beta": solution.x[1],
            "lambda": solution.x[2],
            "min": solution.x[3],
            "max": solution.x[4],
        }
        return parameters


class F:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]

    @property
    def name(self):
        return "f"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df1": 5, "df2": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.cdf(x, self.df1, self.df2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.pdf(x, self.df1, self.df2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        t = scipy.special.betaincinv(self.df1 / 2, self.df2 / 2, u)
        result = (self.df2 * t) / (self.df1 * (1 - t))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.df2 / self.df1) ** k * (scipy.special.gamma(self.df1 / 2 + k) / scipy.special.gamma(self.df1 / 2)) * (scipy.special.gamma(self.df2 / 2 - k) / scipy.special.gamma(self.df2 / 2))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (self.df2 * (self.df1 - 2)) / (self.df1 * (self.df2 + 2))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.f.fit(continuous_measures.data_to_fit)
        parameters = {"df1": scipy_parameters[0], "df2": scipy_parameters[1]}
        return parameters


class FatigueLife:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.gamma = self.parameters["gamma"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "fatigue_life"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"gamma": 5, "loc": 3, "scale": 9}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fatiguelife.cdf(x, self.gamma, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fatiguelife.pdf(x, self.gamma, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fatiguelife.ppf(u, self.gamma, loc=self.loc, scale=self.scale)
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
        return self.loc + self.scale * (1 + self.gamma**2 / 2)

    @property
    def variance(self) -> float:
        return self.scale**2 * self.gamma**2 * (1 + (5 * self.gamma**2) / 4)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (4 * self.gamma**2 * (11 * self.gamma**2 + 6)) / ((5 * self.gamma**2 + 4) * numpy.sqrt(self.gamma**2 * (5 * self.gamma**2 + 4)))

    @property
    def kurtosis(self) -> float:
        return 3 + (6 * self.gamma * self.gamma * (93 * self.gamma * self.gamma + 40)) / (5 * self.gamma**2 + 4) ** 2

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
        v1 = self.scale > 0
        v2 = self.gamma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            gamma, loc, scale = initial_solution
            parametric_mean = loc + scale * (1 + gamma**2 / 2)
            parametric_variance = scale**2 * gamma**2 * (1 + 5 * gamma**2 / 4)
            parametric_kurtosis = 3 + (6 * gamma * gamma * (93 * gamma * gamma + 40)) / (5 * gamma**2 + 4) ** 2
            parametric_median = loc + (scale * (gamma * scipy.stats.norm.ppf(0.5) + numpy.sqrt((gamma * scipy.stats.norm.ppf(0.5)) ** 2 + 4)) ** 2) / 4
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3)

        scipy_parameters = scipy.stats.fatiguelife.fit(continuous_measures.data_to_fit)
        parameters = {"gamma": scipy_parameters[0], "loc": scipy_parameters[1], "scale": scipy_parameters[2]}
        return parameters


class FoldedNormal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "folded_normal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 100, "sigma": 59}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z1 = lambda t: (t + self.mu) / self.sigma
        z2 = lambda t: (t - self.mu) / self.sigma
        result = 0.5 * (scipy.special.erf(z1(x) / numpy.sqrt(2)) + scipy.special.erf(z2(x) / numpy.sqrt(2)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(2 / (numpy.pi * self.sigma**2)) * numpy.exp(-(x**2 + self.mu**2) / (2 * self.sigma**2)) * numpy.cosh(self.mu * x / (self.sigma**2))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.foldnorm.ppf(u, self.mu / self.sigma, loc=0, scale=self.sigma)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        f = lambda x: x**k * self.pdf(x)
        return scipy.integrate.quad(f, 0, self.mu + 4 * self.sigma)[0]

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        return self.sigma * numpy.sqrt(2 / numpy.pi) * numpy.exp((-self.mu * self.mu) / (2 * self.sigma * self.sigma)) - self.mu * (2 * scipy.stats.norm.cdf(-self.mu / self.sigma) - 1)

    @property
    def variance(self) -> float:
        return self.mu * self.mu + self.sigma * self.sigma - self.mean**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = sigma * numpy.sqrt(2 / numpy.pi) * numpy.exp(-(mu**2) / (2 * sigma**2)) + mu * scipy.special.erf(mu / numpy.sqrt(2 * sigma**2))
            parametric_variance = mu**2 + sigma**2 - parametric_mean**2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        x0 = [continuous_measures.mean, continuous_measures.standard_deviation]
        bounds = ((-numpy.inf, 0), (numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
        parameters = {"mu": solution.x[0], "sigma": solution.x[1]}
        return parameters


class Frechet:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "frechet"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 5, "loc": 9, "scale": 21}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invweibull.cdf(x, self.alpha, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invweibull.pdf(x, self.alpha, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.scale * (-numpy.log(u)) ** (-1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return scipy.special.gamma(1 - k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.scale * (self.alpha / (self.alpha + 1)) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, loc, scale = initial_solution
            E = lambda k: scipy.special.gamma(1 - k / alpha)
            parametric_mean = E(1) * scale + loc
            parametric_variance = (scale**2) * (E(2) - E(1) ** 2)
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        scipy_parameters = scipy.stats.invweibull.fit(continuous_measures.data_to_fit)
        parameters = {"alpha": scipy_parameters[0], "loc": scipy_parameters[1], "scale": scipy_parameters[2]}
        return parameters


class F4P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.df1 = self.parameters["df1"]
        self.df2 = self.parameters["df2"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "f_4p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df1": 76, "df2": 36, "loc": 925, "scale": 197}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.cdf(x, self.df1, self.df2, self.loc, self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.f.pdf(x, self.df1, self.df2, self.loc, self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        t = scipy.special.betaincinv(self.df1 / 2, self.df2 / 2, u)
        result = self.loc + (self.scale * (self.df2 * t)) / (self.df1 * (1 - t))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.df2 / self.df1) ** k * (scipy.special.gamma(self.df1 / 2 + k) / scipy.special.gamma(self.df1 / 2)) * (scipy.special.gamma(self.df2 / 2 - k) / scipy.special.gamma(self.df2 / 2))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return ((self.df2 * (self.df1 - 2)) / (self.df1 * (self.df2 + 2))) * self.scale + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df1 > 0
        v2 = self.df2 > 0
        v3 = self.scale > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            df1, df2, loc, scale = initial_solution
            E = lambda k: (df2 / df1) ** k * (scipy.special.gamma(df1 / 2 + k) * scipy.special.gamma(df2 / 2 - k)) / (scipy.special.gamma(df1 / 2) * scipy.special.gamma(df2 / 2))
            parametric_mean = E(1) * scale + loc
            parametric_variance = (E(2) - E(1) ** 2) * (scale) ** 2
            parametric_median = scipy.stats.f.ppf(0.5, df1, df2) * scale + loc
            parametric_mode = ((df2 * (df1 - 2)) / (df1 * (df2 + 2))) * scale + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            eq4 = parametric_mode - continuous_measures.mode
            return (eq1, eq2, eq3, eq4)

        try:
            bounds = ((0, 0, -numpy.inf, 0), (numpy.inf, numpy.inf, continuous_measures.min, numpy.inf))
            x0 = (1, continuous_measures.standard_deviation, continuous_measures.min, continuous_measures.standard_deviation)
            args = [continuous_measures]
            solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
            parameters = {"df1": solution.x[0], "df2": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        except:
            scipy_parameters = scipy.stats.f.fit(continuous_measures.data_to_fit)
            parameters = {"df1": scipy_parameters[0], "df2": scipy_parameters[1], "loc": scipy_parameters[2], "scale": scipy_parameters[3]}
        return parameters


class Gamma:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "gamma"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 1, "beta": 10}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.alpha, x / self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gamma.pdf(x, self.alpha, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.alpha, u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * (scipy.special.gamma(k + self.alpha) / scipy.special.gamma(self.alpha))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.alpha - 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        alpha = continuous_measures.mean**2 / continuous_measures.variance
        beta = continuous_measures.variance / continuous_measures.mean
        parameters = {"alpha": alpha, "beta": beta}
        return parameters


class Gamma3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "gamma_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 22, "loc": 102, "beta": 2}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.alpha, (x - self.loc) / self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gamma.pdf(x, self.alpha, loc=self.loc, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * scipy.special.gammaincinv(self.alpha, u) + self.loc
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * (scipy.special.gamma(k + self.alpha) / scipy.special.factorial(self.alpha - 1))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta * (self.alpha - 1) + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        alpha = (2 / continuous_measures.skewness) ** 2
        beta = numpy.sqrt(continuous_measures.variance / alpha)
        loc = continuous_measures.mean - alpha * beta
        parameters = {"alpha": alpha, "loc": loc, "beta": beta}
        return parameters


class GeneralizedExtremeValue:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.xi = self.parameters["xi"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "generalized_extreme_value"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xi": 0, "mu": 10, "sigma": 1}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return numpy.exp(-numpy.exp(-z(x)))
        else:
            return numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi)))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        if self.xi == 0:
            return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))
        else:
            return (1 / self.sigma) * numpy.exp(-((1 + self.xi * z(x)) ** (-1 / self.xi))) * (1 + self.xi * z(x)) ** (-1 - 1 / self.xi)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        if self.xi == 0:
            result = self.mu - self.sigma * numpy.log(-numpy.log(u))
        else:
            result = self.mu + (self.sigma * ((-numpy.log(u)) ** -self.xi - 1)) / self.xi
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return scipy.special.gamma(1 - self.xi * k)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        if self.xi == 0:
            return self.mu + self.sigma * 0.5772156649
        µ1 = self.non_central_moments(1)
        return self.mu + (self.sigma * (µ1 - 1)) / self.xi

    @property
    def variance(self) -> float:
        if self.xi == 0:
            return self.sigma**2 * (numpy.pi**2 / 6)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.sigma**2 * (µ2 - µ1**2)) / self.xi**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        if self.xi == 0:
            return (12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        if self.xi == 0:
            return 5.4
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.xi == 0:
            return self.mu
        return self.mu + (self.sigma * ((1 + self.xi) ** -self.xi - 1)) / self.xi

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.genextreme.fit(continuous_measures.data_to_fit)
        parameters = {"xi": -scipy_parameters[0], "mu": scipy_parameters[1], "sigma": scipy_parameters[2]}
        return parameters


warnings.filterwarnings("ignore")


class GeneralizedGamma:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]

    @property
    def name(self):
        return "generalized_gamma"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 10, "d": 128, "p": 24}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gengamma.cdf(x, self.d / self.p, self.p, scale=self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gengamma.cdf(x, self.d / self.p, self.p, scale=self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gengamma.ppf(u, self.d / self.p, self.p, scale=self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.a**k * scipy.special.gamma((self.d + k) / self.p)) / scipy.special.gamma(self.d / self.p)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.a * ((self.d - 1) / self.p) ** (1 / self.p)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, d, p = initial_solution
            E = lambda r: a**r * (scipy.special.gamma((d + r) / p) / scipy.special.gamma(d / p))
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        try:
            bounds = ((1e-5, 1e-5, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
            x0 = (continuous_measures.mean, 1, 1)
            args = [continuous_measures]
            response = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
            solution = response.x
            parameters = {"a": solution[0], "d": solution[1], "p": solution[2]}
        except:
            scipy_parameters = scipy.stats.gengamma.fit(continuous_measures.data_to_fit)
            parameters = {"a": scipy_parameters[0], "c": scipy_parameters[1], "mu": scipy_parameters[2]}
        return parameters


class GeneralizedGamma4P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.d = self.parameters["d"]
        self.p = self.parameters["p"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "generalized_gamma_4p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 2, "d": 13, "p": 3, "loc": 28}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gengamma.cdf(x, self.d / self.p, self.p, loc=self.loc, scale=self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (self.p / (self.a**self.d)) * ((x - self.loc) ** (self.d - 1)) * numpy.exp(-(((x - self.loc) / self.a) ** self.p)) / scipy.special.gamma(self.d / self.p)
        result = scipy.stats.gengamma.pdf(x, self.d / self.p, self.p, loc=self.loc, scale=self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gengamma.ppf(u, self.d / self.p, self.p, loc=self.loc, scale=self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.a**k * scipy.special.gamma((self.d + k) / self.p)) / scipy.special.gamma(self.d / self.p)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.a * ((self.d - 1) / self.p) ** (1 / self.p)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a > 0
        v2 = self.d > 0
        v3 = self.p > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, d, p, loc = initial_solution
            E = lambda r: a**r * (scipy.special.gamma((d + r) / p) / scipy.special.gamma(d / p))
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        try:
            bounds = ((1e-5, 1e-5, 1e-5, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
            x0 = (1, 1, continuous_measures.mean, continuous_measures.mean)
            args = [continuous_measures]
            response = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
            solution = response.x
            parameters = {"a": solution[0], "d": solution[1], "p": solution[2], "loc": solution[3]}
        except:
            scipy_parameters = scipy.stats.gengamma.fit(continuous_measures.data_to_fit)
            parameters = {"a": scipy_parameters[3], "d": scipy_parameters[0], "p": scipy_parameters[1], "loc": scipy_parameters[2]}
        return parameters


class GeneralizedLogistic:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "generalized_logistic"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"c": 2, "loc": 25, "scale": 32}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        return 1 / ((1 + numpy.exp(-z(x))) ** self.c)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.scale
        return (self.c / self.scale) * numpy.exp(-z(x)) * ((1 + numpy.exp(-z(x))) ** (-self.c - 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.scale * -numpy.log(u ** (-1 / self.c) - 1)
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
        return self.loc + self.scale * (0.57721 + scipy.special.digamma(self.c))

    @property
    def variance(self) -> float:
        return self.scale * self.scale * ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c))

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2.40411380631918 + scipy.special.polygamma(2, self.c)) / ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c)) ** 1.5

    @property
    def kurtosis(self) -> float:
        return 3 + (6.49393940226682 + scipy.special.polygamma(3, self.c)) / ((numpy.pi * numpy.pi) / 6 + scipy.special.polygamma(1, self.c)) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.scale * numpy.log(self.c)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        v2 = self.c > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            c, loc, scale = initial_solution
            parametric_mean = loc + scale * (0.57721 + scipy.special.digamma(c))
            parametric_variance = scale**2 * (numpy.pi**2 / 6 + scipy.special.polygamma(1, c))
            parametric_median = loc + scale * (-numpy.log(0.5 ** (-1 / c) - 1))
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3)

        x0 = [1, continuous_measures.min, 1]
        bounds = ((1e-5, -numpy.inf, 1e-5), (numpy.inf, numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
        parameters = {"c": solution.x[0], "loc": solution.x[1], "scale": solution.x[2]}
        return parameters


class GeneralizedNormal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.beta = self.parameters["beta"]
        self.mu = self.parameters["mu"]
        self.alpha = self.parameters["alpha"]

    @property
    def name(self):
        return "generalized_normal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"beta": 1, "mu": 0, "alpha": 3}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 0.5 + (numpy.sign(x - self.mu) / 2) * scipy.special.gammainc(1 / self.beta, abs((x - self.mu) / self.alpha) ** self.beta)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.beta / (2 * self.alpha * scipy.special.gamma(1 / self.beta)) * numpy.exp(-((abs(x - self.mu) / self.alpha) ** self.beta))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + numpy.sign(u - 0.5) * (self.alpha**self.beta * scipy.special.gammaincinv(1 / self.beta, 2 * numpy.abs(u - 0.5))) ** (1 / self.beta)
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
        return self.mu

    @property
    def variance(self) -> float:
        return (self.mu**2 * scipy.special.gamma(3 / self.alpha)) / scipy.special.gamma(1 / self.alpha)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return (scipy.special.gamma(5 / self.alpha) * scipy.special.gamma(1 / self.alpha)) / scipy.special.gamma(3 / self.alpha) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.gennorm.fit(continuous_measures.data_to_fit)
        parameters = {"beta": scipy_parameters[0], "mu": scipy_parameters[1], "alpha": scipy_parameters[2]}
        return parameters


class GeneralizedPareto:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "generalized_pareto"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"c": -3, "mu": 31, "sigma": 47}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = 1 - (1 + self.c * z(x)) ** (-1 / self.c)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = (1 / self.sigma) * (1 + self.c * z(x)) ** (-1 / self.c - 1)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + (self.sigma * ((1 - u) ** -self.c - 1)) / self.c
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
        return self.mu + self.sigma / (1 - self.c)

    @property
    def variance(self) -> float:
        return (self.sigma * self.sigma) / ((1 - self.c) * (1 - self.c) * (1 - 2 * self.c))

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 * (1 + self.c) * numpy.sqrt(1 - 2 * self.c)) / (1 - 3 * self.c)

    @property
    def kurtosis(self) -> float:
        return (3 * (1 - 2 * self.c) * (2 * self.c * self.c + self.c + 3)) / ((1 - 3 * self.c) * (1 - 4 * self.c))

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            c, mu, sigma = initial_solution
            parametric_mean = mu + sigma / (1 - c)
            parametric_variance = sigma * sigma / ((1 - c) * (1 - c) * (1 - 2 * c))
            parametric_median = mu + sigma * (2**c - 1) / c
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - numpy.percentile(continuous_measures.data, 50)
            return (eq1, eq2, eq3)

        scipy_parameters = scipy.stats.genpareto.fit(continuous_measures.data_to_fit)
        parameters = {"c": scipy_parameters[0], "mu": scipy_parameters[1], "sigma": scipy_parameters[2]}
        if parameters["c"] < 0:
            scipy_parameters = scipy.stats.genpareto.fit(continuous_measures.data_to_fit)
            c0 = scipy_parameters[0]
            x0 = [c0, continuous_measures.min, 1]
            bounds = ((-numpy.inf, -numpy.inf, 0), (numpy.inf, continuous_measures.min, numpy.inf))
            solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
            parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
            parameters["mu"] = min(parameters["mu"], continuous_measures.min - 1e-3)
            delta_sigma = parameters["c"] * (parameters["mu"] - continuous_measures.max) - parameters["sigma"]
            parameters["sigma"] = parameters["sigma"] + delta_sigma + 1e-8
        return parameters


class Gibrat:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "gibrat"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"loc": 29, "scale": 102}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gibrat.cdf(x, self.loc, self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.gibrat.pdf(x, self.loc, self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.exp(scipy.stats.norm.ppf(u)) * self.scale + self.loc
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
        return self.loc + self.scale * numpy.sqrt(numpy.exp(1))

    @property
    def variance(self) -> float:
        return numpy.exp(1) * (numpy.exp(1) - 1) * self.scale * self.scale

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 + numpy.exp(1)) * numpy.sqrt(numpy.exp(1) - 1)

    @property
    def kurtosis(self) -> float:
        return numpy.exp(1) ** 4 + 2 * numpy.exp(1) ** 3 + 3 * numpy.exp(1) ** 2 - 6

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (1 / numpy.exp(1)) * self.scale + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.scale > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scale = numpy.sqrt(continuous_measures.variance / (numpy.e**2 - numpy.e))
        loc = continuous_measures.mean - scale * numpy.sqrt(numpy.e)
        parameters = {"loc": loc, "scale": scale}
        return parameters


class GumbelLeft:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "gumbel_left"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 100, "sigma": 30}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return 1 - numpy.exp(-numpy.exp(z(x)))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return (1 / self.sigma) * numpy.exp(z(x) - numpy.exp(z(x)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * numpy.log(-numpy.log(1 - u))
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
        return self.mu - 0.5772156649 * self.sigma

    @property
    def variance(self) -> float:
        return self.sigma**2 * (numpy.pi**2 / 6)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (-12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3

    @property
    def kurtosis(self) -> float:
        return 3 + 12 / 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = mu - sigma * 0.5772156649
            parametric_variance = (sigma**2) * (numpy.pi**2) / 6
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        x0 = [continuous_measures.mode, 1]
        bounds = ((-numpy.inf, 1e-5), (numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
        parameters = {"mu": solution.x[0], "sigma": solution.x[1]}
        return parameters


class GumbelRight:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "gumbel_right"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 98, "sigma": 59}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return numpy.exp(-numpy.exp(-z(x)))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        return (1 / self.sigma) * numpy.exp(-z(x) - numpy.exp(-z(x)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu - self.sigma * numpy.log(-numpy.log(u))
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
        return self.mu + 0.5772156649 * self.sigma

    @property
    def variance(self) -> float:
        return self.sigma**2 * (numpy.pi**2 / 6)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (12 * numpy.sqrt(6) * 1.20205690315959) / numpy.pi**3

    @property
    def kurtosis(self) -> float:
        return 3 + 12 / 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            mu, sigma = initial_solution
            parametric_mean = mu + sigma * 0.5772156649
            parametric_variance = (sigma**2) * (numpy.pi**2) / 6
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        x0 = [continuous_measures.mode, 1]
        bounds = ((-numpy.inf, 1e-5), (numpy.inf, numpy.inf))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=[continuous_measures])
        parameters = {"mu": solution.x[0], "sigma": solution.x[1]}
        return parameters


class HalfNormal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "half_normal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 19, "sigma": 7}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = scipy.special.erf(z(x) / numpy.sqrt(2))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = (1 / self.sigma) * numpy.sqrt(2 / numpy.pi) * numpy.exp(-(z(x) ** 2) / 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.norm.ppf((1 + u) / 2) * self.sigma + self.mu
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
        return self.mu + self.sigma * numpy.sqrt(2 / numpy.pi)

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma * (1 - 2 / numpy.pi)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.sqrt(2) * (4 - numpy.pi)) / (numpy.pi - 2) ** 1.5

    @property
    def kurtosis(self) -> float:
        return 3 + (8 * (numpy.pi - 3)) / (numpy.pi - 2) ** 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        sigma = numpy.sqrt(continuous_measures.variance / (1 - 2 / numpy.pi))
        mu = continuous_measures.mean - sigma * numpy.sqrt(2) / numpy.sqrt(numpy.pi)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class HyperbolicSecant:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "hyperbolic_secant"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 1002, "sigma": 198}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (2 / numpy.pi) * numpy.arctan(numpy.exp((z(x))))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.pi * (t - self.mu) / (2 * self.sigma)
        return (1 / numpy.cosh(z(x))) / (2 * self.sigma)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.log(numpy.tan((u * numpy.pi) / 2)) * ((2 * self.sigma) / numpy.pi) + self.mu
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
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        sigma = numpy.sqrt(continuous_measures.variance)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class InverseGamma:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "inverse_gamma"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 4, "beta": 12}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.cdf(x, a=self.alpha, scale=self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.pdf(x, a=self.alpha, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta / scipy.special.gammaincinv(self.alpha, 1 - u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return self.beta**k / (self.alpha - 1)
        if k == 2:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2))
        if k == 3:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3))
        if k == 4:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3) * (self.alpha - 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta / (self.alpha + 1)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.invgamma.fit(continuous_measures.data_to_fit)
        parameters = {"alpha": scipy_parameters[0], "beta": scipy_parameters[2]}
        return parameters


class InverseGamma3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "inverse_gamma_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 5, "loc": 99, "beta": 11}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.cdf(x, a=self.alpha, loc=self.loc, scale=self.beta)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgamma.pdf(x, a=self.alpha, loc=self.loc, scale=self.beta)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.beta / scipy.special.gammaincinv(self.alpha, 1 - u)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return self.beta**k / (self.alpha - 1)
        if k == 2:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2))
        if k == 3:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3))
        if k == 4:
            return self.beta**k / ((self.alpha - 1) * (self.alpha - 2) * (self.alpha - 3) * (self.alpha - 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.beta / (self.alpha + 1) + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.invgamma.fit(continuous_measures.data_to_fit)
        parameters = {"alpha": scipy_parameters[0], "loc": scipy_parameters[1], "beta": scipy_parameters[2]}
        return parameters


class InverseGaussian:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.lambda_ = self.parameters["lambda"]

    @property
    def name(self):
        return "inverse_gaussian"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 10, "lambda": 19}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.cdf(x, self.mu / self.lambda_, scale=self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.pdf(x, self.mu / self.lambda_, scale=self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.ppf(u, self.mu / self.lambda_, scale=self.lambda_)
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
        return self.mu

    @property
    def variance(self) -> float:
        return self.mu**3 / self.lambda_

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 3 * numpy.sqrt(self.mu / self.lambda_)

    @property
    def kurtosis(self) -> float:
        return 15 * (self.mu / self.lambda_) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu * (numpy.sqrt(1 + (9 * self.mu * self.mu) / (4 * self.lambda_ * self.lambda_)) - (3 * self.mu) / (2 * self.lambda_))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        lambda_ = mu**3 / continuous_measures.variance
        parameters = {"mu": mu, "lambda": lambda_}
        return parameters


class InverseGaussian3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.lambda_ = self.parameters["lambda"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "inverse_gaussian_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 9, "lambda": 77, "loc": 60}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.cdf(x, self.mu / self.lambda_, loc=self.loc, scale=self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.pdf(x, self.mu / self.lambda_, loc=self.loc, scale=self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.invgauss.ppf(u, self.mu / self.lambda_, loc=self.loc, scale=self.lambda_)
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
        return self.mu + self.loc

    @property
    def variance(self) -> float:
        return self.mu**3 / self.lambda_

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 3 * numpy.sqrt(self.mu / self.lambda_)

    @property
    def kurtosis(self) -> float:
        return 15 * (self.mu / self.lambda_) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.mu * (numpy.sqrt(1 + (9 * self.mu * self.mu) / (4 * self.lambda_ * self.lambda_)) - (3 * self.mu) / (2 * self.lambda_))

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = 3 * numpy.sqrt(continuous_measures.variance / (continuous_measures.skewness**2))
        lambda_ = mu**3 / continuous_measures.variance
        loc = continuous_measures.mean - mu
        parameters = {"mu": mu, "lambda": lambda_, "loc": loc}
        return parameters


class JohnsonSB:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]

    @property
    def name(self):
        return "johnson_sb"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xi": 102, "lambda": 794, "gamma": 4, "delta": 1}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.johnsonsb.cdf(x, self.gamma_, self.delta_, loc=self.xi_, scale=self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.johnsonsb.pdf(x, self.gamma_, self.delta_, loc=self.xi_, scale=self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.johnsonsb.ppf(u, self.gamma_, self.delta_, loc=self.xi_, scale=self.lambda_)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        f = lambda x: x**k * (self.delta_ / (numpy.sqrt(2 * numpy.pi) * x * (1 - x))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.log(x / (1 - x))) ** 2)
        return scipy.integrate.quad(f, 0, 1)[0]

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.xi_ + self.lambda_ * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.lambda_ * self.lambda_ * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        return central_µ3 / (µ2 - µ1**2) ** 1.5

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        return central_µ4 / (µ2 - µ1**2) ** 2

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
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.johnsonsb.fit(continuous_measures.data_to_fit)
        parameters = {"xi": scipy_parameters[2], "lambda": scipy_parameters[3], "gamma": scipy_parameters[0], "delta": scipy_parameters[1]}
        return parameters


class JohnsonSU:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.xi_ = self.parameters["xi"]
        self.lambda_ = self.parameters["lambda"]
        self.gamma_ = self.parameters["gamma"]
        self.delta_ = self.parameters["delta"]

    @property
    def name(self):
        return "johnson_su"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xi": 43, "lambda": 382, "gamma": -16, "delta": 54}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.xi_) / self.lambda_
        result = scipy.stats.norm.cdf(self.gamma_ + self.delta_ * numpy.arcsinh(z(x)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.xi_) / self.lambda_
        return (self.delta_ / (self.lambda_ * numpy.sqrt(2 * numpy.pi) * numpy.sqrt(z(x) ** 2 + 1))) * numpy.exp(-(1 / 2) * (self.gamma_ + self.delta_ * numpy.arcsinh(z(x))) ** 2)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.lambda_ * numpy.sinh((scipy.stats.norm.ppf(u) - self.gamma_) / self.delta_) + self.xi_
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
        return self.xi_ - self.lambda_ * numpy.exp(self.delta_**-2 / 2) * numpy.sinh(self.gamma_ / self.delta_)

    @property
    def variance(self) -> float:
        return (self.lambda_**2 / 2) * (numpy.exp(self.delta_**-2) - 1) * (numpy.exp(self.delta_**-2) * numpy.cosh((2 * self.gamma_) / self.delta_) + 1)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return -(self.lambda_**3 * numpy.sqrt(numpy.exp(self.delta_**-2)) * (numpy.exp(self.delta_**-2) - 1) ** 2 * (numpy.exp(self.delta_**-2) * (numpy.exp(self.delta_**-2) + 2) * numpy.sinh(3 * (self.gamma_ / self.delta_)) + 3 * numpy.sinh(self.gamma_ / self.delta_))) / (
            4 * self.standard_deviation**3
        )

    @property
    def kurtosis(self) -> float:
        return (
            self.lambda_**4
            * (numpy.exp(self.delta_**-2) - 1) ** 2
            * (
                numpy.exp(self.delta_**-2) ** 2 * (numpy.exp(self.delta_**-2) ** 4 + 2 * numpy.exp(self.delta_**-2) ** 3 + 3 * numpy.exp(self.delta_**-2) ** 2 - 3) * numpy.cosh(4 * (self.gamma_ / self.delta_))
                + 4 * numpy.exp(self.delta_**-2) ** 2 * (numpy.exp(self.delta_**-2) + 2) * numpy.cosh(2 * (self.gamma_ / self.delta_))
                + 3 * (2 * numpy.exp(self.delta_**-2) + 1)
            )
        ) / (8 * self.standard_deviation**4)

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
        v1 = self.delta_ > 0
        v2 = self.lambda_ > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            xi_, lambda_, gamma_, delta_ = initial_solution
            w = numpy.exp(1 / delta_**2)
            omega = gamma_ / delta_
            A = w**2 * (w**4 + 2 * w**3 + 3 * w**2 - 3) * numpy.cosh(4 * omega)
            B = 4 * w**2 * (w + 2) * numpy.cosh(2 * omega)
            C = 3 * (2 * w + 1)
            parametric_mean = xi_ - lambda_ * numpy.sqrt(w) * numpy.sinh(omega)
            parametric_variance = (lambda_**2 / 2) * (w - 1) * (w * numpy.cosh(2 * omega) + 1)
            parametric_kurtosis = ((lambda_**4) * (w - 1) ** 2 * (A + B + C)) / (8 * numpy.sqrt(parametric_variance) ** 4)
            parametric_median = xi_ + lambda_ * numpy.sinh(-omega)
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_kurtosis - continuous_measures.kurtosis
            eq4 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3, eq4)

        bounds = ((-numpy.inf, 1e-5, -numpy.inf, 1e-5), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, 1, 1, 1)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"xi": solution.x[0], "lambda": solution.x[1], "gamma": solution.x[2], "delta": solution.x[3]}
        return parameters


class Kumaraswamy:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "kumaraswamy"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 7, "beta": 5, "min": 11, "max": 19}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        result = 1 - (1 - z(x) ** self.alpha) ** self.beta
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (self.alpha * self.beta) * (z(x) ** (self.alpha - 1)) * ((1 - z(x) ** self.alpha) ** (self.beta - 1)) / (self.max - self.min)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (1 - (1 - u) ** (1 / self.beta)) ** (1 / self.alpha) * (self.max - self.min) + self.min
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.beta * scipy.special.gamma(1 + k / self.alpha) * scipy.special.gamma(self.beta)) / scipy.special.gamma(1 + self.beta + k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.min + (self.max - self.min) * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.max - self.min) ** 2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.min + (self.max - self.min) * ((self.alpha - 1) / (self.alpha * self.beta - 1)) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.min < self.max
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, min_, max_ = initial_solution
            E = lambda r: beta * scipy.special.gamma(1 + r / alpha) * scipy.special.gamma(beta) / scipy.special.gamma(1 + beta + r / alpha)
            parametric_mean = E(1) * (max_ - min_) + min_
            parametric_variance = (E(2) - E(1) ** 2) * (max_ - min_) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            parametric_kurtosis = (E(4) - 4 * E(1) * E(3) + 6 * E(1) ** 2 * E(2) - 3 * E(1) ** 4) / ((E(2) - E(1) ** 2)) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        bounds = ((1e-5, 1e-5, -numpy.inf, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, continuous_measures.min, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1], "min": solution.x[2], "max": solution.x[3]}
        return parameters


class Laplace:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "laplace"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 17, "b": 4}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 0.5 + 0.5 * numpy.sign(x - self.mu) * (1 - numpy.exp(-abs(x - self.mu) / self.b))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 / (2 * self.b)) * numpy.exp(-abs(x - self.mu) / self.b)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu - self.b * numpy.sign(u - 0.5) * numpy.log(1 - 2 * numpy.abs(u - 0.5))
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
        return self.mu

    @property
    def variance(self) -> float:
        return 2 * self.b**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 6

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        b = numpy.sqrt(continuous_measures.variance / 2)
        parameters = {"mu": mu, "b": b}
        return parameters


class Levy:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "levy"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 0, "c": 1}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        y = lambda x: numpy.sqrt(self.c / ((x - self.mu)))
        result = 2 - 2 * scipy.stats.norm.cdf(y(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(self.c / (2 * numpy.pi)) * numpy.exp(-self.c / (2 * (x - self.mu))) / ((x - self.mu) ** 1.5)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.c / scipy.stats.norm.ppf((2 - u) / 2) ** 2
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
        return numpy.inf

    @property
    def variance(self) -> float:
        return numpy.inf

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return None

    @property
    def kurtosis(self) -> float:
        return None

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu + self.c / 3

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.levy.fit(continuous_measures.data_to_fit)
        parameters = {"mu": scipy_parameters[0], "c": scipy_parameters[1]}
        return parameters


class LogGamma:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.c = self.parameters["c"]
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "loggamma"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"c": 2, "mu": 8, "sigma": 4}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        y = lambda x: (x - self.mu) / self.sigma
        result = scipy.special.gammainc(self.c, numpy.exp(y(x)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        y = lambda x: (x - self.mu) / self.sigma
        result = numpy.exp(self.c * y(x) - numpy.exp(y(x)) - scipy.special.gammaln(self.c)) / self.sigma
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * numpy.log(scipy.special.gammaincinv(self.c, u))
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
        return scipy.special.digamma(self.c) * self.sigma + self.mu

    @property
    def variance(self) -> float:
        return scipy.special.polygamma(1, self.c) * self.sigma * self.sigma

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return scipy.special.polygamma(2, self.c) / scipy.special.polygamma(1, self.c) ** 1.5

    @property
    def kurtosis(self) -> float:
        return scipy.special.polygamma(3, self.c) / scipy.special.polygamma(1, self.c) ** 2 + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu + self.sigma * numpy.log(self.c)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.c > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution, data_mean, data_variance, data_skewness):
            c, mu, sigma = initial_solution
            parametric_mean = scipy.special.digamma(c) * sigma + mu
            parametric_variance = scipy.special.polygamma(1, c) * (sigma**2)
            parametric_skewness = scipy.special.polygamma(2, c) / (scipy.special.polygamma(1, c) ** 1.5)
            eq1 = parametric_mean - data_mean
            eq2 = parametric_variance - data_variance
            eq3 = parametric_skewness - data_skewness
            return (eq1, eq2, eq3)

        bounds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, 1)
        args = (continuous_measures.mean, continuous_measures.variance, continuous_measures.skewness)
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"c": solution.x[0], "mu": solution.x[1], "sigma": solution.x[2]}
        return parameters


class Logistic:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "logistic"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 9, "sigma": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.exp(-(t - self.mu) / self.sigma)
        result = 1 / (1 + z(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: numpy.exp(-(t - self.mu) / self.sigma)
        result = z(x) / (self.sigma * (1 + z(x)) ** 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * numpy.log(u / (1 - u))
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
        return self.mu

    @property
    def variance(self) -> float:
        return (self.sigma * self.sigma * numpy.pi * numpy.pi) / 3

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 4.2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        sigma = numpy.sqrt(3 * continuous_measures.variance / (numpy.pi**2))
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class LogLogistic:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "loglogistic"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 5, "beta": 2}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fisk.cdf(x, self.beta, 0, self.alpha)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fisk.pdf(x, self.beta, 0, self.alpha)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fisk.ppf(u, self.beta, 0, self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.alpha**k * ((k * numpy.pi) / self.beta)) / numpy.sin((k * numpy.pi) / self.beta)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.alpha * ((self.beta - 1) / (self.beta + 1)) ** (1 / self.beta)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution, continuous_measures):
            alpha, beta = initial_solution
            E = lambda r: (alpha**r) * (r * numpy.pi / beta) / numpy.sin(r * numpy.pi / beta)
            parametric_mean = E(1)
            parametric_median = alpha
            eq2 = parametric_mean - continuous_measures.mean
            eq1 = parametric_median - continuous_measures.median
            return (eq1, eq2)

        x0 = (continuous_measures.median, continuous_measures.median)
        bounds = ((0, 0), (numpy.inf, numpy.inf))
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
        return parameters


class LogLogistic3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "loglogistic_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"loc": 100, "alpha": 4, "beta": 2}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fisk.cdf(x, self.beta, self.loc, self.alpha)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fisk.pdf(x, self.beta, self.loc, self.alpha)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.fisk.ppf(u, self.beta, self.loc, self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.alpha**k * ((k * numpy.pi) / self.beta)) / numpy.sin((k * numpy.pi) / self.beta)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc + self.alpha * ((self.beta - 1) / (self.beta + 1)) ** (1 / self.beta)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, loc = initial_solution
            E = lambda r: (alpha**r) * (r * numpy.pi / beta) / numpy.sin(r * numpy.pi / beta)
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_median = alpha + loc
            parametric_mode = alpha * ((beta - 1) / (beta + 1)) ** (1 / beta) + loc
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3)

        bounds = ((0, 0, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (continuous_measures.min, continuous_measures.median, continuous_measures.median)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"loc": solution.x[2], "alpha": solution.x[0], "beta": solution.x[1]}
        return parameters


class LogNormal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "lognormal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 2, "sigma": 7}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.norm.cdf((numpy.log(x) - self.mu) / self.sigma)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (1 / (x * self.sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((numpy.log(x) - self.mu) ** 2) / (2 * self.sigma**2)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.exp(self.mu + self.sigma * scipy.stats.norm.ppf(u))
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
        return numpy.exp(self.mu + self.sigma**2 / 2)

    @property
    def variance(self) -> float:
        return (numpy.exp(self.sigma**2) - 1) * numpy.exp(2 * self.mu + self.sigma**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.exp(self.sigma * self.sigma) + 2) * numpy.sqrt(numpy.exp(self.sigma * self.sigma) - 1)

    @property
    def kurtosis(self) -> float:
        return numpy.exp(4 * self.sigma * self.sigma) + 2 * numpy.exp(3 * self.sigma * self.sigma) + 3 * numpy.exp(2 * self.sigma * self.sigma) - 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.exp(self.mu - self.sigma * self.sigma)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.mu > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = numpy.log(continuous_measures.mean**2 / numpy.sqrt(continuous_measures.mean**2 + continuous_measures.variance))
        sigma = numpy.sqrt(numpy.log((continuous_measures.mean**2 + continuous_measures.variance) / (continuous_measures.mean**2)))
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class Maxwell:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "maxwell"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 60, "loc": 100}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.alpha
        result = scipy.special.erf(z(x) / (numpy.sqrt(2))) - numpy.sqrt(2 / numpy.pi) * z(x) * numpy.exp(-z(x) ** 2 / 2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.alpha
        result = 1 / self.alpha * numpy.sqrt(2 / numpy.pi) * z(x) ** 2 * numpy.exp(-z(x) ** 2 / 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.alpha * numpy.sqrt(2 * scipy.special.gammaincinv(1.5, u)) + self.loc
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
        return 2 * numpy.sqrt(2 / numpy.pi) * self.alpha + self.loc

    @property
    def variance(self) -> float:
        return (self.alpha * self.alpha * (3 * numpy.pi - 8)) / numpy.pi

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 * numpy.sqrt(2) * (16 - 5 * numpy.pi)) / (3 * numpy.pi - 8) ** 1.5

    @property
    def kurtosis(self) -> float:
        return (4 * (-96 + 40 * numpy.pi - 3 * numpy.pi * numpy.pi)) / (3 * numpy.pi - 8) ** 2 + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.sqrt(2) * self.alpha + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        alpha = numpy.sqrt(continuous_measures.variance * numpy.pi / (3 * numpy.pi - 8))
        loc = continuous_measures.mean - 2 * alpha * numpy.sqrt(2 / numpy.pi)
        parameters = {"alpha": alpha, "loc": loc}
        return parameters


class McDonald:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "mcdonald"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 3, "b": 4, "c": 2, "min": 5, "max": 15}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        return scipy.special.betainc(self.a, self.b, z(x) ** self.c)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        zx = z(x)
        result = (self.c / scipy.special.beta(self.a, self.b)) * zx ** (self.a * self.c - 1) * (1 - zx**self.c) ** (self.b - 1) / (self.max - self.min)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        z_val = scipy.special.betaincinv(self.a, self.b, u) ** (1 / self.c)
        return self.min + (self.max - self.min) * z_val

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return scipy.special.beta(self.a + k / self.c, self.b) / scipy.special.beta(self.a, self.b)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.min + (self.max - self.min) * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.max - self.min) ** 2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.a * self.c <= 1 or self.b <= 1:
            standard_pdf = lambda z: -(z ** (self.a * self.c - 1) * (1 - z**self.c) ** (self.b - 1))
            res = scipy.optimize.minimize_scalar(standard_pdf, bounds=(1e-6, 1 - 1e-6), method="bounded")
            return self.min + (self.max - self.min) * res.x
        z_mode = ((self.a * self.c - 1) / (self.a * self.c + self.b * self.c - self.c - 1)) ** (1 / self.c)
        return self.min + (self.max - self.min) * z_mode

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a > 0
        v2 = self.b > 0
        v3 = self.c > 0
        v4 = self.min < self.max
        return v1 and v2 and v3 and v4

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, c, min_, max_ = initial_solution
            E = lambda k: scipy.special.beta(a + k / c, b) / scipy.special.beta(a, b)
            e1 = E(1)
            e2 = E(2)
            e3 = E(3)
            e4 = E(4)
            parametric_mean = e1 * (max_ - min_) + min_
            parametric_variance = (e2 - e1**2) * (max_ - min_) ** 2
            parametric_skewness = (e3 - 3 * e2 * e1 + 2 * e1**3) / ((e2 - e1**2)) ** 1.5
            parametric_kurtosis = (e4 - 4 * e1 * e3 + 6 * e1**2 * e2 - 3 * e1**4) / ((e2 - e1**2)) ** 2
            z_med = scipy.special.betaincinv(a, b, 0.5) ** (1 / c)
            parametric_median = min_ + (max_ - min_) * z_med
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            eq5 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3, eq4, eq5)

        bounds = (
            (1e-5, 1e-5, 1e-5, -numpy.inf, continuous_measures.mean),
            (numpy.inf, numpy.inf, numpy.inf, continuous_measures.mean, numpy.inf),
        )
        x0 = (1, 1, 1, continuous_measures.min - 1e-3, continuous_measures.max + 1e-3)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {
            "a": solution.x[0],
            "b": solution.x[1],
            "c": solution.x[2],
            "min": solution.x[3],
            "max": solution.x[4],
        }
        return parameters


class Moyal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "moyal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 19, "sigma": 9}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = scipy.special.erfc(numpy.exp(-0.5 * z(x)) / numpy.sqrt(2))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = numpy.exp(-0.5 * (z(x) + numpy.exp(-z(x)))) / (self.sigma * numpy.sqrt(2 * numpy.pi))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu - self.sigma * numpy.log(scipy.stats.norm.ppf(1 - u / 2) ** 2)
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
        return self.mu + self.sigma * (numpy.log(2) + 0.577215664901532)

    @property
    def variance(self) -> float:
        return (self.sigma * self.sigma * numpy.pi * numpy.pi) / 2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 1.5351415907229

    @property
    def kurtosis(self) -> float:
        return 7

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        sigma = numpy.sqrt(2 * continuous_measures.variance / (numpy.pi * numpy.pi))
        mu = continuous_measures.mean - sigma * (numpy.log(2) + 0.577215664901532)
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class Nakagami:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.m = self.parameters["m"]
        self.omega = self.parameters["omega"]

    @property
    def name(self):
        return "nakagami"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"m": 11, "omega": 27}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.gammainc(self.m, (self.m / self.omega) * x**2)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (2 * self.m**self.m) / (scipy.special.gamma(self.m) * self.omega**self.m) * (x ** (2 * self.m - 1) * numpy.exp(-(self.m / self.omega) * x**2))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(scipy.special.gammaincinv(self.m, u) * (self.omega / self.m))
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
        return (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(self.omega / self.m)

    @property
    def variance(self) -> float:
        return self.omega * (1 - (1 / self.m) * (scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) ** 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m) * (1 - 4 * self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2))) / (
            2 * self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2) ** 1.5
        )

    @property
    def kurtosis(self) -> float:
        return 3 + (-6 * ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 4 * self.m + (8 * self.m - 2) * ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2 - 2 * self.m + 1) / (
            self.m * (1 - ((scipy.special.gamma(self.m + 0.5) / scipy.special.gamma(self.m)) * numpy.sqrt(1 / self.m)) ** 2) ** 2
        )

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return (numpy.sqrt(2) / 2) * numpy.sqrt((self.omega * (2 * self.m - 1)) / self.m)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.m >= 0.5
        v2 = self.omega > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        E_x2 = numpy.sum(numpy.power(continuous_measures.data, 2)) / continuous_measures.size
        E_x4 = numpy.sum(numpy.power(continuous_measures.data, 4)) / continuous_measures.size
        omega = E_x2
        m = E_x2**2 / (E_x4 - E_x2**2)
        parameters = {"m": m, "omega": omega}
        return parameters


class NonCentralChiSquare:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]

    @property
    def name(self):
        return "non_central_chi_square"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 101, "n": 54}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncx2.cdf(x, self.n, self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncx2.pdf(x, self.n, self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncx2.ppf(u, self.n, self.lambda_)
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
        return self.lambda_ + self.n

    @property
    def variance(self) -> float:
        return 2 * (self.n + 2 * self.lambda_)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2**1.5 * (self.n + 3 * self.lambda_)) / (self.n + 2 * self.lambda_) ** 1.5

    @property
    def kurtosis(self) -> float:
        return 3 + (12 * (self.n + 4 * self.lambda_)) / (self.n + 2 * self.lambda_) ** 2

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
        v1 = self.lambda_ > 0
        v2 = self.n > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        lambda_ = continuous_measures.variance / 2 - continuous_measures.mean
        n = 2 * continuous_measures.mean - continuous_measures.variance / 2
        parameters = {"lambda": lambda_, "n": n}
        return parameters


class NonCentralF:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.lambda_ = self.parameters["lambda"]
        self.n1 = self.parameters["n1"]
        self.n2 = self.parameters["n2"]

    @property
    def name(self):
        return "non_central_f"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 81, "n1": 12, "n2": 72}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncf.cdf(x, self.n1, self.n2, self.lambda_)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncf.pdf(x, self.n1, self.n2, self.lambda_)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.ncf.ppf(u, self.n1, self.n2, self.lambda_)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (self.n2 / self.n1) * ((self.n1 + self.lambda_) / (self.n2 - 2))
        if k == 2:
            return (self.n2 / self.n1) ** 2 * (1 / ((self.n2 - 2) * (self.n2 - 4))) * (self.lambda_**2 + (2 * self.lambda_ + self.n1) * (self.n1 + 2))
        if k == 3:
            return (self.n2 / self.n1) ** 3 * (1 / ((self.n2 - 2) * (self.n2 - 4) * (self.n2 - 6))) * (self.lambda_**3 + 3 * (self.n1 + 4) * self.lambda_**2 + (3 * self.lambda_ + self.n1) * (self.n1 + 4) * (self.n1 + 2))
        if k == 4:
            return (
                (self.n2 / self.n1) ** 4
                * (1 / ((self.n2 - 2) * (self.n2 - 4) * (self.n2 - 6) * (self.n2 - 8)))
                * (self.lambda_**4 + 4 * (self.n1 + 6) * self.lambda_**3 + 6 * (self.n1 + 6) * (self.n1 + 4) * self.lambda_**2 + (4 * self.lambda_ + self.n1) * (self.n1 + 2) * (self.n1 + 4) * (self.n1 + 6))
            )
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

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
        v1 = self.lambda_ > 0
        v2 = self.n1 > 0
        v3 = self.n2 > 0
        return v1 and v2 and v3

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            lambda_, n1, n2 = initial_solution
            E_1 = (n2 / n1) * ((n1 + lambda_) / (n2 - 2))
            E_2 = (n2 / n1) ** 2 * ((lambda_**2 + (2 * lambda_ + n1) * (n1 + 2)) / ((n2 - 2) * (n2 - 4)))
            E_3 = (n2 / n1) ** 3 * ((lambda_**3 + 3 * (n1 + 4) * lambda_**2 + (3 * lambda_ + n1) * (n1 + 2) * (n1 + 4)) / ((n2 - 2) * (n2 - 4) * (n2 - 6)))
            parametric_mean = E_1
            parametric_variance = E_2 - E_1**2
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        bounds = ((0, 0, 0), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, continuous_measures.mean, continuous_measures.mean)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"lambda": solution.x[0], "n1": solution.x[1], "n2": solution.x[2]}
        return parameters


class NonCentralTStudent:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.lambda_ = self.parameters["lambda"]
        self.n = self.parameters["n"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "non_central_t_student"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda": 8, "n": 9, "loc": 474, "scale": 6}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nct.cdf(x, self.n, self.lambda_, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nct.pdf(x, self.n, self.lambda_, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.nct.ppf(u, self.n, self.lambda_, loc=self.loc, scale=self.scale)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (self.lambda_ * numpy.sqrt(self.n / 2) * scipy.special.gamma((self.n - 1) / 2)) / scipy.special.gamma(self.n / 2)
        if k == 2:
            return (self.n * (1 + self.lambda_ * self.lambda_)) / (self.n - 2)
        if k == 3:
            return (self.n**1.5 * numpy.sqrt(2) * scipy.special.gamma((self.n - 3) / 2) * self.lambda_ * (3 + self.lambda_ * self.lambda_)) / (4 * scipy.special.gamma(self.n / 2))
        if k == 4:
            return (self.n * self.n * (self.lambda_**4 + 6 * self.lambda_**2 + 3)) / ((self.n - 2) * (self.n - 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + self.scale * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return self.scale**2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance())

    @property
    def skewness(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ3 = self.central_moments(3)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        central_µ4 = self.central_moments(4)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

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
        v1 = self.n > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            lambda_, n, loc, scale = initial_solution
            E_1 = lambda_ * numpy.sqrt(n / 2) * scipy.special.gamma((n - 1) / 2) / scipy.special.gamma(n / 2)
            E_2 = (1 + lambda_**2) * n / (n - 2)
            E_3 = lambda_ * (3 + lambda_**2) * n**1.5 * numpy.sqrt(2) * scipy.special.gamma((n - 3) / 2) / (4 * scipy.special.gamma(n / 2))
            E_4 = (lambda_**4 + 6 * lambda_**2 + 3) * n**2 / ((n - 2) * (n - 4))
            parametric_mean = E_1 * scale + loc
            parametric_variance = (E_2 - E_1**2) * (scale**2)
            parametric_skewness = (E_3 - 3 * E_2 * E_1 + 2 * E_1**3) / ((E_2 - E_1**2)) ** 1.5
            parametric_kurtosis = (E_4 - 4 * E_1 * E_3 + 6 * E_1**2 * E_2 - 3 * E_1**4) / ((E_2 - E_1**2)) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            eq4 = parametric_kurtosis - continuous_measures.kurtosis
            return (eq1, eq2, eq3, eq4)

        bounds = ((-numpy.inf, 1e-5, -numpy.inf, 1e-5), (numpy.inf, numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 5, continuous_measures.mean, 1)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"lambda": solution.x[0], "n": solution.x[1], "loc": solution.x[2], "scale": solution.x[3]}
        return parameters


class Normal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.mu = self.parameters["mu"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "normal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"mu": 5, "sigma": 3}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.mu) / self.sigma
        result = 0.5 * (1 + scipy.special.erf(z(x) / numpy.sqrt(2)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (1 / (self.sigma * numpy.sqrt(2 * numpy.pi))) * numpy.exp(-(((x - self.mu) ** 2) / (2 * self.sigma**2)))
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.mu + self.sigma * scipy.stats.norm.ppf(u)
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
        return self.mu

    @property
    def variance(self) -> float:
        return self.sigma**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.mu

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        mu = continuous_measures.mean
        sigma = continuous_measures.standard_deviation
        parameters = {"mu": mu, "sigma": sigma}
        return parameters


class ParetoFirstKind:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "pareto_first_kind"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xm": 9, "alpha": 6, "loc": 100}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.pareto.cdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.pareto.pdf(x, self.alpha, loc=self.loc, scale=self.xm)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.xm * (1 - u) ** -(1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.alpha * self.xm**k) / (self.alpha - k)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.xm + self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.pareto.fit(continuous_measures.data_to_fit)
        parameters = {"xm": scipy_parameters[2], "alpha": scipy_parameters[0], "loc": scipy_parameters[1]}
        return parameters


class ParetoSecondKind:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.xm = self.parameters["xm"]
        self.alpha = self.parameters["alpha"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "pareto_second_kind"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"xm": 32, "alpha": 7, "loc": 17}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.lomax.cdf(x, self.alpha, scale=self.xm, loc=self.loc)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.alpha * self.xm**self.alpha) / (((x - self.loc) + self.xm) ** (self.alpha + 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.xm / (1 - u) ** (1 / self.alpha) - self.xm
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.xm**k * scipy.special.gamma(self.alpha - k) * scipy.special.gamma(1 + k)) / scipy.special.gamma(self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.xm > 0
        v2 = self.alpha > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        m = continuous_measures.mean
        v = continuous_measures.variance
        loc = scipy.stats.lomax.fit(continuous_measures.data_to_fit)[1]
        xm = -((m - loc) * ((m - loc) ** 2 + v)) / ((m - loc) ** 2 - v)
        alpha = -(2 * v) / ((m - loc) ** 2 - v)
        parameters = {"xm": xm, "alpha": alpha, "loc": loc}
        return parameters


class Pert:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.alpha1 = (4 * self.b + self.c - 5 * self.a) / (self.c - self.a)
        self.alpha2 = (5 * self.c - self.a - 4 * self.b) / (self.c - self.a)

    @property
    def name(self):
        return "pert"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 63, "b": 513, "c": 970}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.a) / (self.c - self.a)
        result = scipy.special.betainc(self.alpha1, self.alpha2, z(x))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (x - self.a) ** (self.alpha1 - 1) * (self.c - x) ** (self.alpha2 - 1) / (scipy.special.beta(self.alpha1, self.alpha2) * (self.c - self.a) ** (self.alpha1 + self.alpha2 - 1))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a + (self.c - self.a) * scipy.special.betaincinv(self.alpha1, self.alpha2, u)
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
        return (self.a + 4 * self.b + self.c) / 6

    @property
    def variance(self) -> float:
        return ((self.mean - self.a) * (self.c - self.mean)) / 7

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (2 * (self.alpha2 - self.alpha1) * numpy.sqrt(self.alpha1 + self.alpha2 + 1)) / ((self.alpha1 + self.alpha2 + 2) * numpy.sqrt(self.alpha1 * self.alpha2))

    @property
    def kurtosis(self) -> float:
        return (6 * ((self.alpha2 - self.alpha1) ** 2 * (self.alpha1 + self.alpha2 + 1) - self.alpha1 * self.alpha2 * (self.alpha1 + self.alpha2 + 2))) / (self.alpha1 * self.alpha2 * (self.alpha1 + self.alpha2 + 2) * (self.alpha1 + self.alpha2 + 3)) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.b

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.b < self.c
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            a, b, c = initial_solution
            self.alpha1 = (4 * b + c - 5 * a) / (c - a)
            self.alpha2 = (5 * c - a - 4 * b) / (c - a)
            parametric_mean = (a + 4 * b + c) / 6
            parametric_variance = ((parametric_mean - a) * (c - parametric_mean)) / 7
            parametric_median = scipy.special.betaincinv(self.alpha1, self.alpha2, 0.5) * (c - a) + a
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_median - continuous_measures.median
            return (eq1, eq2, eq3)

        bounds = ((-numpy.inf, continuous_measures.min, continuous_measures.mode), (continuous_measures.mode, continuous_measures.max, numpy.inf))
        x0 = (continuous_measures.min, continuous_measures.mode, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"a": solution.x[0], "b": solution.x[1], "c": solution.x[2]}
        parameters["a"] = min(continuous_measures.min - 1e-2, parameters["a"])
        parameters["c"] = max(continuous_measures.max + 1e-2, parameters["c"])
        return parameters


class PowerFunction:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "power_function"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 11, "a": -13, "b": 99}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return ((x - self.a) / (self.b - self.a)) ** self.alpha

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return self.alpha * ((x - self.a) ** (self.alpha - 1)) / ((self.b - self.a) ** self.alpha)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = u ** (1 / self.alpha) * (self.b - self.a) + self.a
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (self.a + self.b * self.alpha) / (self.alpha + 1)
        if k == 2:
            return (2 * self.a**2 + 2 * self.alpha * self.a * self.b + self.alpha * (self.alpha + 1) * self.b**2) / ((self.alpha + 1) * (self.alpha + 2))
        if k == 3:
            return (6 * self.a**3 + 6 * self.a**2 * self.b * self.alpha + 3 * self.a * self.b**2 * self.alpha * (1 + self.alpha) + self.b**3 * self.alpha * (1 + self.alpha) * (2 + self.alpha)) / ((1 + self.alpha) * (2 + self.alpha) * (3 + self.alpha))
        if k == 4:
            return (
                24 * self.a**4 + 24 * self.alpha * self.a**3 * self.b + 12 * self.alpha * (self.alpha + 1) * self.a**2 * self.b**2 + 4 * self.alpha * (self.alpha + 1) * (self.alpha + 2) * self.a * self.b**3 + self.alpha * (self.alpha + 1) * (self.alpha + 2) * (self.alpha + 3) * self.b**4
            ) / ((self.alpha + 1) * (self.alpha + 2) * (self.alpha + 3) * (self.alpha + 4))
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return numpy.max([self.a, self.b])

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.b > self.a
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, a, b = initial_solution
            E1 = (a + b * alpha) / (1 + alpha)
            E2 = (2 * a**2 + 2 * a * b * alpha + b**2 * alpha * (1 + alpha)) / ((1 + alpha) * (2 + alpha))
            E3 = (6 * a**3 + 6 * a**2 * b * alpha + 3 * a * b**2 * alpha * (1 + alpha) + b**3 * alpha * (1 + alpha) * (2 + alpha)) / ((1 + alpha) * (2 + alpha) * (3 + alpha))
            parametric_mean = E1
            parametric_variance = E2 - E1**2
            parametric_skewness = (E3 - 3 * E2 * E1 + 2 * E1**3) / ((E2 - E1**2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        bounds = ((0, -numpy.inf, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, continuous_measures.max)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "a": solution.x[1], "b": continuous_measures.max + 1e-3}
        return parameters


class Rayleigh:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.gamma = self.parameters["gamma"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "rayleigh"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"gamma": 10, "sigma": 2}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.gamma) / self.sigma
        return 1 - numpy.exp(-0.5 * (z(x) ** 2))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.gamma) / self.sigma
        return z(x) * numpy.exp(-0.5 * (z(x) ** 2)) / self.sigma

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sqrt(-2 * numpy.log(1 - u)) * self.sigma + self.gamma
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
        return self.sigma * numpy.sqrt(numpy.pi / 2) + self.gamma

    @property
    def variance(self) -> float:
        return self.sigma * self.sigma * (2 - numpy.pi / 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0.6311

    @property
    def kurtosis(self) -> float:
        return (24 * numpy.pi - 6 * numpy.pi * numpy.pi - 16) / ((4 - numpy.pi) * (4 - numpy.pi)) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.gamma + self.sigma

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.sigma > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        sigma = numpy.sqrt(continuous_measures.variance * 2 / (4 - numpy.pi))
        gamma = continuous_measures.mean - sigma * numpy.sqrt(numpy.pi / 2)
        parameters = {"gamma": gamma, "sigma": sigma}
        return parameters


class Reciprocal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]

    @property
    def name(self):
        return "reciprocal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 20, "b": 99}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (numpy.log(x) - numpy.log(self.a)) / (numpy.log(self.b) - numpy.log(self.a))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 / (x * (numpy.log(self.b) - numpy.log(self.a)))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.exp(u * (numpy.log(self.b) - numpy.log(self.a)) + numpy.log(self.a))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (self.b**k - self.a**k) / (k * (numpy.log(self.b) - numpy.log(self.a)))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.a

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.b > self.a
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-8
        b = continuous_measures.max + 1e-8
        parameters = {"a": a, "b": b}
        return parameters


class Rice:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.v = self.parameters["v"]
        self.sigma = self.parameters["sigma"]

    @property
    def name(self):
        return "rice"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"v": 4, "sigma": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.rice.cdf(x, self.v / self.sigma, scale=self.sigma)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.rice.pdf(x, self.v / self.sigma, scale=self.sigma)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.rice.ppf(u, self.v / self.sigma, scale=self.sigma)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        if k == 1:
            return (
                self.sigma
                * numpy.sqrt(numpy.pi / 2)
                * numpy.exp((-self.v * self.v) / (2 * self.sigma * self.sigma) / 2)
                * ((1 - (-self.v * self.v) / (2 * self.sigma * self.sigma)) * scipy.special.iv(0, (-self.v * self.v) / (4 * self.sigma * self.sigma)) + ((-self.v * self.v) / (2 * self.sigma * self.sigma)) * scipy.special.iv(1, (-self.v * self.v) / (4 * self.sigma * self.sigma)))
            )
        if k == 2:
            return 2 * self.sigma * self.sigma + self.v * self.v
        if k == 3:
            return (
                3
                * self.sigma**3
                * numpy.sqrt(numpy.pi / 2)
                * numpy.exp((-self.v * self.v) / (2 * self.sigma * self.sigma) / 2)
                * (
                    (2 * ((-self.v * self.v) / (2 * self.sigma * self.sigma)) ** 2 - 6 * ((-self.v * self.v) / (2 * self.sigma * self.sigma)) + 3) * scipy.special.iv(0, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                    - 2 * ((-self.v * self.v) / (2 * self.sigma * self.sigma) - 2) * ((-self.v * self.v) / (2 * self.sigma * self.sigma)) * scipy.special.iv(1, (-self.v * self.v) / (4 * self.sigma * self.sigma))
                )
            ) / 3
        if k == 4:
            return 8 * self.sigma**4 + 8 * self.sigma * self.sigma * self.v * self.v + self.v**4
        return None

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

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
        v1 = self.v > 0
        v2 = self.sigma > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            v, sigma = initial_solution
            E = lambda k: sigma**k * 2 ** (k / 2) * scipy.special.gamma(1 + k / 2) * scipy.special.eval_laguerre(k / 2, -v * v / (2 * sigma * sigma))
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        bounds = ((0, 0), (numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, numpy.sqrt(continuous_measures.variance))
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"v": solution.x[0], "sigma": solution.x[1]}
        return parameters


class Semicircular:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.loc = self.parameters["loc"]
        self.R = self.parameters["R"]

    @property
    def name(self):
        return "semicircular"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"loc": 19, "R": 5}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: t - self.loc
        result = 0.5 + z(x) * numpy.sqrt(self.R**2 - z(x) ** 2) / (numpy.pi * self.R**2) + numpy.arcsin(z(x) / self.R) / numpy.pi
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: t - self.loc
        result = 2 * numpy.sqrt(self.R**2 - z(x) ** 2) / (numpy.pi * self.R**2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.R * (2 * scipy.special.betaincinv(1.5, 1.5, u) - 1)
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
        return self.loc

    @property
    def variance(self) -> float:
        return (self.R * self.R) / 4

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 2

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.R > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        loc = continuous_measures.mean
        R = numpy.sqrt(4 * continuous_measures.variance)
        d1 = (loc - R) - continuous_measures.min
        d2 = continuous_measures.max - (loc + R)
        delta = max(max(d1, 0), max(d2, 0)) + 1e-2
        R = R + delta
        parameters = {"loc": loc, "R": R}
        return parameters


class ToppLeone:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "topp_leone"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 3, "min": 5, "max": 17}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (z(x) * (2 - z(x))) ** self.alpha

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (2 * self.alpha * (1 - z(x)) * (z(x) * (2 - z(x))) ** (self.alpha - 1)) / (self.max - self.min)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.min + (self.max - self.min) * (1 - numpy.sqrt(1 - u ** (1 / self.alpha)))
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        result = 0
        for i in range(k + 1):
            result += scipy.special.comb(k, i) * ((-1) ** i) * self.alpha * scipy.special.beta(self.alpha, 1 + i / 2)
        return result

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.min + (self.max - self.min) * µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.max - self.min) ** 2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.alpha <= 1:
            return self.min
        z_mode = 1 - numpy.sqrt((self.alpha - 1) / (2 * self.alpha - 1))
        return self.min + (self.max - self.min) * z_mode

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.min < self.max
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, min_, max_ = initial_solution
            E = lambda k: sum(scipy.special.comb(k, i) * ((-1) ** i) * alpha * scipy.special.beta(alpha, 1 + i / 2) for i in range(k + 1))
            parametric_mean = E(1) * (max_ - min_) + min_
            parametric_variance = (E(2) - E(1) ** 2) * (max_ - min_) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        bounds = ((1e-5, -numpy.inf, continuous_measures.mean), (numpy.inf, continuous_measures.mean, numpy.inf))
        x0 = (1, continuous_measures.min - 1e-3, continuous_measures.max + 1e-3)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "min": solution.x[1], "max": solution.x[2]}
        return parameters


class Trapezoidal:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]
        self.d = self.parameters["d"]

    @property
    def name(self):
        return "trapezoidal"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 110, "b": 267, "c": 741, "d": 980}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.trapezoid.cdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.trapezoid.pdf(x, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.trapezoid.ppf(u, (self.b - self.a) / (self.d - self.a), (self.c - self.a) / (self.d - self.a), loc=self.a, scale=self.d - self.a)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return (2 / (self.d + self.c - self.b - self.a)) * (1 / ((k + 1) * (k + 2))) * ((self.d ** (k + 2) - self.c ** (k + 2)) / (self.d - self.c) - (self.b ** (k + 2) - self.a ** (k + 2)) / (self.b - self.a))

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

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
        v1 = self.a < self.b < self.c < self.d
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution, continuous_measures, a, d):
            b, c = initial_solution
            parametric_mean = (1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))
            parametric_variance = (1 / (6 * (d + c - a - b))) * ((d**4 - c**4) / (d - c) - (b**4 - a**4) / (b - a)) - ((1 / (3 * (d + c - a - b))) * ((d**3 - c**3) / (d - c) - (b**3 - a**3) / (b - a))) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        a = continuous_measures.min - 1e-3
        d = continuous_measures.max + 1e-3
        x0 = [(d + a) * 0.25, (d + a) * 0.75]
        bounds = ((a, a), (d, d))
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=([continuous_measures, a, d]))
        parameters = {"a": a, "b": solution.x[0], "c": solution.x[1], "d": d}
        return parameters


class Triangular:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.a = self.parameters["a"]
        self.b = self.parameters["b"]
        self.c = self.parameters["c"]

    @property
    def name(self):
        return "triangular"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 104, "b": 988, "c": 183}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.triang.cdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.triang.pdf(x, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.triang.ppf(u, (self.c - self.a) / (self.b - self.a), loc=self.a, scale=self.b - self.a)
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
        return (self.a + self.b + self.c) / 3

    @property
    def variance(self) -> float:
        return (self.a**2 + self.b**2 + self.c**2 - self.a * self.b - self.a * self.c - self.b * self.c) / 18

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return (numpy.sqrt(2) * (self.a + self.b - 2 * self.c) * (2 * self.a - self.b - self.c) * (self.a - 2 * self.b + self.c)) / (5 * (self.a**2 + self.b**2 + self.c**2 - self.a * self.b - self.a * self.c - self.b * self.c) ** (3 / 2))

    @property
    def kurtosis(self) -> float:
        return 3 - 3 / 5

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.c

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.a < self.c < self.b
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-3
        b = continuous_measures.max + 1e-3
        c = 3 * continuous_measures.mean - a - b
        parameters = {"a": a, "b": b, "c": c}
        return parameters


class TStudent:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.df = self.parameters["df"]

    @property
    def name(self):
        return "t_student"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df": 8}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.special.betainc(self.df / 2, self.df / 2, (x + numpy.sqrt(x * x + self.df)) / (2 * numpy.sqrt(x * x + self.df)))
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = (1 / (numpy.sqrt(self.df) * scipy.special.beta(0.5, self.df / 2))) * (1 + x * x / self.df) ** (-(self.df + 1) / 2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = numpy.sign(u - 0.5) * numpy.sqrt(self.df * (1 - scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u]))) / scipy.special.betaincinv(self.df / 2, 0.5, 2 * numpy.min([u, 1 - u])))
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
        return 0

    @property
    def variance(self) -> float:
        return self.df / (self.df - 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 6 / (self.df - 4) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return 0

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        df = 2 * continuous_measures.variance / (continuous_measures.variance - 1)
        parameters = {"df": df}
        return parameters


class TStudent3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.df = self.parameters["df"]
        self.loc = self.parameters["loc"]
        self.scale = self.parameters["scale"]

    @property
    def name(self):
        return "t_student_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"df": 15, "loc": 100, "scale": 3}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.t.cdf(x, self.df, loc=self.loc, scale=self.scale)
        return result

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        result = scipy.stats.t.pdf(x, self.df, loc=self.loc, scale=self.scale)
        return result

    def ppf(self, u):
        result = scipy.stats.t.ppf(u, self.df, loc=self.loc, scale=self.scale)
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
        return self.loc

    @property
    def variance(self) -> float:
        return (self.scale * self.scale * self.df) / (self.df - 2)

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 6 / (self.df - 4) + 3

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        return self.loc

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.df > 0
        v2 = self.scale > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        scipy_parameters = scipy.stats.t.fit(continuous_measures.data_to_fit)
        parameters = {"df": scipy_parameters[0], "loc": scipy_parameters[1], "scale": scipy_parameters[2]}
        return parameters


class Uniform:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
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
        return {"a": 50, "b": 299}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (x - self.a) / (self.b - self.a)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        pdf_value = 1 / (self.b - self.a)
        if isinstance(x, numpy.ndarray):
            return numpy.full_like(x, pdf_value)
        else:
            return pdf_value

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.a + u * (self.b - self.a)
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
        return (self.b - self.a) ** 2 / 12

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        return 0

    @property
    def kurtosis(self) -> float:
        return 3 - 6 / 5

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
        return v1

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        a = continuous_measures.min - 1e-8
        b = continuous_measures.max + 1e-8
        parameters = {"a": a, "b": b}
        return parameters


class Weibull:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]

    @property
    def name(self):
        return "weibull"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 7, "beta": 9}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return 1 - numpy.exp(-((x / self.beta) ** self.alpha))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        return (self.alpha / self.beta) * ((x / self.beta) ** (self.alpha - 1)) * numpy.exp(-((x / self.beta) ** self.alpha))

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.beta * (-numpy.log(1 - u)) ** (1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * scipy.special.gamma(1 + k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.alpha <= 1:
            return 0
        return self.beta * ((self.alpha - 1) / self.alpha) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta = initial_solution
            E = lambda k: (beta**k) * scipy.special.gamma(1 + k / alpha)
            parametric_mean = E(1)
            parametric_variance = E(2) - E(1) ** 2
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            return (eq1, eq2)

        bounds = ((1e-5, 1e-5), (numpy.inf, numpy.inf))
        x0 = (continuous_measures.mean, continuous_measures.mean)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "beta": solution.x[1]}
        return parameters


class Weibull3P:
    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError("You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True.")
        if continuous_measures != None:
            self.parameters = self.get_parameters(continuous_measures=continuous_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example
        self.alpha = self.parameters["alpha"]
        self.beta = self.parameters["beta"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "weibull_3p"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 5, "loc": 99, "beta": 3}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.beta
        return 1 - numpy.exp(-(z(x) ** self.alpha))

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        z = lambda t: (t - self.loc) / self.beta
        return (self.alpha / self.beta) * (z(x) ** (self.alpha - 1)) * numpy.exp(-z(x) ** self.alpha)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        result = self.loc + self.beta * (-numpy.log(1 - u)) ** (1 / self.alpha)
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        return self.beta**k * scipy.special.gamma(1 + k / self.alpha)

    def central_moments(self, k: int) -> float | None:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        µ3 = self.non_central_moments(3)
        µ4 = self.non_central_moments(4)
        if k == 1:
            return 0
        if k == 2:
            return µ2 - µ1**2
        if k == 3:
            return µ3 - 3 * µ1 * µ2 + 2 * µ1**3
        if k == 4:
            return µ4 - 4 * µ1 * µ3 + 6 * µ1**2 * µ2 - 3 * µ1**4
        return None

    @property
    def mean(self) -> float:
        µ1 = self.non_central_moments(1)
        return self.loc + µ1

    @property
    def variance(self) -> float:
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return µ2 - µ1**2

    @property
    def standard_deviation(self) -> float:
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        central_µ3 = self.central_moments(3)
        return central_µ3 / self.standard_deviation**3

    @property
    def kurtosis(self) -> float:
        central_µ4 = self.central_moments(4)
        return central_µ4 / self.standard_deviation**4

    @property
    def median(self) -> float:
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        if self.alpha <= 1:
            return 0
        return self.loc + self.beta * ((self.alpha - 1) / self.alpha) ** (1 / self.alpha)

    @property
    def num_parameters(self) -> int:
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        v1 = self.alpha > 0
        v2 = self.beta > 0
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        def equations(initial_solution: tuple[float], continuous_measures) -> tuple[float]:
            alpha, beta, loc = initial_solution
            E = lambda k: (beta**k) * scipy.special.gamma(1 + k / alpha)
            parametric_mean = E(1) + loc
            parametric_variance = E(2) - E(1) ** 2
            parametric_skewness = (E(3) - 3 * E(2) * E(1) + 2 * E(1) ** 3) / ((E(2) - E(1) ** 2)) ** 1.5
            eq1 = parametric_mean - continuous_measures.mean
            eq2 = parametric_variance - continuous_measures.variance
            eq3 = parametric_skewness - continuous_measures.skewness
            return (eq1, eq2, eq3)

        bounds = ((1e-5, 1e-5, -numpy.inf), (numpy.inf, numpy.inf, numpy.inf))
        x0 = (1, 1, continuous_measures.mean)
        args = [continuous_measures]
        solution = scipy.optimize.least_squares(equations, x0=x0, bounds=bounds, args=args)
        parameters = {"alpha": solution.x[0], "loc": solution.x[2], "beta": solution.x[1]}
        return parameters


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


class ContinuousMeasures:
    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        num_bins: int | None = None,
        confidence_level: float = 0.95,
        subsample_size: int | None = None,
        subsample_estimation_size: int | None = None,
    ):
        self.data = numpy.sort(data) if subsample_size == None else numpy.sort(numpy.random.choice(data, size=subsample_size, replace=False))
        self.data_unique = numpy.unique(self.data)
        self.size = self.data.size
        self.data_to_fit = self.data if subsample_estimation_size == None else numpy.random.choice(self.data, size=min(self.size, subsample_estimation_size), replace=False)
        self.min = self.data[0]
        self.max = self.data[-1]
        self.mean = numpy.mean(self.data)
        self.variance = numpy.var(self.data, ddof=1)
        self.standard_deviation = numpy.sqrt(self.variance)
        self.skewness = scipy.stats.skew(self.data, bias=False)
        self.kurtosis = scipy.stats.kurtosis(self.data, fisher=False, bias=False)
        self.median = numpy.median(self.data)
        self.mode = self.calculate_mode()
        self.num_bins = num_bins if num_bins != None else len(numpy.histogram_bin_edges(self.data, bins="doane"))
        self.absolutes_frequencies, self.bin_edges = numpy.histogram(self.data, self.num_bins)
        self.densities_frequencies, _ = numpy.histogram(self.data, self.num_bins, density=True)
        self.central_values = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        self.idx_ks = numpy.concatenate([numpy.where(self.data[:-1] != self.data[1:])[0], [self.size - 1]])
        self.Sn_ks_before = (numpy.arange(self.size)) / self.size
        self.Sn_ks_after = (numpy.arange(self.size) + 1) / self.size
        self.confidence_level = confidence_level
        self.critical_value_ks = scipy.stats.kstwo.ppf(self.confidence_level, self.size)
        self.critical_value_ad = self.ad_critical_value(self.confidence_level, self.size)
        self.ecdf_frequencies = numpy.searchsorted(self.data, self.data_unique, side="right") / self.data.size
        self.qq_arr = (numpy.arange(1, self.size + 1) - 0.5) / self.size

    def __str__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def __repr__(self) -> str:
        return str({"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode})

    def get_dict(self) -> str:
        return {"size": self.size, "mean": self.mean, "variance": self.variance, "skewness": self.skewness, "kurtosis": self.kurtosis, "median": self.median, "mode": self.mode}

    def calculate_mode(self) -> float:
        distribution = scipy.stats.gaussian_kde(self.data)
        x = numpy.linspace(self.min, self.max, 10000)
        y = distribution.pdf(x)
        return x[numpy.argmax(y)]

    def critical_value_chi2(self, freedom_degrees: int):
        return scipy.stats.chi2.ppf(self.confidence_level, freedom_degrees)

    def adinf(self, z: float):
        if z < 2:
            return (z**-0.5) * numpy.exp(-1.2337141 / z) * (2.00012 + (0.247105 - (0.0649821 - (0.0347962 - (0.011672 - 0.00168691 * z) * z) * z) * z) * z)
        return numpy.exp(-numpy.exp(1.0776 - (2.30695 - (0.43424 - (0.082433 - (0.008056 - 0.0003146 * z) * z) * z) * z) * z))

    def errfix(self, n: float, x: float) -> float:
        def g1(t: float) -> float:
            return numpy.sqrt(t) * (1 - t) * (49 * t - 102)

        def g2(t: float) -> float:
            return -0.00022633 + (6.54034 - (14.6538 - (14.458 - (8.259 - 1.91864 * t) * t) * t) * t) * t

        def g3(t: float) -> float:
            return -130.2137 + (745.2337 - (1705.091 - (1950.646 - (1116.360 - 255.7844 * t) * t) * t) * t) * t

        c = 0.01265 + 0.1757 / n
        if x < c:
            return (0.0037 / (n**3) + 0.00078 / (n**2) + 0.00006 / n) * g1(x / c)
        elif x > c and x < 0.8:
            return (0.04213 / n + 0.01365 / (n**2)) * g2((x - c) / (0.8 - c))
        else:
            return (g3(x)) / n

    def AD(self, n: float, z: float) -> float:
        return self.adinf(z) + self.errfix(n, self.adinf(z))

    def ad_critical_value(self, q: float, n: float) -> float:
        f = lambda x: self.AD(n, x) - q
        root = scipy.optimize.newton(f, 2)
        return root

    def ad_p_value(self, n: float, z: float) -> float:
        return 1 - self.AD(n, z)


def evaluate_continuous_test_chi_square(distribution, continuous_measures):
    N = continuous_measures.size
    freedom_degrees = max(continuous_measures.num_bins - 1 - distribution.num_parameters, 1)
    expected_values = N * (distribution.cdf(continuous_measures.bin_edges[1:]) - distribution.cdf(continuous_measures.bin_edges[:-1]))
    errors = ((continuous_measures.absolutes_frequencies - expected_values) ** 2) / expected_values
    statistic_chi2 = numpy.sum(errors)
    critical_value = continuous_measures.critical_value_chi2(freedom_degrees)
    p_value = 1 - scipy.stats.chi2.cdf(statistic_chi2, freedom_degrees)
    rejected = statistic_chi2 >= critical_value
    result_test_chi2 = {
        "test_statistic": statistic_chi2,
        "critical_value": critical_value,
        "p-value": p_value,
        "rejected": rejected,
    }
    return result_test_chi2


def evaluate_continuous_test_kolmogorov_smirnov(distribution, continuous_measures):
    Fn = distribution.cdf(continuous_measures.data)
    errors_before = numpy.abs(continuous_measures.Sn_ks_before[continuous_measures.idx_ks] - Fn[continuous_measures.idx_ks])
    errors_after = numpy.abs(continuous_measures.Sn_ks_after[continuous_measures.idx_ks] - Fn[continuous_measures.idx_ks])
    statistic_ks = max(numpy.max(errors_before), numpy.max(errors_after))
    critical_value = continuous_measures.critical_value_ks
    p_value = 1 - scipy.stats.kstwo.cdf(statistic_ks, continuous_measures.size)
    rejected = statistic_ks >= critical_value
    result_test_ks = {
        "test_statistic": statistic_ks,
        "critical_value": critical_value,
        "p-value": p_value,
        "rejected": rejected,
    }
    return result_test_ks


def evaluate_continuous_test_anderson_darling(distribution, continuous_measures):
    N = continuous_measures.size
    S = numpy.sum(((2 * (numpy.arange(N) + 1) - 1) / N) * (numpy.log(distribution.cdf(continuous_measures.data)) + numpy.log(1 - distribution.cdf(continuous_measures.data[::-1]))))
    A2 = -N - S
    critical_value = continuous_measures.critical_value_ad
    p_value = continuous_measures.ad_p_value(N, A2)
    rejected = A2 >= critical_value
    result_test_ad = {
        "test_statistic": A2,
        "critical_value": critical_value,
        "p-value": p_value,
        "rejected": rejected,
    }
    return result_test_ad


class PhitterContinuous:
    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        num_bins: int | None = None,
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
            self.distributions_to_fit = list(CONTINUOUS_DISTRIBUTIONS.keys())
        if distributions_to_fit != "all" and exclude_distributions == "any":
            not_distributions_ids = [id_distribution for id_distribution in distributions_to_fit if id_distribution not in CONTINUOUS_DISTRIBUTIONS.keys()]
            if len(not_distributions_ids) > 0:
                raise ValueError(f"The following distributions are not found in the continuous distributions list: {not_distributions_ids}")
            self.distributions_to_fit = distributions_to_fit
        if distributions_to_fit == "all" and exclude_distributions != "any":
            not_distributions_ids = [id_distribution for id_distribution in exclude_distributions if id_distribution not in CONTINUOUS_DISTRIBUTIONS.keys()]
            if len(not_distributions_ids) > 0:
                raise ValueError(f"The following distributions to exclude are not found in the continuous distributions list: {not_distributions_ids}")
            self.distributions_to_fit = [dist for dist in CONTINUOUS_DISTRIBUTIONS.keys() if dist not in exclude_distributions]
        self.continuous_measures = ContinuousMeasures(
            data=data,
            num_bins=num_bins,
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

    def test(self, test_function, label: str, distribution):
        validation_test = False
        try:
            test = test_function(distribution, self.continuous_measures)
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

    def log_likelihood(self, distribution, eps=1e-300) -> float:
        x = self.continuous_measures.data
        p = distribution.pdf(x)
        bad = ~numpy.isfinite(p) | (p <= 0)
        if bad.mean() > 0.01:
            edges = self.continuous_measures.bin_edges
            counts = self.continuous_measures.absolutes_frequencies
            probs = distribution.cdf(edges[1:]) - distribution.cdf(edges[:-1])
            probs = numpy.clip(numpy.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0), eps, 1.0)
            return float(numpy.sum(counts * numpy.log(probs)))
        p = numpy.clip(numpy.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0), eps, None)
        return float(numpy.sum(numpy.log(p)))

    def get_aic_and_bic(self, distribution) -> tuple[float, float]:
        ll = self.log_likelihood(distribution)
        k = distribution.num_parameters
        n = self.continuous_measures.size
        aic = 2.0 * k - 2.0 * ll
        bic = k * numpy.log(n) - 2.0 * ll
        return aic, bic

    def process_distribution(self, id_distribution: str) -> tuple[str, dict, typing.Any] | None:
        distribution_class = CONTINUOUS_DISTRIBUTIONS[id_distribution]
        validate_estimation = True
        sse = 0
        try:
            distribution = distribution_class(continuous_measures=self.continuous_measures)
            pdf_values = distribution.pdf(self.continuous_measures.central_values)
            sse = numpy.sum(numpy.power(self.continuous_measures.densities_frequencies - pdf_values, 2.0))
        except:
            validate_estimation = False
        self.distribution_results = {}
        if validate_estimation and distribution.parameter_restrictions() and not numpy.isnan(sse) and not numpy.isinf(sse) and sse < self.minimum_sse:
            v1 = self.test(evaluate_continuous_test_chi_square, "chi_square", distribution)
            v2 = self.test(evaluate_continuous_test_kolmogorov_smirnov, "kolmogorov_smirnov", distribution)
            v3 = self.test(evaluate_continuous_test_anderson_darling, "anderson_darling", distribution)
            if v1 or v2 or v3:
                aic, bic = self.get_aic_and_bic(distribution)
                self.distribution_results["aic"] = aic
                self.distribution_results["bic"] = bic
                self.distribution_results["sse"] = sse
                self.distribution_results["parameters"] = distribution.parameters
                self.distribution_results["n_test_passed"] = +int(self.distribution_results["chi_square"]["rejected"] == False) + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == False) + int(self.distribution_results["anderson_darling"]["rejected"] == False)
                self.distribution_results["n_test_null"] = +int(self.distribution_results["chi_square"]["rejected"] == None) + int(self.distribution_results["kolmogorov_smirnov"]["rejected"] == None) + int(self.distribution_results["anderson_darling"]["rejected"] == None)
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
        self.sorted_distributions = {
            distribution: results
            for distribution, results, _ in sorted(
                processing_results,
                key=lambda x: (
                    -x[1]["n_test_passed"],
                    x[1].get("bic", numpy.inf),
                    x[1].get("aic", numpy.inf),
                    x[1]["sse"],
                ),
            )
        }
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
        central_values = self.continuous_measures.central_values
        densities_frequencies = self.continuous_measures.densities_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=central_values, y=densities_frequencies, marker_color=plot_bar_color, name="Data", showlegend=True))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=plot_title,
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
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
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.hist(self.continuous_measures.data, density=True, label="Data", bins=self.continuous_measures.num_bins, ec="white", color=self.parse_rgba_color(plot_bar_color))
        plt.title(plot_title)
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def plot_histogram_distributions_pdf_plotly(
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
        central_values = self.continuous_measures.central_values
        densities_frequencies = self.continuous_measures.densities_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=central_values, y=densities_frequencies, marker_color=plot_bar_color, showlegend=False, name="Data"))
        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        for idx, (id_distribution, result) in enumerate(list(self.sorted_distributions.items())[:n_distributions]):
            y_plot = self.distribution_instances[id_distribution].pdf(x_plot)
            distribution_sse = result["sse"]
            is_visible = True if idx + 1 <= n_distributions_visible else "legendonly"
            is_rejected = "✅" if id_distribution in self.not_rejected_distributions else ""
            scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
            scatter_line = dict(color=px.colors.qualitative.G10[idx], width=2) if idx < len(px.colors.qualitative.G10) else dict(width=3)
            try:
                fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", visible=is_visible, name=scatter_name, line=scatter_line))
            except Exception:
                fig.add_trace(go.Scatter(x=x_plot, y=numpy.zeros(len(x_plot)), mode="lines", visible=is_visible, name=scatter_name, line=scatter_line))
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
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            bargap=plot_bargap,
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_histogram_distributions_pdf_matplotlib(
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
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.hist(self.continuous_measures.data, density=True, bins=self.continuous_measures.num_bins, ec="white", color=self.parse_rgba_color(plot_bar_color))
        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        for idx, (id_distribution, result) in enumerate(list(self.sorted_distributions.items())[:n_distributions]):
            y_plot = self.distribution_instances[id_distribution].pdf(x_plot)
            distribution_sse = result["sse"]
            is_rejected = "✓" if id_distribution in self.not_rejected_distributions else ""
            scatter_name = f"{idx+1:02d}. {id_distribution}: {distribution_sse:.4E}{is_rejected}"
            plt.plot(x_plot, y_plot, label=scatter_name, color=(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
        plt.title(f"{plot_title} - PDF DISTRIBUTIONS")
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def plot_distribution_pdf_plotly(
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
        central_values = self.continuous_measures.central_values
        densities_frequencies = self.continuous_measures.densities_frequencies
        fig = go.Figure()
        fig.add_trace(go.Bar(x=central_values, y=densities_frequencies, marker_color=plot_bar_color, showlegend=True, name="Data"))
        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        y_plot = self.distribution_instances[id_distribution].pdf(x_plot)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✅" if id_distribution in self.not_rejected_distributions else ""
        scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
        scatter_line = dict(color=plot_line_color, width=plot_line_width)
        try:
            fig.add_trace(go.Scatter(x=x_plot, y=y_plot, mode="lines", name=scatter_name, line=scatter_line))
        except Exception:
            fig.add_trace(go.Scatter(x=x_plot, y=numpy.zeros(len(x_plot)), mode="lines", name=scatter_name, line=scatter_line))
        fig.update_layout(
            height=plot_height,
            width=plot_width,
            title=f"{plot_title} - PDF {id_distribution.upper().replace('_', ' ')} DISTRIBUTION",
            xaxis_title=plot_xaxis_title,
            yaxis_title=plot_yaxis_title,
            legend_title=plot_legend_title,
            template="ggplot2",
            xaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            yaxis=dict(title_font_size=12, tickfont=dict(size=10)),
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            bargap=plot_bargap,
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_distribution_pdf_matplotlib(
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
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.hist(self.continuous_measures.data, density=True, label="Data", bins=self.continuous_measures.num_bins, ec="white", color=self.parse_rgba_color(plot_bar_color))
        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        y_plot = self.distribution_instances[id_distribution].pdf(x_plot)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✓" if id_distribution in self.not_rejected_distributions else ""
        scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
        try:
            plt.plot(x_plot, y_plot, label=scatter_name, color=self.parse_rgba_color(plot_line_color), linewidth=plot_line_width)
        except Exception:
            plt.plot(x_plot, numpy.zeros(len(x_plot)), label=scatter_name, color=self.parse_rgba_color(plot_line_color), linewidth=plot_line_width)
        plt.title(f"{plot_title} - PDF {id_distribution.upper().replace('_', ' ')} DISTRIBUTION")
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()

    def plot_ecdf_plotly(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_xaxis_min_offset: float,
        plot_xaxis_max_offset: float,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_line_color: str,
        plot_line_width: int,
        plot_line_name: str,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=numpy.repeat(self.continuous_measures.data_unique, 2)[1:-1],
                y=numpy.repeat(self.continuous_measures.ecdf_frequencies, 2)[:-2],
                mode="lines",
                name=plot_line_name,
                line=dict(color=plot_line_color, width=plot_line_width),
                showlegend=True,
            )
        )
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
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            xaxis_range=[self.continuous_measures.min - plot_xaxis_min_offset, self.continuous_measures.max + plot_xaxis_max_offset],
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_ecdf_matplotlib(
        self,
        plot_title: str,
        plot_xaxis_title: str,
        plot_xaxis_min_offset: float,
        plot_xaxis_max_offset: float,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_line_color: str,
        plot_line_width: int,
        plot_line_name: str,
    ):
        plt.style.use("ggplot")
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.plot(
            numpy.repeat(self.continuous_measures.data_unique, 2)[1:-1],
            numpy.repeat(self.continuous_measures.ecdf_frequencies, 2)[:-2],
            label=plot_line_name,
            color=self.parse_rgba_color(plot_line_color),
            linewidth=plot_line_width,
        )
        plt.title(plot_title)
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.xlim([self.continuous_measures.min - plot_xaxis_min_offset, self.continuous_measures.max + plot_xaxis_max_offset])
        plt.show()

    def plot_ecdf_distribution_plotly(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_xaxis_min_offset: float,
        plot_xaxis_max_offset: float,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_empirical_line_color: str,
        plot_empirical_line_width: int,
        plot_empirical_line_name: str,
        plot_distribution_line_color: str,
        plot_distribution_line_width: int,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None,
    ):
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=numpy.repeat(self.continuous_measures.data_unique, 2)[1:-1],
                y=numpy.repeat(self.continuous_measures.ecdf_frequencies, 2)[:-1],
                mode="lines",
                name=plot_empirical_line_name,
                line=dict(color=plot_empirical_line_color, width=plot_empirical_line_width),
                showlegend=True,
            )
        )
        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        y_plot = self.distribution_instances[id_distribution].cdf(x_plot)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✅" if id_distribution in self.not_rejected_distributions else ""
        try:
            fig.add_trace(
                go.Scatter(
                    x=x_plot,
                    y=y_plot,
                    mode="lines",
                    name=f"{id_distribution}: {distribution_sse:.4E}{is_rejected}",
                    line=dict(color=plot_distribution_line_color, width=plot_distribution_line_width),
                )
            )
        except Exception:
            fig.add_trace(go.Scatter(x=x_plot, y=numpy.zeros(len(x_plot)), mode="lines", name=f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"))
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
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
            xaxis_range=[self.continuous_measures.min - plot_xaxis_min_offset, self.continuous_measures.max + plot_xaxis_max_offset],
        )
        fig.show(renderer=plotly_plot_renderer)

    def plot_ecdf_distribution_matplotlib(
        self,
        id_distribution: str,
        plot_title: str,
        plot_xaxis_title: str,
        plot_xaxis_min_offset: float,
        plot_xaxis_max_offset: float,
        plot_yaxis_title: str,
        plot_legend_title: str,
        plot_height: int,
        plot_width: int,
        plot_empirical_line_color: str,
        plot_empirical_line_width: int,
        plot_empirical_line_name: str,
        plot_distribution_line_color: str,
        plot_distribution_line_width: int,
    ):
        if id_distribution not in self.distribution_instances:
            raise Exception(f"{id_distribution} distribution not founded")
        plt.style.use("ggplot")
        plt.figure(figsize=(plot_width / 100, plot_height / 100))
        plt.plot(
            numpy.repeat(self.continuous_measures.data_unique, 2)[1:-1],
            numpy.repeat(self.continuous_measures.ecdf_frequencies, 2)[:-2],
            label=plot_empirical_line_name,
            color=self.parse_rgba_color(plot_empirical_line_color),
            linewidth=plot_empirical_line_width,
        )
        x_plot = numpy.linspace(self.continuous_measures.min, self.continuous_measures.max, 1000)
        y_plot = self.distribution_instances[id_distribution].cdf(x_plot)
        distribution_sse = self.sorted_distributions[id_distribution]["sse"]
        is_rejected = "✓" if id_distribution in self.not_rejected_distributions else ""
        scatter_name = f"{id_distribution}: {distribution_sse:.4E}{is_rejected}"
        try:
            plt.plot(x_plot, y_plot, label=scatter_name, color=self.parse_rgba_color(plot_distribution_line_color), linewidth=plot_distribution_line_width)
        except Exception:
            plt.plot(x_plot, numpy.zeros(len(x_plot)), label=scatter_name, color=self.parse_rgba_color(plot_distribution_line_color), linewidth=plot_distribution_line_width)
        plt.title(plot_title)
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.xlim([self.continuous_measures.min - plot_xaxis_min_offset, self.continuous_measures.max + plot_xaxis_max_offset])
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
        x = self.distribution_instances[id_distribution].ppf(self.continuous_measures.qq_arr)
        y = self.continuous_measures.data
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
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
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
        x = self.distribution_instances[id_distribution].ppf(self.continuous_measures.qq_arr)
        y = self.continuous_measures.data
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
        x = self.distribution_instances[id_distribution].ppf(self.continuous_measures.qq_arr)
        y = self.continuous_measures.data
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
            legend=dict(orientation="v", yanchor="auto", y=1, xanchor="left", font=dict(size=10), title_font_size=10),
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
        x = self.distribution_instances[id_distribution].ppf(self.continuous_measures.qq_arr)
        y = self.continuous_measures.data
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
