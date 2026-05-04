import numpy
import scipy.integrate
import scipy.optimize
import scipy.special


class ExponentiatedKumaraswamy:
    """
    Exponentiated Kumaraswamy distribution.
    - CDF on (0,1): F(z) = [1 - (1 - z^alpha)^beta]^lambda
    - Parameters Exponentiated Kumaraswamy Distribution: {"alpha": *, "beta": *, "lambda": *, "min": *, "max": *}
    - https://phitter.io/distributions/continuous/exponentiated_kumaraswamy
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Exponentiated Kumaraswamy Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Exponentiated Kumaraswamy Distribution: {"alpha": *, "beta": *, "lambda": *, "min": *, "max": *}
        - https://phitter.io/distributions/continuous/exponentiated_kumaraswamy
        """
        if continuous_measures is None and parameters is None and init_parameters_examples == False:
            raise ValueError(
                "You must initialize the distribution by providing one of the following: distribution parameters, a Continuous Measures [ContinuousMeasures] instance, or by setting init_parameters_examples to True."
            )
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
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (1 - (1 - z(x) ** self.alpha) ** self.beta) ** self.lambda_

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        zx = z(x)
        result = (
            self.alpha
            * self.beta
            * self.lambda_
            * zx ** (self.alpha - 1)
            * (1 - zx**self.alpha) ** (self.beta - 1)
            * (1 - (1 - zx**self.alpha) ** self.beta) ** (self.lambda_ - 1)
        ) / (self.max - self.min)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.min + (self.max - self.min) * (1 - (1 - u ** (1 / self.lambda_)) ** (1 / self.beta)) ** (
            1 / self.alpha
        )
        return result

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        """
        Sample of n elements of ditribution
        """
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        """
        Parametric no central moments on standard support (0,1). µ[k] = E[Zᵏ]
        Using the series: E[Z^k] = lambda · Σ_{j=0}^∞ (-1)^j · C(k/alpha, j) · B(j/beta + 1, lambda)
        Truncated to 200 terms.
        """
        total = 0.0
        for j in range(200):
            term = (
                ((-1) ** j)
                * scipy.special.binom(k / self.alpha, j)
                * scipy.special.beta(j / self.beta + 1, self.lambda_)
            )
            total += term
            if j > 20 and abs(term) < 1e-16 * max(1.0, abs(total)):
                break
        return self.lambda_ * total

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = ∫(x-µ[k])ᵏ∙f(x) dx
        """
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
        """
        Parametric mean
        """
        µ1 = self.non_central_moments(1)
        return self.min + (self.max - self.min) * µ1

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        return (self.max - self.min) ** 2 * (µ2 - µ1**2)

    @property
    def standard_deviation(self) -> float:
        """
        Parametric standard deviation
        """
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """
        Parametric skewness
        """
        central_µ3 = self.central_moments(3)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ3 / std**3

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis
        """
        central_µ4 = self.central_moments(4)
        µ1 = self.non_central_moments(1)
        µ2 = self.non_central_moments(2)
        std = numpy.sqrt(µ2 - µ1**2)
        return central_µ4 / std**4

    @property
    def median(self) -> float:
        """
        Parametric median
        """
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        """
        Parametric mode computed via numerical optimization on (0,1).
        """
        standard_pdf = lambda z: -(
            self.alpha
            * self.beta
            * self.lambda_
            * z ** (self.alpha - 1)
            * (1 - z**self.alpha) ** (self.beta - 1)
            * (1 - (1 - z**self.alpha) ** self.beta) ** (self.lambda_ - 1)
        )
        res = scipy.optimize.minimize_scalar(standard_pdf, bounds=(1e-6, 1 - 1e-6), method="bounded")
        return self.min + (self.max - self.min) * res.x

    @property
    def num_parameters(self) -> int:
        """
        Number of parameters of the distribution
        """
        return len(self.parameters)

    def parameter_restrictions(self) -> bool:
        """
        Check parameters restrictions
        """
        v1 = self.alpha > 0
        v2 = self.beta > 0
        v3 = self.lambda_ > 0
        v4 = self.min < self.max
        return v1 and v2 and v3 and v4

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        Solves 5 equations (mean, variance, skewness, kurtosis, median) via least_squares.

        Returns
        =======
        parameters: {"alpha": *, "beta": *, "lambda": *, "min": *, "max": *}
        """

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


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    path = "../../../distributions_samples/continuous_distributions_sample/sample_exponentiated_kumaraswamy.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = ExponentiatedKumaraswamy(continuous_measures=continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.parameters}")
    print(
        f"CDF: {distribution.cdf(continuous_measures.mean)} {distribution.cdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}"
    )
    print(
        f"PDF: {distribution.pdf(continuous_measures.mean)} {distribution.pdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}"
    )
    print(
        f"PPF: {distribution.ppf(0.5)} {distribution.ppf(numpy.array([0.5, 0.5]))} - V: {distribution.cdf(distribution.ppf(0.5))}"
    )
    print(f"SAMPLE: {distribution.sample(5)}")
    print(f"\nSTATS")
    print(f"mean: {distribution.mean} - {continuous_measures.mean}")
    print(f"variance: {distribution.variance} - {continuous_measures.variance}")
    print(f"skewness: {distribution.skewness} - {continuous_measures.skewness}")
    print(f"kurtosis: {distribution.kurtosis} - {continuous_measures.kurtosis}")
    print(f"median: {distribution.median} - {continuous_measures.median}")
    print(f"mode: {distribution.mode} - {continuous_measures.mode}")
