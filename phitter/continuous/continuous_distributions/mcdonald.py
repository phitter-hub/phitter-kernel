import numpy
import scipy.optimize
import scipy.special
import scipy.stats


class McDonald:
    """
    McDonald distribution (also known as Beta Power distribution).
    - On (0,1): F(z) = I_{z^c}(a, b)  and  f(z) = (c / B(a, b)) · z^(a·c - 1) · (1 - z^c)^(b - 1)
    - If X^c ~ Beta(a, b) then X ~ McDonald(a, b, c). Reduces to Beta when c=1 and Kumaraswamy when a=1.
    - Parameters McDonald Distribution: {"a": *, "b": *, "c": *, "min": *, "max": *}
    - https://phitter.io/distributions/continuous/mcdonald
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the McDonald Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters McDonald Distribution: {"a": *, "b": *, "c": *, "min": *, "max": *}
        - https://phitter.io/distributions/continuous/mcdonald
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
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        return scipy.special.betainc(self.a, self.b, z(x) ** self.c)

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        zx = z(x)
        result = (self.c / scipy.special.beta(self.a, self.b)) * zx ** (self.a * self.c - 1) * (1 - zx**self.c) ** (self.b - 1) / (self.max - self.min)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        z_val = scipy.special.betaincinv(self.a, self.b, u) ** (1 / self.c)
        return self.min + (self.max - self.min) * z_val

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
        Using X = U^(1/c) where U ~ Beta(a,b): E[Z^k] = B(a + k/c, b) / B(a, b)
        """
        return scipy.special.beta(self.a + k / self.c, self.b) / scipy.special.beta(self.a, self.b)

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
        Parametric mode. Obtained from df/dz = 0 on (0,1).
        z_mode = ((a·c - 1) / (a·c + b·c - c - 1))^(1/c)  when a·c > 1 and b > 1
        """
        if self.a * self.c <= 1 or self.b <= 1:
            standard_pdf = lambda z: -(z ** (self.a * self.c - 1) * (1 - z**self.c) ** (self.b - 1))
            res = scipy.optimize.minimize_scalar(standard_pdf, bounds=(1e-6, 1 - 1e-6), method="bounded")
            return self.min + (self.max - self.min) * res.x
        z_mode = ((self.a * self.c - 1) / (self.a * self.c + self.b * self.c - self.c - 1)) ** (1 / self.c)
        return self.min + (self.max - self.min) * z_mode

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
        v1 = self.a > 0
        v2 = self.b > 0
        v3 = self.c > 0
        v4 = self.min < self.max
        return v1 and v2 and v3 and v4

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        Solves 5 equations (mean, variance, skewness, kurtosis, median) via least_squares.

        Returns
        =======
        parameters: {"a": *, "b": *, "c": *, "min": *, "max": *}
        """

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


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    path = "../../../distributions_samples/continuous_distributions_sample/sample_mcdonald.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = McDonald(continuous_measures=continuous_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.parameters}")
    print(f"CDF: {distribution.cdf(continuous_measures.mean)} {distribution.cdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PDF: {distribution.pdf(continuous_measures.mean)} {distribution.pdf(numpy.array([continuous_measures.mean, continuous_measures.mean]))}")
    print(f"PPF: {distribution.ppf(0.5)} {distribution.ppf(numpy.array([0.5, 0.5]))} - V: {distribution.cdf(distribution.ppf(0.5))}")
    print(f"SAMPLE: {distribution.sample(5)}")
    print(f"\nSTATS")
    print(f"mean: {distribution.mean} - {continuous_measures.mean}")
    print(f"variance: {distribution.variance} - {continuous_measures.variance}")
    print(f"skewness: {distribution.skewness} - {continuous_measures.skewness}")
    print(f"kurtosis: {distribution.kurtosis} - {continuous_measures.kurtosis}")
    print(f"median: {distribution.median} - {continuous_measures.median}")
    print(f"mode: {distribution.mode} - {continuous_measures.mode}")
