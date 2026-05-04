import numpy
import scipy.stats


class DiscreteLaplace:
    """
    Discrete Laplace distribution (also called discrete double-exponential).
    - PMF: P(X=k) = tanh(a/2) * exp(-a*|k - loc|), for k ∈ loc + ℤ, a > 0
    - Parameters Discrete Laplace Distribution: {"a": *, "loc": *}
    - https://docs.scipy.org/doc/scipy/tutorial/stats/discrete_dlaplace.html
    - https://phitter.io/distributions/discrete/discrete_laplace
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Discrete Laplace Distribution by either providing a Discrete Measures instance [DiscreteMeasures] or a dictionary with the distribution's parameters.
        - Parameters Discrete Laplace Distribution: {"a": *, "loc": *}
        - https://phitter.io/distributions/discrete/discrete_laplace
        """
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.a = self.parameters["a"]
        self.loc = self.parameters["loc"]

    @property
    def name(self):
        return "discrete_laplace"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"a": 0.6, "loc": 50}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        return scipy.stats.dlaplace.cdf(x, self.a, loc=self.loc)

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability mass function
        """
        return scipy.stats.dlaplace.pmf(x, self.a, loc=self.loc)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        return scipy.stats.dlaplace.ppf(u, self.a, loc=self.loc)

    def sample(self, n: int, seed: int | None = None) -> numpy.ndarray:
        """
        Sample of n elements of ditribution
        """
        if seed:
            numpy.random.seed(seed)
        return self.ppf(numpy.random.rand(n))

    def non_central_moments(self, k: int) -> float | None:
        """
        Parametric no central moments. µ[k] = E[Xᵏ] = Σ xᵏ ∙ P(X=x)
        """
        return None

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments. µ'[k] = E[(X - E[X])ᵏ] = Σ (x-µ)ᵏ ∙ P(X=x)
        """
        return None

    @property
    def mean(self) -> float:
        """
        Parametric mean
        """
        return self.loc

    @property
    def variance(self) -> float:
        """
        Parametric variance: Var = 2·e^(-a) / (1 - e^(-a))^2
        Location-invariant.
        """
        y = numpy.exp(-self.a)
        return 2 * y / (1 - y) ** 2

    @property
    def standard_deviation(self) -> float:
        """
        Parametric standard deviation
        """
        return numpy.sqrt(self.variance)

    @property
    def skewness(self) -> float:
        """
        Parametric skewness (symmetric distribution)
        """
        return 0

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis (non-excess): 5 + cosh(a)
        """
        return 5 + numpy.cosh(self.a)

    @property
    def median(self) -> float:
        """
        Parametric median (symmetric distribution)
        """
        return self.loc

    @property
    def mode(self) -> float:
        """
        Parametric mode (PMF maximum at k=loc)
        """
        return self.loc

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
        return v1

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample discrete_measures.
        Closed-form method-of-moments:
            loc = round(mean)              (integer location, since support is loc + ℤ)
            Var = 2·e^(-a) / (1 - e^(-a))^2
            Letting y = e^(-a) and v = Var, the equation 2y = v(1-y)^2 yields
                y = ((v+1) - sqrt(2v+1)) / v        (root in (0,1))
                a = -ln(y)

        Parameters
        ==========
        discrete_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"a": *, "loc": *}
        """
        loc = int(round(discrete_measures.mean))
        v = max(discrete_measures.variance, 1e-12)
        y = ((v + 1) - numpy.sqrt(2 * v + 1)) / v
        y = min(max(y, 1e-12), 1 - 1e-12)
        a = -numpy.log(y)
        parameters = {"a": a, "loc": loc}
        return parameters


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from discrete_measures import DiscreteMeasures

    def get_data(path: str) -> list[int]:
        sample_distribution_file = open(path, "r")
        data = [int(x) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    path = "../../../distributions_samples/discrete_distributions_sample/sample_discrete_laplace.txt"
    data = get_data(path)
    discrete_measures = DiscreteMeasures(data)
    distribution = DiscreteLaplace(discrete_measures=discrete_measures)

    print(f"{distribution.name} distribution")
    print(f"Parameters: {distribution.parameters}")
    print(f"CDF: {distribution.cdf(int(discrete_measures.mean))} {distribution.cdf(numpy.array([int(discrete_measures.mean), int(discrete_measures.mean)]))}")
    print(f"PMF: {distribution.pmf(int(discrete_measures.mean))} {distribution.pmf(numpy.array([int(discrete_measures.mean), int(discrete_measures.mean)]))}")
    print(f"PPF: {distribution.ppf(0.5)} {distribution.ppf(numpy.array([0.5, 0.5]))} - V: {distribution.cdf(distribution.ppf(0.5))}")
    print(f"SAMPLE: {distribution.sample(5)}")
    print(f"\nSTATS")
    print(f"mean: {distribution.mean} - {discrete_measures.mean}")
    print(f"variance: {distribution.variance} - {discrete_measures.variance}")
    print(f"skewness: {distribution.skewness} - {discrete_measures.skewness}")
    print(f"kurtosis: {distribution.kurtosis} - {discrete_measures.kurtosis}")
    print(f"median: {distribution.median} - {discrete_measures.median}")
    print(f"mode: {distribution.mode} - {discrete_measures.mode}")
