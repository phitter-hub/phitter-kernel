import numpy
import scipy.optimize
import scipy.special


class ToppLeone:
    """
    Topp-Leone distribution
    - Parameters Topp-Leone Distribution: {"alpha": *, "min": *, "max": *}
    - https://phitter.io/distributions/continuous/topp_leone
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Topp-Leone Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Topp-Leone Distribution: {"alpha": *, "min": *, "max": *}
        - https://phitter.io/distributions/continuous/topp_leone
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
        self.min = self.parameters["min"]
        self.max = self.parameters["max"]

    @property
    def name(self):
        return "topp_leone"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"alpha": 3, "min": 5, "max": 17}

    def cdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (z(x) * (2 - z(x))) ** self.alpha

    def pdf(self, x: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability density function
        """
        z = lambda t: (t - self.min) / (self.max - self.min)
        return (2 * self.alpha * (1 - z(x)) * (z(x) * (2 - z(x))) ** (self.alpha - 1)) / (self.max - self.min)

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = self.min + (self.max - self.min) * (1 - numpy.sqrt(1 - u ** (1 / self.alpha)))
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
        Parametric no central moments on standard support (0,1). µ[k] = E[Zᵏ] = ∫zᵏ∙f(z) dz
        Uses binomial expansion: Z = 1 - sqrt(1 - V), V ~ Beta(alpha, 1)
        E[Zᵏ] = Σ C(k,i) (-1)ⁱ E[(1-V)^(i/2)]  with E[(1-V)^(i/2)] = alpha·B(alpha, 1 + i/2)
        """
        result = 0
        for i in range(k + 1):
            result += scipy.special.comb(k, i) * ((-1) ** i) * self.alpha * scipy.special.beta(self.alpha, 1 + i / 2)
        return result

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
        Parametric mode
        """
        if self.alpha <= 1:
            return self.min
        z_mode = 1 - numpy.sqrt((self.alpha - 1) / (2 * self.alpha - 1))
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
        v1 = self.alpha > 0
        v2 = self.min < self.max
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        The parameters are calculated by solving the equations of the measures expected
        for this distribution.The number of equations to consider is equal to the number
        of parameters.

        Parameters
        ==========
        continuous_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"alpha": *, "min": *, "max": *}
        """

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


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    path = "../../../distributions_samples/continuous_distributions_sample/sample_topp_leone.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = ToppLeone(continuous_measures=continuous_measures)

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
