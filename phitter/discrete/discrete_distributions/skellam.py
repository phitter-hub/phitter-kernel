import numpy
import scipy.special
import scipy.stats


class Skellam:
    """
    Skellam distribution. Distribution of the difference X = N1 - N2 of two independent
    Poisson random variables N1 ~ Poisson(lambda1) and N2 ~ Poisson(lambda2).
    - Parameters Skellam Distribution: {"lambda1": *, "lambda2": *}
    - https://phitter.io/distributions/discrete/skellam
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        discrete_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Skellam Distribution by either providing a Discrete Measures instance [DiscreteMeasures] or a dictionary with the distribution's parameters.
        - Parameters Skellam Distribution: {"lambda1": *, "lambda2": *}
        - https://phitter.io/distributions/discrete/skellam
        """
        if discrete_measures is None and parameters is None and init_parameters_examples == False:
            raise Exception("You must initialize the distribution by either providing the Discrete Measures [DiscreteMeasures] instance or a dictionary of the distribution's parameters.")
        if discrete_measures != None:
            self.parameters = self.get_parameters(discrete_measures=discrete_measures)
        if parameters != None:
            self.parameters = parameters
        if init_parameters_examples:
            self.parameters = self.parameters_example

        self.lambda1 = self.parameters["lambda1"]
        self.lambda2 = self.parameters["lambda2"]

    @property
    def name(self):
        return "skellam"

    @property
    def parameters_example(self) -> dict[str, int | float]:
        return {"lambda1": 8, "lambda2": 3}

    def cdf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Cumulative distribution function
        """
        result = scipy.stats.skellam.cdf(x, self.lambda1, self.lambda2)
        return result

    def pmf(self, x: int | numpy.ndarray) -> float | numpy.ndarray:
        """
        Probability mass function.
        P(X=k) = exp(-(lambda1+lambda2)) * (lambda1/lambda2)^(k/2) * I_|k|(2*sqrt(lambda1*lambda2))
        """
        result = scipy.stats.skellam.pmf(x, self.lambda1, self.lambda2)
        return result

    def ppf(self, u: float | numpy.ndarray) -> float | numpy.ndarray:
        """
        Percent point function. Inverse of Cumulative distribution function. If CDF[x] = u => PPF[u] = x
        """
        result = scipy.stats.skellam.ppf(u, self.lambda1, self.lambda2)
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
        return self.lambda1 - self.lambda2

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return self.lambda1 + self.lambda2

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
        return (self.lambda1 - self.lambda2) / (self.lambda1 + self.lambda2) ** 1.5

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis (non-excess)
        """
        return 3 + 1 / (self.lambda1 + self.lambda2)

    @property
    def median(self) -> float:
        """
        Parametric median
        """
        return self.ppf(0.5)

    @property
    def mode(self) -> float:
        """
        Parametric mode. No closed form; computed by comparing the PMF at the two integers
        closest to the mean (lambda1 - lambda2).
        """
        center = self.lambda1 - self.lambda2
        candidates = numpy.array([int(numpy.floor(center)), int(numpy.ceil(center))])
        pmfs = self.pmf(candidates)
        return float(candidates[int(numpy.argmax(pmfs))])

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
        v1 = self.lambda1 > 0
        v2 = self.lambda2 > 0
        return v1 and v2

    def get_parameters(self, discrete_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample discrete_measures.
        Closed-form method-of-moments:
            mean     = lambda1 - lambda2
            variance = lambda1 + lambda2
        =>  lambda1 = (variance + mean) / 2
            lambda2 = (variance - mean) / 2

        Parameters
        ==========
        discrete_measures: MEASUREMESTS
            attributes: mean, std, variance, skewness, kurtosis, median, mode, min, max, size, num_bins, data

        Returns
        =======
        parameters: {"lambda1": *, "lambda2": *}
        """
        lambda1 = (discrete_measures.variance + discrete_measures.mean) / 2
        lambda2 = (discrete_measures.variance - discrete_measures.mean) / 2
        parameters = {"lambda1": lambda1, "lambda2": lambda2}
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

    path = "../../../distributions_samples/discrete_distributions_sample/sample_skellam.txt"
    data = get_data(path)
    discrete_measures = DiscreteMeasures(data)
    distribution = Skellam(discrete_measures=discrete_measures)

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
