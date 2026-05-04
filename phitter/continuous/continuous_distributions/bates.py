import numpy
import scipy.optimize
import scipy.special


class Bates:
    """
    Bates distribution (mean of n independent Uniform(0,1) random variables, scaled to (min, max)).
    - Parameters Bates Distribution: {"n": *, "min": *, "max": *}
    - https://phitter.io/distributions/continuous/bates
    """

    def __init__(
        self,
        parameters: dict[str, int | float] = None,
        continuous_measures=None,
        init_parameters_examples=False,
    ):
        """
        - Initializes the Bates Distribution by either providing a Continuous Measures instance [ContinuousMeasures] or a dictionary with the distribution's parameters.
        - Parameters Bates Distribution: {"n": *, "min": *, "max": *}
        - https://phitter.io/distributions/continuous/bates
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
        """
        Cumulative distribution function (uses integer n internally via Irwin-Hall)
        """
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
        """
        Probability density function (uses integer n internally via Irwin-Hall)
        """
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
        """
        Percent point function computed via numerical inversion of the CDF.
        """
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
        """
        Sample of n elements of ditribution
        """
        if seed:
            numpy.random.seed(seed)
        n_int = self._n_int()
        u = numpy.random.rand(n, n_int).mean(axis=1)
        return self.min + (self.max - self.min) * u

    def non_central_moments(self, k: int) -> float | None:
        """
        Parametric no central moments on standard support (0,1). µ[k] = E[Yᵏ]
        Computed via the central moments about 1/2 using the MGF of Irwin-Hall.
        """
        return None

    def central_moments(self, k: int) -> float | None:
        """
        Parametric central moments on standard support. µ'[k] = E[(Y - 1/2)ᵏ]
        For Bates (mean of n uniforms):
          µ'[2] = 1/(12 n)
          µ'[3] = 0
          µ'[4] = (1/(80 n²)) · (3 + 2/n) · (... ) ; using known result: µ'[4] = (1/80) · (3n + 2) / n³
        """
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
        """
        Parametric mean
        """
        return (self.min + self.max) / 2

    @property
    def variance(self) -> float:
        """
        Parametric variance
        """
        return (self.max - self.min) ** 2 / (12 * self.n)

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
        return 0

    @property
    def kurtosis(self) -> float:
        """
        Parametric kurtosis (non-excess): 3 - 6/(5n)
        """
        return 3 - 6 / (5 * self.n)

    @property
    def median(self) -> float:
        """
        Parametric median
        """
        return (self.min + self.max) / 2

    @property
    def mode(self) -> float:
        """
        Parametric mode
        """
        return (self.min + self.max) / 2

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
        v1 = self.n >= 1
        v2 = self.min < self.max
        return v1 and v2

    def get_parameters(self, continuous_measures) -> dict[str, float | int]:
        """
        Calculate proper parameters of the distribution from sample continuous_measures.
        Uses closed-form method-of-moments relations:
          min + max = 2·mean
          (max - min)² = 12·n·variance
          kurtosis = 3 - 6/(5n)
        Estimate n from kurtosis, then min/max from mean and variance.

        Returns
        =======
        parameters: {"n": *, "min": *, "max": *}
        """
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


if __name__ == "__main__":
    import sys

    sys.path.append("../")
    from continuous_measures import ContinuousMeasures

    def get_data(path: str) -> list[float]:
        sample_distribution_file = open(path, "r")
        data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
        sample_distribution_file.close()
        return data

    path = "../../../distributions_samples/continuous_distributions_sample/sample_bates.txt"
    data = get_data(path)
    continuous_measures = ContinuousMeasures(data)
    distribution = Bates(continuous_measures=continuous_measures)

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
