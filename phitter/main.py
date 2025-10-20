import sys
import typing

import numpy
import pandas

sys.path.append("../")
from phitter.continuous.phitter_continuous import PhitterContinuous
from phitter.discrete.phitter_discrete import PhitterDiscrete


class Phitter:
    """
    Fit continuous or discrete distributions to a given dataset.
    """

    def __init__(
        self,
        data: list[int | float] | numpy.ndarray,
        fit_type: typing.Literal["continuous", "discrete"] = "continuous",
        num_bins: int | None = None,
        confidence_level=0.95,
        minimum_sse=numpy.inf,
        subsample_size: int | None = None,
        subsample_estimation_size: int | None = None,
        distributions_to_fit: list[str] | typing.Literal["all"] = "all",
        exclude_distributions: list[str] | typing.Literal["any"] = "any",
    ):
        """
        Parameters
        ----------
        data : list[int | float] or numpy.ndarray
            The dataset to fit the distributions to.
        fit_type : str, optional
            The type of fit, either "continuous" or "discrete" (default is "continuous").
        num_bins : int, optional
            The number of bins to use for the histogram if the fit type is "continuous" (default is calculated using Doane's rule).
        confidence_level : float, optional
            The confidence level for the distribution fitting (default is 0.95).
        minimum_sse : float, optional
            The minimum sum of squared errors (default is numpy.inf).
        subsample_size : int, optional
            The size of the subsample to use for fitting (default is None).
        subsample_estimation_size : int, optional
            The size of the subsample used to estimate the parameters of each distribution (default is None).
        distributions_to_fit : list[str] or str, optional
            The list of distributions to fit or "all" to fit all available distributions (default is "all").
        exclude_distributions : list[str] or str, optional
            The list of distributions to exclude or "any" to exclude none (default is "any").

        Raises
        ------
        ValueError
            If the subsample size is larger than the dataset size.
            If the subsample estimation size is larger than the dataset size.
            If the subsample estimation size is larger than the subsample size.
            If both 'distributions_to_fit' and 'exclude_distributions' are specified.
            If any specified distribution is not found in the list of available distributions.
        """
        if subsample_size != None:
            if subsample_size > len(data):
                raise ValueError("The subsample size must be smaller than the size of the dataset.")

        if subsample_estimation_size != None:
            if subsample_estimation_size > len(data):
                raise ValueError("The subsample size used to estimate the parameters of each distribution must be smaller than the size of the dataset.")

        if subsample_estimation_size != None and subsample_size != None:
            if subsample_estimation_size > subsample_size:
                raise ValueError("The subsample size used to estimate the parameters of each distribution must be smaller than the subsample size.")

        self.data = data
        self.fit_type = fit_type
        self.num_bins = num_bins if num_bins != None else numpy.histogram_bin_edges(self.data, bins="doane").size
        self.confidence_level = confidence_level
        self.minimum_sse = minimum_sse
        self.subsample_size = subsample_size
        self.subsample_estimation_size = subsample_estimation_size
        self.distributions_to_fit = distributions_to_fit
        self.exclude_distributions = exclude_distributions

    def fit(self, n_workers: int = 1):
        """
        Fits the appropriate distributions to the data using the specified number of workers.

        Parameters
        ----------
        n_workers : int, optional
            The number of workers to use for fitting the distributions (default is 1).
        """
        if self.fit_type == "continuous":
            self.phitter_continuous = PhitterContinuous(
                data=self.data,
                num_bins=self.num_bins,
                confidence_level=self.confidence_level,
                minimum_sse=self.minimum_sse,
                subsample_size=self.subsample_size,
                subsample_estimation_size=self.subsample_estimation_size,
                distributions_to_fit=self.distributions_to_fit,
                exclude_distributions=self.exclude_distributions,
            )
            self.phitter_continuous.fit(n_workers=n_workers)

        if self.fit_type == "discrete":
            self.phitter_discrete = PhitterDiscrete(
                data=self.data,
                confidence_level=self.confidence_level,
                minimum_sse=self.minimum_sse,
                subsample_size=self.subsample_size,
                subsample_estimation_size=self.subsample_estimation_size,
                distributions_to_fit=self.distributions_to_fit,
                exclude_distributions=self.exclude_distributions,
            )
            self.phitter_discrete.fit(n_workers=n_workers)

    @property
    def best_distribution(self):
        id_distribution = list(self.sorted_distributions.keys())[0]
        return {"id": id_distribution, "parameters": self.sorted_distributions[id_distribution]["parameters"]}

    @property
    def sorted_distributions(self):
        if self.fit_type == "continuous":
            return self.phitter_continuous.sorted_distributions

        if self.fit_type == "discrete":
            return self.phitter_discrete.sorted_distributions

    @property
    def not_rejected_distributions(self):
        if self.fit_type == "continuous":
            return self.phitter_continuous.not_rejected_distributions

        if self.fit_type == "discrete":
            return self.phitter_discrete.not_rejected_distributions

    def get_parameters(self, id_distribution: str) -> dict:
        if id_distribution not in self.sorted_distributions:
            raise Exception(f"{id_distribution} distribution not founded")
        return self.sorted_distributions[id_distribution]["parameters"]

    def get_test_chi_square(self, id_distribution: str) -> dict:
        if id_distribution not in self.sorted_distributions:
            raise Exception(f"{id_distribution} distribution not founded")
        return self.sorted_distributions[id_distribution]["chi_square"]

    def get_test_kolmogorov_smirnov(self, id_distribution: str) -> dict:
        if id_distribution not in self.sorted_distributions:
            raise Exception(f"{id_distribution} distribution not founded")
        return self.sorted_distributions[id_distribution]["kolmogorov_smirnov"]

    def get_test_anderson_darling(self, id_distribution: str) -> dict:
        if id_distribution not in self.sorted_distributions:
            raise Exception(f"{id_distribution} distribution not founded")
        return self.sorted_distributions[id_distribution]["anderson_darling"]

    def get_sse(self, id_distribution: str) -> float:
        if id_distribution not in self.sorted_distributions:
            raise Exception(f"{id_distribution} distribution not founded")
        return self.sorted_distributions[id_distribution]["sse"]

    def get_n_test_passed(self, id_distribution: str) -> int:
        if id_distribution not in self.sorted_distributions:
            raise Exception(f"{id_distribution} distribution not founded")
        return self.sorted_distributions[id_distribution]["n_test_passed"]

    def get_n_test_null(self, id_distribution: str) -> int:
        if id_distribution not in self.sorted_distributions:
            raise Exception(f"{id_distribution} distribution not founded")
        return self.sorted_distributions[id_distribution]["n_test_null"]

    def dict_to_dataframe(self, data: dict[str, dict]) -> pandas.DataFrame:
        flat_data = []
        for distribution, values in data.items():
            is_rejected = "✅" if values["n_test_passed"] >= 1 else "✖️"
            flat_entry = {("distribution", ""): distribution}
            flat_entry[("passed", "")] = is_rejected
            for test, results in values.items():
                if test == "parameters":
                    flat_entry[("parameters", "")] = ", ".join([f"{k}: {v:.4g}" for k, v in results.items()])
                elif isinstance(results, dict):
                    for key, value in results.items():
                        flat_entry[(test, key)] = value
                else:
                    flat_entry[(test, "")] = results
            flat_data.append(flat_entry)

        df = pandas.DataFrame(flat_data)
        df.columns = pandas.MultiIndex.from_tuples(df.columns)
        return df

    def summarize(self, k: int = 20):
        if self.fit_type == "continuous":
            summarize = []
            for id_distribution, info in self.phitter_continuous.sorted_distributions.items():
                summarize.append(
                    {
                        "distribution": id_distribution,
                        # "sse": info["sse"],
                        "aic": info["aic"],
                        "bic": info["bic"],
                        "parameters": ", ".join([f"{k}: {v:.4g}" for k, v in info["parameters"].items()]),
                        "chi_square": "✅" if info["chi_square"]["rejected"] == False else "✖️",
                        "kolmogorov_smirnov": "✅" if info["kolmogorov_smirnov"]["rejected"] == False else "✖️",
                        "anderson_darling": "✅" if info["anderson_darling"]["rejected"] == False else "✖️",
                    }
                )
            return pandas.DataFrame(summarize).head(k)

        if self.fit_type == "discrete":
            summarize = []
            for id_distribution, info in self.phitter_discrete.sorted_distributions.items():
                summarize.append(
                    {
                        "distribution": id_distribution,
                        # "sse": info["sse"],
                        "aic": info["aic"],
                        "bic": info["bic"],
                        "parameters": ", ".join([f"{k}: {v:.4g}" for k, v in info["parameters"].items()]),
                        "chi_square": "✅" if info["chi_square"]["rejected"] == False else "✖️",
                        "kolmogorov_smirnov": "✅" if info["kolmogorov_smirnov"]["rejected"] == False else "✖️",
                    }
                )
            return pandas.DataFrame(summarize).head(k)

    def summarize_info(self, k: int = 10):
        if self.fit_type == "continuous":
            summarize = []
            for id_distribution, info in self.phitter_continuous.sorted_distributions.items():
                summarize.append(
                    {
                        "distribution": id_distribution,
                        # "sse": info["sse"],
                        "aic": info["aic"],
                        "bic": info["bic"],
                        "parameters": info["parameters"],
                        "chi_square": info["chi_square"]["rejected"],
                        "kolmogorov_smirnov": info["kolmogorov_smirnov"]["rejected"],
                        "anderson_darling": info["anderson_darling"]["rejected"],
                    }
                )
            return pandas.DataFrame(summarize).head(k)

        if self.fit_type == "discrete":
            summarize = []
            for id_distribution, info in self.phitter_discrete.sorted_distributions.items():
                summarize.append(
                    {
                        "distribution": id_distribution,
                        # "sse": info["sse"],
                        "aic": info["aic"],
                        "bic": info["bic"],
                        "parameters": info["parameters"],
                        "chi_square": info["chi_square"]["rejected"],
                        "kolmogorov_smirnov": info["kolmogorov_smirnov"]["rejected"],
                    }
                )
            return pandas.DataFrame(summarize).head(k)

    @property
    def df_sorted_distributions(self) -> pandas.DataFrame | None:
        if self.fit_type == "continuous":
            if self.phitter_continuous.sorted_distributions is None:
                return None
            if len(self.phitter_continuous.sorted_distributions) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            # ("sse", ""),
                            ("aic", ""),
                            ("bic", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                            ("anderson_darling", "test_statistic"),
                            ("anderson_darling", "critical_value"),
                            ("anderson_darling", "p_value"),
                            ("anderson_darling", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_continuous.sorted_distributions)
            return df[["distribution", "passed", "aic", "bic", "parameters", "chi_square", "kolmogorov_smirnov", "anderson_darling"]]

        if self.fit_type == "discrete":
            if self.phitter_discrete.sorted_distributions is None:
                return None
            if len(self.phitter_discrete.sorted_distributions) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            # ("sse", ""),
                            ("aic", ""),
                            ("bic", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_discrete.sorted_distributions)
            return df[["distribution", "passed", "aic", "bic", "parameters", "chi_square", "kolmogorov_smirnov"]]

    @property
    def df_not_rejected_distributions(self) -> pandas.DataFrame | None:
        if self.fit_type == "continuous":
            if self.phitter_continuous.not_rejected_distributions is None:
                return None
            if len(self.phitter_continuous.not_rejected_distributions) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            # ("sse", ""),
                            ("aic", ""),
                            ("bic", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                            ("anderson_darling", "test_statistic"),
                            ("anderson_darling", "critical_value"),
                            ("anderson_darling", "p_value"),
                            ("anderson_darling", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_continuous.not_rejected_distributions)
            return df[["distribution", "aic", "bic", "parameters", "chi_square", "kolmogorov_smirnov", "anderson_darling"]]

        if self.fit_type == "discrete":
            if self.phitter_discrete.not_rejected_distributions is None:
                return None
            if len(self.phitter_discrete.not_rejected_distributions) == 0:
                return pandas.DataFrame(
                    columns=pandas.MultiIndex.from_tuples(
                        [
                            ("distribution", ""),
                            ("passed", ""),
                            # ("sse", ""),
                            ("aic", ""),
                            ("bic", ""),
                            ("parameters", ""),
                            ("chi_square", "test_statistic"),
                            ("chi_square", "critical_value"),
                            ("chi_square", "p_value"),
                            ("chi_square", "rejected"),
                            ("kolmogorov_smirnov", "test_statistic"),
                            ("kolmogorov_smirnov", "critical_value"),
                            ("kolmogorov_smirnov", "p_value"),
                            ("kolmogorov_smirnov", "rejected"),
                        ]
                    )
                )
            df = self.dict_to_dataframe(self.phitter_discrete.not_rejected_distributions)
            return df[["distribution", "aic", "bic", "parameters", "chi_square", "kolmogorov_smirnov"]]

    def plot_histogram(
        self,
        plot_title="HISTOGRAM",
        plot_xaxis_title=None,
        plot_yaxis_title=None,
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_bar_color="rgba(128,128,128,1)",
        plot_bargap=0.15,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if plot_engine == "matplotlib":
                self.phitter_continuous.plot_histogram_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                )
            else:
                self.phitter_continuous.plot_histogram_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_histogram_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                )
            else:
                self.phitter_discrete.plot_histogram_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

    def plot_histogram_distributions(
        self,
        n_distributions=10,
        n_distributions_visible=1,
        plot_title="HISTOGRAM",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Probability Density/Mass Function",
        plot_legend_title: str | None = "DISTRIBUTIONS",
        plot_height=400,
        plot_width=600,
        plot_bar_color="rgba(128,128,128,1)",
        plot_bargap=0.15,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if plot_engine == "matplotlib":
                self.phitter_continuous.plot_histogram_distributions_pdf_matplotlib(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                )
            else:
                self.phitter_continuous.plot_histogram_distributions_pdf_plotly(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_histogram_distributions_pmf_matplotlib(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                )
            else:
                self.phitter_discrete.plot_histogram_distributions_pmf_plotly(
                    n_distributions=n_distributions,
                    n_distributions_visible=n_distributions_visible,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

    def plot_distribution(
        self,
        id_distribution: str,
        plot_title="HISTOGRAM",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Probability Density/Mass Function",
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_bar_color="rgba(128,128,128,1)",
        plot_bargap=0.15,
        plot_line_color="rgba(255,0,0,1)",
        plot_line_width=3,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if plot_engine == "matplotlib":
                self.phitter_continuous.plot_distribution_pdf_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                )
            else:
                self.phitter_continuous.plot_distribution_pdf_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_distribution_pmf_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                )
            else:
                self.phitter_discrete.plot_distribution_pmf_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_bargap=plot_bargap,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

    def plot_ecdf(
        self,
        plot_title="EMPIRICAL CUMULATIVE DISTRIBUTION FUNCTION",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Cumulative Distribution Function",
        plot_xaxis_min_offset=0.3,
        plot_xaxis_max_offset=0.3,
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_line_color="rgba(255,0,0,1)",
        plot_line_width=2,
        plot_empirical_bargap=0.15,
        plot_line_name="Empirical Distribution",
        plot_bar_color="rgba(128,128,128,1)",
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.plot_ecdf_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_line_name=plot_line_name,
                )
            else:
                self.phitter_continuous.plot_ecdf_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_line_color=plot_line_color,
                    plot_line_width=plot_line_width,
                    plot_line_name=plot_line_name,
                    plotly_plot_renderer=plotly_plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_ecdf_matplotlib(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                )
            else:
                self.phitter_discrete.plot_ecdf_plotly(
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_bar_color=plot_bar_color,
                    plot_empirical_bargap=plot_empirical_bargap,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

    def plot_ecdf_distribution(
        self,
        id_distribution: str,
        plot_title="ECDF",
        plot_xaxis_title="Domain",
        plot_yaxis_title="Cumulative Distribution Function",
        plot_xaxis_min_offset=0.3,
        plot_xaxis_max_offset=0.3,
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        plot_empirical_line_color="rgba(128,128,128,1)",
        plot_empirical_line_width=4,
        plot_empirical_line_name="Empirical Distribution",
        plot_empirical_bar_color="rgba(128,128,128,1)",
        plot_empirical_bargap=0.15,
        plot_distribution_line_color="rgba(255,0,0,1)",
        plot_distribution_line_width=2,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.plot_ecdf_distribution_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_line_color=plot_empirical_line_color,
                    plot_empirical_line_width=plot_empirical_line_width,
                    plot_empirical_line_name=plot_empirical_line_name,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                )
            else:
                self.phitter_continuous.plot_ecdf_distribution_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_xaxis_min_offset=plot_xaxis_min_offset,
                    plot_xaxis_max_offset=plot_xaxis_max_offset,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_line_color=plot_empirical_line_color,
                    plot_empirical_line_width=plot_empirical_line_width,
                    plot_empirical_line_name=plot_empirical_line_name,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                    plotly_plot_renderer=plotly_plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if plot_engine == "matplotlib":
                self.phitter_discrete.plot_ecdf_distribution_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_bar_color=plot_empirical_bar_color,
                    plot_empirical_bargap=plot_empirical_bargap,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                )
            else:
                self.phitter_discrete.plot_ecdf_distribution_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    plot_empirical_bar_color=plot_empirical_bar_color,
                    plot_empirical_bargap=plot_empirical_bargap,
                    plot_distribution_line_color=plot_distribution_line_color,
                    plot_distribution_line_width=plot_distribution_line_width,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

    def qq_plot(
        self,
        id_distribution: str,
        plot_title="QQ PLOT",
        plot_xaxis_title="Theoretical Quantiles",
        plot_yaxis_title="Sample Quantiles",
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        qq_marker_name="Markers QQ",
        qq_marker_color="rgba(128,128,128,1)",
        qq_marker_size=6,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.qq_plot_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                )
            else:
                self.phitter_continuous.qq_plot_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    plotly_plot_renderer=plotly_plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_discrete.qq_plot_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                )
            else:
                self.phitter_discrete.qq_plot_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    plotly_plot_renderer=plotly_plot_renderer,
                )

    def qq_plot_regression(
        self,
        id_distribution: str,
        plot_title="QQ PLOT",
        plot_xaxis_title="Theoretical Quantiles",
        plot_yaxis_title="Sample Quantiles",
        plot_legend_title: str | None = None,
        plot_height=400,
        plot_width=600,
        qq_marker_name="Markers QQ",
        qq_marker_color="rgba(128,128,128,1)",
        qq_marker_size=6,
        regression_line_name="Regression",
        regression_line_color="rgba(255,0,0,1)",
        regression_line_width=2,
        plotly_plot_renderer: typing.Literal["png", "jpeg", "svg"] | None = None,
        plot_engine: typing.Literal["plotly", "matplotlib"] = "plotly",
    ):
        if self.fit_type == "continuous":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_continuous.qq_plot_regression_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                )
            else:
                self.phitter_continuous.qq_plot_regression_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                    plotly_plot_renderer=plotly_plot_renderer if len(self.data) <= 10000 else "png",
                )

        if self.fit_type == "discrete":
            if len(self.data) > 10000 or plot_engine == "matplotlib":
                self.phitter_discrete.qq_plot_regression_matplotlib(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                )
            else:
                self.phitter_discrete.qq_plot_regression_plotly(
                    id_distribution=id_distribution,
                    plot_title=plot_title,
                    plot_xaxis_title=plot_xaxis_title,
                    plot_yaxis_title=plot_yaxis_title,
                    plot_legend_title=plot_legend_title,
                    plot_height=plot_height,
                    plot_width=plot_width,
                    qq_marker_name=qq_marker_name,
                    qq_marker_color=qq_marker_color,
                    qq_marker_size=qq_marker_size,
                    regression_line_name=regression_line_name,
                    regression_line_color=regression_line_color,
                    regression_line_width=regression_line_width,
                    plotly_plot_renderer=plotly_plot_renderer if len(self.data) <= 10000 else "png",
                )


if __name__ == "__main__":
    path = "../datasets_test/discrete/book2.txt"
    # path = "../datasets_test/continuous/data.txt"
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    data = [float(x.strip().replace(",", ".")) for x in sample_distribution_file.read().splitlines()]

    phitter = Phitter(data=data, fit_type="discrete")
    phitter.fit(n_workers=2)

    print(f"Best Distribution: {phitter.best_distribution}")

    for distribution, results in phitter.sorted_distributions.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
