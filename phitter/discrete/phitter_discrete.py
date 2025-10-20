import concurrent.futures
import random
import re
import sys
import typing

import matplotlib.pyplot as plt
import numpy
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats

sys.path.append("../..")
from phitter.discrete.discrete_distributions import DISCRETE_DISTRIBUTIONS
from phitter.discrete.discrete_measures import DiscreteMeasures
from phitter.discrete.discrete_statistical_tests import evaluate_discrete_test_chi_square
from phitter.discrete.discrete_statistical_tests import evaluate_discrete_test_kolmogorov_smirnov


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
                self.distribution_results["n_test_passed"] = int(self.distribution_results["chi_square"]["rejected"] == False) + int(
                    self.distribution_results["kolmogorov_smirnov"]["rejected"] == False
                )
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
        # plt.hist(self.discrete_measures.data, density=True, label="Data", bins=self.discrete_measures.num_bins, ec="white", color=self.parse_rgba_color(plot_bar_color))
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
        # plt.hist(self.discrete_measures.data, density=True, bins=self.discrete_measures.num_bins, ec="white", color=self.parse_rgba_color(plot_bar_color))
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
        # plt.hist(self.discrete_measures.data, density=True, label="Data", bins=self.discrete_measures.num_bins, ec="white", color=self.parse_rgba_color(plot_bar_color))
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
        plot_empirical_bargap: float,
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
            bargap=plot_empirical_bargap,
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
        # plt.hist(
        #     self.discrete_measures.data,
        #     density=True,
        #     cumulative=True,
        #     label="Data",
        #     ec="white",
        #     color=self.parse_rgba_color(plot_empirical_bar_color),
        # )
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
        plt.title(
            f"{plot_title} {id_distribution.upper().replace('_', ' ')} DISTRIBUTION\nRegression: {linear_regression.intercept:.4f} + x * {linear_regression.slope:.4f} • r = {linear_regression.rvalue:.4f}"
        )
        plt.xlabel(plot_xaxis_title, fontsize=10)
        plt.ylabel(plot_yaxis_title, fontsize=10)
        plt.legend(title=plot_legend_title, fontsize=8, bbox_to_anchor=(1.01, 1.01), loc="upper left")
        plt.show()


if __name__ == "__main__":
    from discrete_distributions import DISCRETE_DISTRIBUTIONS
    from discrete_measures import DiscreteMeasures
    from discrete_statistical_tests import evaluate_discrete_test_chi_square, evaluate_discrete_test_kolmogorov_smirnov

    path = "../../datasets_test/discrete/sample_binomial.txt"
    sample_distribution_file = open(path, "r", encoding="utf-8-sig")
    data = [float(x.replace(",", ".")) for x in sample_distribution_file.read().splitlines()]
    sample_distribution_file.close()

    phitter_discrete = PhitterDiscrete(data)
    phitter_discrete.fit()

    for distribution, results in phitter_discrete.sorted_distributions.items():
        print(f"Distribution: {distribution}, SSE: {results['sse']}, Aprobados: {results['n_test_passed']}")
