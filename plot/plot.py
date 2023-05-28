from typing import List, Optional
import os
import argparse
import matplotlib as mpl
import numpy as np
from parse_results import read_bench_data, BenchmarkData, RunResult, BenchConfig, AlgorithmConstants
from dataclasses import dataclass

# For number formatting.
import locale

locale.setlocale(locale.LC_NUMERIC, 'lt_LT.utf8')
# Remove thousands separator by turning off grouping:
# https://docs.python.org/3/library/locale.html#locale.localeconv
# https://stackoverflow.com/a/51938429
# https://stackoverflow.com/a/67186977
locale._override_localeconv = {'grouping': [locale.CHAR_MAX]}

DEFAULT_PLOT_DIR = "graphs"
ALGOS = ["PACO", "CABC", "qCABC"]
CORE_COUNTS = [1, 2, 4, 6, 8]
TEST_CASES = ["eil51", "eil76", "kroA100", "kroA200", "d198", "lin318"]
EXCHANGE_GENERATIONS = [2, 4, 8, 32]
# <x axis>_<y axis>_<what we are comparing>
PLOT_KINDS = [
    "gens_diff_excg",
    "cores_diff_test_cases",
    "cores_diff_gens",
    "cores_diff_algos",
    "gens_diff_popsizes",
    "cores_diff_popsizes",
    # "relative_times", # does not exist
    "gens_diff_cls",
    "cores_diff_cls"  # neįdomus, turbūt nenaudosiu
]
MAX_GENERATIONS = 4000
# Population is the same size as city count on PACO.
POPULATION_SIZES = [20, 40, 80]
CAPITAL_LS = [2, 3, 4]

# ALGO_TO_FILE_NAME_PART = {"paco": "paco", "cabc": "CABC", "qcabc": "qCABC"}
# ALGO_TO_UPPER = {"paco": "PACO", "cabc": "CABC", "qcabc": "qCABC"}

GENS_START = 499
GENS_STEP = 700

# GENS_NAME = "k"
GENS_NAME = "kartos"

# controls where in the plot the legend is placed
PLOT_LEGEND_LOCATION = "upper right"

DIFF = "percent"

CORE_COUNT_AXIS_LABEL = "branduolių skaičius"
DIFF_FROM_OPTIMAL_AXIS_LABEL = "skirtumas nuo optimalaus, $\%$"
ITERATIONS_AXIS_LABEL = "algoritmo iteracija"

# controls plot image resolution (png)
PLOT_DPI = 220

PLOT_FORMAT = "pgf"
# PLOT_FORMAT = "png"

# pgf plot scale
PLOT_SCALE = 1.0

# For controlling Y axis range (ylim)
Y_TOP = None
Y_BOTTOM = None
# Caps y_top and y_bottom axis values. Y_TOP and Y_BOTTOM
# take precedence over these.
COND_Y_TOP = 1.6
COND_Y_BOTTOM = 0.9

# height / width
PLOT_ASPECT_RATIO = 0.8

# Got it with '\showthe\textwidth' in Latex
# (stops comilation and shows the number)
DOCUMENT_WIDTH_PT = 469.47049


def assert_eq(v1, v2, msg=None):
    if msg is None:
        assert v1 == v2, f"{v1} == {v2}"
    else:
        assert v1 == v2, msg


@dataclass(kw_only=True)
class PlotConfig:
    y_top: Optional[float]
    y_bottom: Optional[float]
    cond_y_top: Optional[float]
    cond_y_bottom: Optional[float]
    apply_cond: bool
    add_titles: bool
    scale: float
    format: str


# TODO: add varying prameters: capital_l, maybe others?
def main():
    parser = argparse.ArgumentParser(prog="plot")
    # directory is where benchmark results files are stored
    # the program itself will decide which files it needs
    parser.add_argument("-d", "--directories", nargs="+", required=True)
    parser.add_argument("-p",
                        "--plot-directory",
                        required=False,
                        default=DEFAULT_PLOT_DIR)
    parser.add_argument("-a",
                        "--algorithms",
                        choices=ALGOS,
                        nargs="+",
                        required=False,
                        default=ALGOS)
    parser.add_argument("-c",
                        "--core-counts",
                        type=int,
                        choices=CORE_COUNTS,
                        nargs="+",
                        required=False,
                        default=CORE_COUNTS)
    parser.add_argument("-t",
                        "--test-cases",
                        choices=TEST_CASES,
                        nargs="+",
                        required=False,
                        default=TEST_CASES)
    # Averages over given exchange generations.
    parser.add_argument("-e",
                        "--exchange-generations",
                        choices=EXCHANGE_GENERATIONS,
                        type=int,
                        nargs="+",
                        required=False,
                        default=[8, 32])
    # show results after this many generations
    parser.add_argument("-g",
                        "--generation-count",
                        type=int,
                        required=False,
                        default=MAX_GENERATIONS)
    parser.add_argument("--capital-ls",
                        type=int,
                        nargs="+",
                        required=True,
                        default=CAPITAL_LS)
    parser.add_argument("--gens-start",
                        type=int,
                        required=False,
                        default=GENS_START)
    parser.add_argument("--gens-step",
                        type=int,
                        required=False,
                        default=GENS_STEP)
    # Population size
    parser.add_argument("--population-sizes",
                        choices=POPULATION_SIZES,
                        type=int,
                        nargs="+",
                        required=False,
                        default=POPULATION_SIZES)
    parser.add_argument("--diff-type",
                        choices=["percent", "times"],
                        required=False,
                        default=DIFF)
    # what kind of plots to generate
    parser.add_argument("-k",
                        "--plot-kinds",
                        choices=PLOT_KINDS,
                        nargs="+",
                        required=False,
                        default=PLOT_KINDS)
    parser.add_argument("-f",
                        "--plot-format",
                        choices=["pgf", "png"],
                        required=False,
                        default=PLOT_FORMAT)
    # for pgf only, means width proportion of textwidth
    parser.add_argument("-s",
                        "--plot-scale",
                        type=float,
                        required=False,
                        default=PLOT_SCALE)
    parser.add_argument("--y-top", type=float, required=False, default=Y_TOP)
    parser.add_argument("--y-bottom",
                        type=float,
                        required=False,
                        default=Y_BOTTOM)
    parser.add_argument("--cond-y-top",
                        type=float,
                        required=False,
                        default=COND_Y_TOP)
    parser.add_argument("--cond-y-bottom",
                        type=float,
                        required=False,
                        default=COND_Y_BOTTOM)
    parser.add_argument("--dont-apply-cond",
                        type=bool,
                        required=False,
                        default=False)
    # Whether to add diagram title to the specified plot kind.
    # Currently only affects gens_diff_excg.
    parser.add_argument("--no-titles",
                        type=bool,
                        required=False,
                        default=False)
    args = parser.parse_args()
    # directory = canonicalize_dir(args.directory)
    plot_dir = canonicalize_dir(args.plot_directory)
    if not os.path.isdir(plot_dir):
        os.mkdir(plot_dir)

    plot_config = PlotConfig(y_top=args.y_top,
                             y_bottom=args.y_bottom,
                             add_titles=not args.no_titles,
                             scale=args.plot_scale,
                             format=args.plot_format,
                             cond_y_top=args.cond_y_top,
                             cond_y_bottom=args.cond_y_bottom,
                             apply_cond=not args.dont_apply_cond)

    # algos = list(map(lambda x: ALGO_TO_FILE_NAME_PART[x], args.algorithms))
    # algos = list(map(lambda x: ALGO_TO_UPPER[x], args.algorithms))
    algos = args.algorithms
    core_counts = args.core_counts
    test_cases = args.test_cases
    max_generations = args.generation_count
    exc_gens = args.exchange_generations
    population_sizes = args.population_sizes
    plot_kinds = args.plot_kinds

    # Read all results. Plotting functions will filter and take what's needed.
    all_results = []
    for dir in args.directories:
        print(f"Parsing files in directory '{dir}'")
        for file in os.listdir(dir):
            if file.endswith(".json"):
                filen = os.path.join(dir, file)
                # print(filen)
                data = read_bench_data(filen)
                all_results.append(data)
                # data.to_dict()
            else:
                print("Unrelated file found in benchmark results directory:\n",
                      file)

    if "gens_diff_excg" in plot_kinds:
        for a in algos:
            for t in test_cases:
                for c in core_counts:
                    if a == "PACO":
                        plot_paco_gens_diff_from_opt_exc_gens(
                            all_results=all_results,
                            plot_dir=plot_dir,
                            test_case=t,
                            core_count=c,
                            max_gens=max_generations,
                            exc_gens=exc_gens,
                            plot_config=plot_config)
                    else:
                        for capital_l in args.capital_ls:
                            for p in population_sizes:
                                plot_abc_gens_diff_from_opt_exc_gens(
                                    all_results=all_results,
                                    plot_dir=plot_dir,
                                    test_case=t,
                                    core_count=c,
                                    algo=a,
                                    pop_size=p,
                                    capital_l=capital_l,
                                    max_gens=max_generations,
                                    exc_gens=exc_gens,
                                    plot_config=plot_config)

    if "cores_diff_test_cases" in plot_kinds:
        for e in exc_gens:
            for a in algos:
                if a == "PACO":
                    plot_paco_cores_diff_from_opt_test_cases(
                        all_results=all_results,
                        core_counts=core_counts,
                        test_cases=test_cases,
                        exc_gens=e,
                        max_gens=max_generations,
                        plot_dir=plot_dir,
                        plot_config=plot_config)
                else:
                    for p in population_sizes:
                        for capital_l in args.capital_ls:
                            plot_abc_cores_diff_from_opt_test_cases(
                                all_results=all_results,
                                core_counts=core_counts,
                                test_cases=test_cases,
                                exc_gens=e,
                                algo=a,
                                capital_l=capital_l,
                                max_gens=max_generations,
                                pop_size=p,
                                plot_dir=plot_dir,
                                plot_config=plot_config)

    if "cores_diff_gens" in plot_kinds:
        for t in test_cases:
            for e in exc_gens:
                for a in algos:
                    if a == "PACO":
                        plot_paco_cores_diff_from_opt_generations(
                            all_results=all_results,
                            test_case=t,
                            core_counts=core_counts,
                            exc_gens=e,
                            max_gens=max_generations,
                            gens_start=args.gens_start,
                            gens_step=args.gens_step,
                            plot_dir=plot_dir,
                            plot_config=plot_config)
                    else:
                        for p in population_sizes:
                            for capital_l in args.capital_ls:
                                plot_abc_cores_diff_from_opt_generations(
                                    all_results=all_results,
                                    test_case=t,
                                    core_counts=core_counts,
                                    exc_gens=e,
                                    max_gens=max_generations,
                                    algo=a,
                                    capital_l=capital_l,
                                    pop_size=p,
                                    gens_start=args.gens_start,
                                    gens_step=args.gens_step,
                                    plot_dir=plot_dir,
                                    plot_config=plot_config)

    if "cores_diff_algos" in plot_kinds:
        for t in test_cases:
            for p in population_sizes:
                for capital_l in args.capital_ls:
                    for e in exc_gens:
                        plot_cores_diff_from_opt_algos(
                            all_results=all_results,
                            test_case=t,
                            algos=algos,
                            core_counts=core_counts,
                            exc_gens=e,
                            max_gens=max_generations,
                            pop_size=p,
                            capital_l=capital_l,
                            plot_dir=plot_dir,
                            plot_config=plot_config)

    if "gens_diff_popsizes" in plot_kinds:
        for a in algos:
            if a == "PACO":
                print(
                    "PACO algorithm does not have different population sizes")
                continue
            for t in test_cases:
                for c in core_counts:
                    for e in exc_gens:
                        for capital_l in args.capital_ls:
                            plot_abc_generations_diff_from_opt_pop_sizes(
                                all_results=all_results,
                                test_case=t,
                                core_count=c,
                                algo=a,
                                exc_gens=e,
                                capital_l=capital_l,
                                max_gens=max_generations,
                                pop_sizes=population_sizes,
                                plot_dir=plot_dir,
                                plot_config=plot_config)

    if "gens_diff_cls" in plot_kinds:
        for a in algos:
            if a == "PACO":
                print("PACO does not have L parameter.")
                continue
            else:
                for t in test_cases:
                    for c in core_counts:
                        for e in exc_gens:
                            for p in population_sizes:
                                plot_abc_generations_diff_from_opt_capital_ls(
                                    all_results=all_results,
                                    test_case=t,
                                    core_count=c,
                                    algo=a,
                                    exc_gens=e,
                                    capital_ls=args.capital_ls,
                                    max_gens=max_generations,
                                    pop_size=p,
                                    plot_dir=plot_dir,
                                    plot_config=plot_config)

    if "cores_diff_cls" in plot_kinds:
        for a in algos:
            if a == "PACO":
                print("PACO does not have L parameter.")
                continue
            else:
                for t in test_cases:
                    for e in exc_gens:
                        for p in population_sizes:
                            plot_abc_cores_diff_from_opt_capital_ls(
                                all_results=all_results,
                                test_case=t,
                                core_counts=core_counts,
                                algo=a,
                                exc_gens=e,
                                capital_ls=args.capital_ls,
                                max_gens=max_generations,
                                pop_size=p,
                                plot_dir=plot_dir,
                                plot_config=plot_config)

    if "cores_diff_popsizes" in plot_kinds:
        for a in algos:
            if a == "PACO":
                print(
                    "PACO algorithm does not have different population sizes")
                continue
            for t in test_cases:
                for e in exc_gens:
                    for capital_l in args.capital_ls:
                        plot_abc_cores_diff_from_opt_pop_sizes(
                            all_results=all_results,
                            test_case=t,
                            algo=a,
                            core_counts=core_counts,
                            exc_gens=e,
                            capital_l=capital_l,
                            max_gens=max_generations,
                            pop_sizes=population_sizes,
                            plot_dir=plot_dir,
                            plot_config=plot_config)


def canonicalize_dir(directory):
    if not directory.endswith("/"):
        return directory + "/"
    else:
        return directory


def percent_diff_from_optimal(x: int, bench_data: BenchmarkData):
    optimal = bench_data.bench_config.problem.optimal_length
    diff = x - optimal
    return (diff / optimal) * 100.0


# Averages tour lengths for each generation over multiple benchmark runs.
def found_tours_avg(run_results: List[RunResult], bench_config: BenchConfig,
                    max_gens: int):
    # Average the generation lengths
    avg_gen_lengths = []
    for i in range(0, max_gens):
        sum_total = 0
        for j in range(bench_config.repeat_times):
            sum_total += run_results[j].shortest_iteration_tours[i]
        avg = sum_total / bench_config.repeat_times
        avg_gen_lengths.append(avg)

    return avg_gen_lengths


# x_values = array
# y_values = array of arrays
# labels = array of labels, len(labels) == len(y_values)
def plot_and_save(*,
                  x_values: List[int],
                  y_values: List[float],
                  labels: List[str],
                  title: str,
                  xlabel: str,
                  ylabel: str,
                  file_name: str,
                  config: PlotConfig,
                  xticks=None,
                  yticks=None,
                  style={
                      "marker": ".",
                      "linestyle": "dashed",
                      "linewidth": 0.75
                  },
                  legend_location=PLOT_LEGEND_LOCATION):
    # Label (legend) count must match number of plot lines.
    assert_eq(len(labels), len(y_values))
    # X axis value count must match value count in each plot line.
    assert_eq(len(x_values), len(y_values[0]))
    if config.format == "pgf":
        # mpl.use() must be called before importing pyplot
        mpl.use("pgf")
        from matplotlib import pyplot as plt
        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
            "figure.figsize": set_size(fraction=config.scale)
        })
        # TODO: why is fig not used?
        # plt.figure(figsize=set_size(fraction=PLOT_SCALE))
    else:
        from matplotlib import pyplot as plt
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "legend.labelspacing": 0.1,
        "legend.handlelength": 1.2,
        "legend.frameon": False,
        "legend.shadow": False,
        "legend.framealpha": 0.2,
        "legend.facecolor": "grey",
        "legend.borderpad": 0.4,
        "legend.borderaxespad": 0.0,
        # use ',' decimal separator and other locale settings
        "axes.formatter.use_locale": True
    })
    for (y, l) in zip(y_values, labels):
        plt.plot(x_values, y, label=l, **style)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    # If there are only a few X values, do not conditionally set y_top and y_bottom.
    if len(x_values) <= 10:
        config.apply_cond = False
    bottom, top = plt.gca().get_ylim()
    y_min = 100000000
    y_min_max = 0
    for i in range(len(y_values)):
        mm = np.min(y_values[i])
        if mm > y_min_max:
            y_min_max = mm
        if mm < y_min:
            y_min = mm
    # print("y_min", y_min, "y_min_max", y_min_max)
    if config.y_top is not None:
        plt.gca().set_ylim(top=config.y_top)
    elif config.apply_cond and top >= y_min_max * config.cond_y_top:
        plt.gca().set_ylim(top=y_min_max * config.cond_y_top)
    if config.y_bottom is not None:
        plt.gca().set_ylim(bottom=config.y_bottom)
    elif config.apply_cond:  # and bottom <= 0.8 * y_min * config.cond_y_bottom:
        plt.gca().set_ylim(bottom=y_min * config.cond_y_bottom)
    # plt.yscale("log")
    # plt.xscale("log")
    plt.legend(loc=legend_location)
    if config.add_titles:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    final_file_name = f"{file_name}.{config.format}"
    if os.path.exists(final_file_name):
        raise FileExistsError(
            f"Will not overwrite existing plot file:\n{final_file_name}")
    print(f"saving plot: {final_file_name}")
    # plt.ticklabel_format(useOffset=False)
    plt.margins(0.02)
    plt.tight_layout(pad=0.0)
    # dpi is ignored when using pgf
    plt.savefig(final_file_name, format=config.format, dpi=PLOT_DPI)
    plt.clf()
    plt.close()


# Plots the difference from the optimal length.
# x axis - generations
# y axis - diff from optimal
# in one plot: one test case, one thread count, all F_mig (exchange gens)
def plot_paco_gens_diff_from_opt_exc_gens(*, all_results: List[BenchmarkData],
                                          test_case: str, core_count: int,
                                          exc_gens: List[int], max_gens: int,
                                          plot_dir: str,
                                          plot_config: PlotConfig):
    # This graph is pointless with one thread.
    if core_count == 1:
        print("gens_diff_excg graph is pointless with 1 core")
        return
    title = f"PACO, \\texttt{{{test_case}}}, $B = {core_count}$"
    plot_file_name = f"gens_diff_from_opt_exc_gens_{test_case}_PACO_m{max_gens}_c{core_count}"
    x_values = np.arange(1, max_gens + 1)
    xlabel = ITERATIONS_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    br_init = list(
        filter(
            lambda r: r.bench_config.algorithm == "PACO" and r.bench_config.
            problem.name == test_case and r.bench_config.process_count ==
            core_count and r.bench_config.problem.name.endswith(
                str(r.bench_config.algorithm_constants.population_size)),
            all_results))
    # print_file_names(br_init)
    labels_all_exc_gens = []
    y_values = []
    for exc in exc_gens:
        labels_all_exc_gens.append(f"$D_m={exc}$")
        br_exc = list(
            filter(
                lambda r: r.bench_config.algorithm_constants.
                exchange_generations == exc, br_init))
        assert_eq(len(br_exc), 1)
        result = br_exc[0]
        gen_tour_lengths = found_tours_avg(result.run_results,
                                           result.bench_config, max_gens)
        # Plot the percentage difference from the optimal tour.
        diff = map(lambda length: percent_diff_from_optimal(length, result),
                   gen_tour_lengths)
        y_values.append(list(diff))

    plot_and_save(x_values=x_values,
                  y_values=y_values,
                  labels=labels_all_exc_gens,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config,
                  style={"linewidth": 1})


# Plots the difference from the optimal length.
# x axis - generations
# y axis - diff from optimal
# in one plot: one test case, one thread count, all F_mig (exchange gens)
def plot_abc_gens_diff_from_opt_exc_gens(*, all_results: List[BenchmarkData],
                                         test_case: str, core_count: int,
                                         pop_size: int, algo: str,
                                         exc_gens: List[int], max_gens: int,
                                         capital_l: int, plot_dir: str,
                                         plot_config: PlotConfig):
    # This graph is pointless with one thread.
    if core_count == 1:
        print("gens_diff_excg graph is pointless with 1 core")
        return
    title = f"{algo}, \\texttt{{{test_case}}}, $B = {core_count}$, $P = {pop_size}$"
    plot_file_name = f"gens_diff_from_opt_exc_gens_{test_case}_{algo}_m{max_gens}_c{core_count}_p{pop_size}_cl{capital_l}"
    x_values = np.arange(1, max_gens + 1)
    xlabel = ITERATIONS_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    br_init = list(
        filter(
            lambda r: r.bench_config.algorithm == algo and r.bench_config.
            problem.name == test_case and r.bench_config.algorithm_constants.
            colony_size == pop_size and r.bench_config.process_count ==
            core_count and r.bench_config.algorithm_constants.capital_l ==
            capital_l, all_results))
    # print_file_names(br_init)
    labels_all_exc_gens = []
    y_values = []
    for exc in exc_gens:
        labels_all_exc_gens.append(f"$D_m={exc}$")
        br_exc = list(
            filter(
                lambda r: r.bench_config.algorithm_constants.
                exchange_generations == exc, br_init))
        # print_file_names(br_exc)
        assert_eq(len(br_exc), 1)
        result = br_exc[0]
        gen_tour_lengths = found_tours_avg(result.run_results,
                                           result.bench_config, max_gens)
        # Plot the percentage difference from the optimal tour.
        diff = map(lambda length: percent_diff_from_optimal(length, result),
                   gen_tour_lengths)
        y_values.append(list(diff))

    plot_and_save(x_values=x_values,
                  y_values=y_values,
                  labels=labels_all_exc_gens,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config,
                  style={"linewidth": 1})


def print_file_names(results: List[BenchmarkData]):
    print("File names:")
    for r in results:
        print(r.results_file_name)


# Core count on X axis, difference from optimal on Y,
# different test cases in one plot.
def plot_paco_cores_diff_from_opt_test_cases(
        *, all_results: List[BenchmarkData], core_counts: List[int],
        test_cases: List[str], exc_gens: int, max_gens: int, plot_dir: str,
        plot_config: PlotConfig):
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    bench_results_algo = list(
        filter(
            lambda r: r.bench_config.algorithm == "PACO" and r.bench_config.
            problem.name.endswith(
                str(r.bench_config.algorithm_constants.population_size)) and r.
            bench_config.algorithm_constants.exchange_generations == exc_gens,
            all_results))
    labels_all_test_cases = []
    diffs_all_test_cases = []
    for t in test_cases:
        br_t = list(
            filter(lambda r: r.bench_config.problem.name == t,
                   bench_results_algo))
        # print_file_names(br_t)
        labels_all_test_cases.append(f"\\texttt{{{t}}}")
        diffs_all_core_counts = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c, br_t))
            # There must be exactly one file left satisfying all the conditions.
            assert_eq(len(br_c), 1, msg=f"len(br_c) = {len(br_c)}")
            results = br_c[0]
            total = 0
            for rr in results.run_results:
                total += rr.shortest_iteration_tours[max_gens - 1]
            avg = total / results.bench_config.repeat_times
            diff = percent_diff_from_optimal(avg, results)
            diffs_all_core_counts.append(diff)
        diffs_all_test_cases.append(diffs_all_core_counts)

    # if len(exc_gens) == 1:
    title = f"PACO, $D_m = {exc_gens}$, $i_{{maks}} = {max_gens}$"
    plot_file_name = f"cores_diff_from_opt_test_cases_PACO_egen_{exc_gens}_m{max_gens}"
    # else:
    # title = f"{ALGO_DISPLAY_NAMES[algo]}, $K = {max_gens}$ kart\\~{{ų}}, $P = {pop_size}$"
    # plot_file_name = f"cores_diff_from_opt_test_cases_{algo}_p{pop_size}"

    plot_config.apply_cond = False
    plot_and_save(x_values=x_values,
                  y_values=diffs_all_test_cases,
                  labels=labels_all_test_cases,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config)


def plot_abc_cores_diff_from_opt_test_cases(
        *, all_results: List[BenchmarkData], core_counts: List[int], algo: str,
        test_cases: List[str], exc_gens: int, max_gens: int, pop_size: int,
        capital_l: int, plot_dir: str, plot_config: PlotConfig):
    title = f"{algo}, $D_m = {exc_gens}$, $P = {pop_size}$, $L = {capital_l}$"
    plot_file_name = f"cores_diff_from_opt_test_cases_{algo}_cs{pop_size}_egen_{exc_gens}_m{max_gens}_cl{capital_l}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    bench_results_algo = list(
        filter(
            lambda r: r.bench_config.algorithm == algo and r.bench_config.
            algorithm_constants.exchange_generations == exc_gens and r.
            bench_config.algorithm_constants.colony_size == pop_size and r.
            bench_config.algorithm_constants.capital_l == capital_l,
            all_results))
    # print_file_names(bench_results_algo)
    labels_all_test_cases = []
    diffs_all_test_cases = []
    for t in test_cases:
        br_t = list(
            filter(lambda r: r.bench_config.problem.name == t,
                   bench_results_algo))
        # print("t: ", t)
        # print_file_names(br_t)
        labels_all_test_cases.append(f"\\texttt{{{t}}}")
        diffs_all_core_counts = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c, br_t))
            # print("c: ", c)
            assert_eq(len(br_c), 1)
            results = br_c[0]
            total = 0
            for rr in results.run_results:
                total += rr.shortest_iteration_tours[max_gens - 1]
            avg = total / results.bench_config.repeat_times
            diff = percent_diff_from_optimal(avg, results)
            diffs_all_core_counts.append(diff)
        diffs_all_test_cases.append(diffs_all_core_counts)

    plot_config.apply_cond = False
    plot_and_save(x_values=x_values,
                  y_values=diffs_all_test_cases,
                  labels=labels_all_test_cases,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config)


# Core count on X axis, difference from optimal on Y,
# plots single test case, varies generations count.
# plots every gens_step gens, up to but not including max_gens
def plot_paco_cores_diff_from_opt_generations(
        *, all_results: List[BenchmarkData], test_case: str,
        core_counts: List[int], exc_gens: int, max_gens: int, gens_start: int,
        gens_step: int, plot_dir: str, plot_config: PlotConfig):
    title = f"PACO, \\texttt{{{test_case}}}, $D_m = {exc_gens}$"
    plot_file_name = f"cores_diff_from_opt_gens_{test_case}_PACO_egen_{exc_gens}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    if DIFF == "percent":
        ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    bench_results_algo = list(
        filter(
            lambda r: r.bench_config.algorithm == "PACO" and r.bench_config.
            algorithm_constants.exchange_generations == exc_gens and r.
            bench_config.problem.name.endswith(
                str(r.bench_config.algorithm_constants.population_size)
            ) and r.bench_config.problem.name == test_case, all_results))
    labels_all_gens_counts = []
    diffs_all_gens_counts = []
    # TODO: include generation 0
    for g in range(gens_start, max_gens, gens_step):
        # if GENS_NAME == "k":
        # labels_all_gens_counts.append(f"$K = {str(g + 1)}$")
        # elif GENS_NAME == "kartos":
        labels_all_gens_counts.append(f"${str(g + 1)}$ iteracijų")
        diffs_single_gens_count = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c,
                       bench_results_algo))
            # print_file_names(br_c)
            assert_eq(len(br_c), 1, msg=f"len(br_c) = {len(br_c)}")
            results = br_c[0]
            total = 0
            for rr in results.run_results:
                total += rr.shortest_iteration_tours[g]
            avg = total / results.bench_config.repeat_times
            if DIFF == "percent":
                diff = percent_diff_from_optimal(avg, results)
            elif DIFF == "times":
                diff = avg / results.bench_config.problem.optimal_length
            diffs_single_gens_count.append(diff)
        diffs_all_gens_counts.append(diffs_single_gens_count)

    plot_config.apply_cond = False
    plot_and_save(x_values=x_values,
                  y_values=diffs_all_gens_counts,
                  labels=labels_all_gens_counts,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config)


# Core count on X axis, difference from optimal on Y,
# plots single test case, varies generations count.
# plots every gens_step gens, up to and including max_gens
def plot_abc_cores_diff_from_opt_generations(
        *, all_results: List[BenchmarkData], test_case: str,
        core_counts: List[int], exc_gens: int, max_gens: int, algo: str,
        capital_l: int, pop_size: int, gens_start: int, gens_step: int,
        plot_dir: str, plot_config: PlotConfig):
    title = f"{algo}, \\texttt{{{test_case}}}, $D_m = {exc_gens}$, $P = {pop_size}$, $L = {capital_l}$"
    plot_file_name = f"cores_diff_from_opt_gens_{test_case}_{algo}_egen_{exc_gens}_cs{pop_size}_cl{capital_l}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    if DIFF == "percent":
        ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    bench_results_algo = list(
        filter(
            lambda r: r.bench_config.algorithm == algo and r.bench_config.
            algorithm_constants.capital_l == capital_l and r.bench_config.
            algorithm_constants.exchange_generations == exc_gens and r.
            bench_config.problem.name == test_case and r.bench_config.
            algorithm_constants.colony_size == pop_size, all_results))
    # print_file_names(bench_results_algo)
    labels_all_gens_counts = []
    diffs_all_gens_counts = []
    # TODO: include generation 0
    for g in range(gens_start, max_gens, gens_step):
        # if GENS_NAME == "k":
        # labels_all_gens_counts.append(f"$K = {str(g + 1)}$")
        # elif GENS_NAME == "kartos":
        labels_all_gens_counts.append(f"${str(g + 1)}$ iteracijų")
        diffs_single_gens_count = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c,
                       bench_results_algo))
            assert_eq(len(br_c), 1, msg=f"c = {c}, len(br_c) = {len(br_c)}")
            results = br_c[0]
            total = 0
            for rr in results.run_results:
                total += rr.shortest_iteration_tours[g]
            avg = total / results.bench_config.repeat_times
            if DIFF == "percent":
                diff = percent_diff_from_optimal(avg, results)
            elif DIFF == "times":
                diff = avg / results.bench_config.problem.optimal_length
            diffs_single_gens_count.append(diff)
        diffs_all_gens_counts.append(diffs_single_gens_count)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_gens_counts,
                  labels=labels_all_gens_counts,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config)


# Core count on X axis, difference from optimal on Y,
# plots multiple algorithms and a single test case.
def plot_cores_diff_from_opt_algos(*, all_results: List[BenchmarkData],
                                   test_case: str, algos: List[str],
                                   core_counts: List[int], exc_gens: int,
                                   capital_l: int, max_gens: int,
                                   pop_size: int, plot_dir: str,
                                   plot_config: PlotConfig):
    title = f"\\texttt{{{test_case}}}, $D_m = {exc_gens}$, $i_{{maks}} = {max_gens}$, $P = {pop_size}$, $L = {capital_l}$"
    plot_file_name = f"cores_diff_from_opt_algos_{test_case}_mgen_{max_gens}_egen_{exc_gens}_p{pop_size}_l{capital_l}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    br_init = list(
        filter(
            lambda r: r.bench_config.algorithm_constants.max_iterations ==
            max_gens and r.bench_config.problem.name == test_case and r.
            bench_config.algorithm_constants.exchange_generations == exc_gens
            and r.bench_config.algorithm_constants.colony_size in
            [None, pop_size] and r.bench_config.algorithm_constants.capital_l
            in [None, capital_l] and (r.bench_config.problem.name.endswith(
                str(r.bench_config.algorithm_constants.population_size)
            ) or r.bench_config.algorithm_constants.population_size is None),
            all_results))
    # print_file_names(br_init)
    labels_all_algos = []
    diffs_all_algos = []
    for a in algos:
        labels_all_algos.append(a)
        br_a = list(filter(lambda r: r.bench_config.algorithm == a, br_init))
        # print(a)
        # print_file_names(br_a)
        diffs_single_algo = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c, br_a))
            # print_file_names(br_c)
            assert_eq(len(br_c), 1, msg=f"c = {c}, len(br_c) = {len(br_c)}")
            result = br_c[0]
            total = 0
            for rr in result.run_results:
                total += rr.shortest_iteration_tours[max_gens - 1]
            avg = total / result.bench_config.repeat_times
            diff = percent_diff_from_optimal(avg, result)
            diffs_single_algo.append(diff)
        diffs_all_algos.append(diffs_single_algo)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_algos,
                  labels=labels_all_algos,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config)


# Generations on X axis, difference from optimal on Y,
# plots multiple population sizes and a single test case.
def plot_abc_generations_diff_from_opt_pop_sizes(
        *, all_results: List[BenchmarkData], test_case: str, algo: str,
        core_count: int, exc_gens: int, max_gens: int, pop_sizes: List[int],
        plot_dir: str, capital_l: int, plot_config: PlotConfig):
    title = f"{algo}, \\texttt{{{test_case}}}, $D_m = {exc_gens}$, $B = {core_count}$"
    plot_file_name = f"gens_diff_from_opt_pop_sizes_{test_case}_{algo}_c{core_count}_mgen{max_gens}_egen{exc_gens}_cl{capital_l}"
    br_init = list(
        filter(
            lambda r: r.bench_config.problem.name == test_case and r.
            bench_config.algorithm == algo and r.bench_config.process_count ==
            core_count and r.bench_config.algorithm_constants.
            exchange_generations == exc_gens and r.bench_config.
            algorithm_constants.capital_l == capital_l, all_results))
    # print_file_names(br_init)
    x_values = np.arange(1, max_gens + 1)
    xlabel = ITERATIONS_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    labels_all_pop_sizes = []
    diffs_all_pop_sizes = []
    for p in pop_sizes:
        diffs_single_pop_size = []
        labels_all_pop_sizes.append(f"$P = {p}$")
        br_p = list(
            filter(
                lambda r: r.bench_config.algorithm_constants.colony_size == p,
                br_init))
        # print_file_names(br_p)
        assert_eq(len(br_p), 1)
        result = br_p[0]
        for g in range(max_gens):
            total = 0
            for rr in result.run_results:
                total += rr.shortest_iteration_tours[g]
            avg = total / result.bench_config.repeat_times
            diff = percent_diff_from_optimal(avg, result)
            diffs_single_pop_size.append(diff)
        diffs_all_pop_sizes.append(diffs_single_pop_size)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_pop_sizes,
                  labels=labels_all_pop_sizes,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config,
                  style={"linewidth": 1})


# Generations on X axis, difference from optimal on Y,
# plots multiple Ls and a single test case.
def plot_abc_generations_diff_from_opt_capital_ls(
        *, all_results: List[BenchmarkData], test_case: str, algo: str,
        core_count: int, exc_gens: int, max_gens: int, pop_size: int,
        plot_dir: str, capital_ls: List[int], plot_config: PlotConfig):
    title = f"{algo}, \\texttt{{{test_case}}}, $D_m = {exc_gens}$, $B = {core_count}$"
    plot_file_name = f"gens_diff_from_opt_capital_ls_{test_case}_{algo}_c{core_count}_mgen{max_gens}_egen{exc_gens}_p{pop_size}"
    br_init = list(
        filter(
            lambda r: r.bench_config.problem.name == test_case and r.
            bench_config.algorithm == algo and r.bench_config.process_count ==
            core_count and r.bench_config.algorithm_constants.
            exchange_generations == exc_gens and r.bench_config.
            algorithm_constants.colony_size == pop_size, all_results))
    # print_file_names(br_init)
    x_values = np.arange(1, max_gens + 1)
    xlabel = ITERATIONS_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    labels_all_capital_ls = []
    diffs_all_pop_sizes = []
    for cl in capital_ls:
        diffs_single_pop_size = []
        labels_all_capital_ls.append(f"$L = {cl}$")
        br_p = list(
            filter(
                lambda r: r.bench_config.algorithm_constants.capital_l == cl,
                br_init))
        # print_file_names(br_p)
        assert_eq(len(br_p), 1)
        result = br_p[0]
        for g in range(max_gens):
            total = 0
            for rr in result.run_results:
                total += rr.shortest_iteration_tours[g]
            avg = total / result.bench_config.repeat_times
            diff = percent_diff_from_optimal(avg, result)
            diffs_single_pop_size.append(diff)
        diffs_all_pop_sizes.append(diffs_single_pop_size)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_pop_sizes,
                  labels=labels_all_capital_ls,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config,
                  style={"linewidth": 1})


# Core count on X axis, difference from optimal on Y,
# plots multiple Ls and a single test case.
def plot_abc_cores_diff_from_opt_capital_ls(
        *, all_results: List[BenchmarkData], test_case: str, algo: str,
        core_counts: List[int], exc_gens: int, max_gens: int, pop_size: int,
        plot_dir: str, capital_ls: List[int], plot_config: PlotConfig):
    title = f"{algo}, \\texttt{{{test_case}}}, $D_m = {exc_gens}$, $P = {pop_size}$"
    plot_file_name = f"cores_diff_from_opt_capital_ls_{test_case}_{algo}_mgen{max_gens}_egen{exc_gens}_p{pop_size}"
    br_init = list(
        filter(
            lambda r: r.bench_config.problem.name == test_case and r.
            bench_config.algorithm == algo and r.bench_config.
            algorithm_constants.exchange_generations == exc_gens and r.
            bench_config.algorithm_constants.colony_size == pop_size,
            all_results))
    # print_file_names(br_init)
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    labels_all_pop_sizes = []
    diffs_all_cls = []
    for cl in capital_ls:
        labels_all_pop_sizes.append(f"$L = {cl}$")
        br_cl = list(
            filter(
                lambda r: r.bench_config.algorithm_constants.capital_l == cl,
                br_init))
        # print_file_names(br_p)
        diffs_single_cl = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c, br_cl))
            assert_eq(len(br_c), 1)
            result = br_c[0]
            total = 0
            for rr in result.run_results:
                total += rr.shortest_iteration_tours[max_gens - 1]
            avg = total / result.bench_config.repeat_times
            diff = percent_diff_from_optimal(avg, result)
            diffs_single_cl.append(diff)
        diffs_all_cls.append(diffs_single_cl)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_cls,
                  labels=labels_all_pop_sizes,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config)


# Core count on X axis, difference from optimal on Y,
# plots multiple population sizes and a single test case.
def plot_abc_cores_diff_from_opt_pop_sizes(
        *, all_results: List[BenchmarkData], test_case: str, algo: str,
        core_counts: List[int], exc_gens: int, max_gens: int, capital_l: int,
        pop_sizes: List[int], plot_dir: str, plot_config: PlotConfig):

    popstring = '_'.join(map(str, pop_sizes))
    # if GENS_NAME == "kartos":
    # title = f"{algo}, \\texttt{{{test_case}}}, ${max_gens}$ iteracijų"
    # elif GENS_NAME == "k":
    title = f"{algo}, \\texttt{{{test_case}}}, $D_m = {exc_gens}$, $i_{{maks}} = {max_gens}$"
    plot_file_name = f"cores_diff_from_opt_pop_sizes_{test_case}_{algo}_mgen_{max_gens}_egen_{exc_gens}_p{popstring}"
    br_init = list(
        filter(
            lambda r: r.bench_config.algorithm == algo and r.bench_config.
            problem.name == test_case and r.bench_config.algorithm_constants.
            exchange_generations == exc_gens and r.bench_config.
            algorithm_constants.capital_l == capital_l, all_results))
    # print_file_names(br_init)
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    labels_all_pop_sizes = []
    diffs_all_pop_sizes = []
    for p in pop_sizes:
        diffs_single_pop_size = []
        labels_all_pop_sizes.append(f"$P = {p}$")
        br_p = list(
            filter(
                lambda r: r.bench_config.algorithm_constants.colony_size == p,
                br_init))
        # print_file_names(br_p)
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c, br_p))
            # print_file_names(br_c)
            assert_eq(len(br_c), 1)
            result = br_c[0]
            total = 0
            for rr in result.run_results:
                total += rr.shortest_iteration_tours[max_gens - 1]
            avg = total / result.bench_config.repeat_times
            diff = percent_diff_from_optimal(avg, result)
            diffs_single_pop_size.append(diff)
        diffs_all_pop_sizes.append(diffs_single_pop_size)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_pop_sizes,
                  labels=labels_all_pop_sizes,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  config=plot_config)


def average(a):
    return sum(a) / len(a)


# for pgf
def set_size(fraction=1, subplots=(1, 1)):
    """Set figure dimensions to sit nicely in our document.

    Parameters
    ----------
    width_pt: float
            Document width in points
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    # Width of figure (in pts)
    fig_width_pt = DOCUMENT_WIDTH_PT * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * PLOT_ASPECT_RATIO * (subplots[0] /
                                                        subplots[1])

    return (fig_width_in, fig_height_in)


if __name__ == "__main__":
    main()
