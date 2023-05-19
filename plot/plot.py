from typing import List
import os
import argparse
import matplotlib as mpl
import numpy as np
from parse_results import read_bench_data, BenchmarkData, RunResult, BenchConfig, AlgorithmConstants

# For number decimal separator formatting.
import locale

locale.setlocale(locale.LC_NUMERIC, 'lt_LT.utf8')

DEFAULT_PLOT_DIR = "graphs"
ALGOS = ["PACO", "CABC", "qCABC"]
CORE_COUNTS = [1, 2, 4, 6, 8]
TEST_CASES = ["eil51", "eil76", "kroA100", "kroA200", "d198", "lin318"]
EXCHANGE_GENERATIONS = [8, 32]
# <x axis>_<y axis>_<what we are comparing>
PLOT_KINDS = [
    "gens_diff_excg", "cores_diff_test_cases", "cores_diff_gens",
    "cores_diff_algos", "gens_diff_popsizes", "cores_diff_popsizes",
    "relative_times"
]
MAX_GENERATIONS = 4000
# Population is the same size as city count on PACO.
POPULATION_SIZES = [20, 40, 80]
DEFAULT_L = 2

# ALGO_TO_FILE_NAME_PART = {"paco": "paco", "cabc": "CABC", "qcabc": "qCABC"}
# ALGO_TO_UPPER = {"paco": "PACO", "cabc": "CABC", "qcabc": "qCABC"}

GENS_START = 99
GENS_STEP = 100

# GENS_NAME = "k"
GENS_NAME = "kartos"

# controls where in the plot the legend is placed
PLOT_LEGEND_LOCATION = "upper right"

DIFF = "percent"

CORE_COUNT_AXIS_LABEL = "branduolių skaičius"
DIFF_FROM_OPTIMAL_AXIS_LABEL = "skirtumas nuo optimalaus, $\%$"
GENERATIONS_AXIS_LABEL = "genetinio algoritmo karta"

# controls plot image resolution (png)
PLOT_DPI = 220

PLOT_FORMAT = "pgf"
# PLOT_FORMAT = "png"

# pgf plot scale
PLOT_SCALE = 1.0

# For controlling Y axis range (ylim)
Y_TOP = None
Y_BOTTOM = None

# height / width
PLOT_ASPECT_RATIO = 0.8

# Got it with '\showthe\textwidth' in Latex
# (stops comilation and shows the number)
DOCUMENT_WIDTH_PT = 469.47049


# TODO: add varying prameters: capital_l, maybe others?
def main():
    parser = argparse.ArgumentParser(prog="plot")
    # directory is where benchmark results files are stored
    # the program itself will decide which files it needs
    parser.add_argument("-d", "--directory", required=True)
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
                        default=EXCHANGE_GENERATIONS)
    # show results after this many generations
    parser.add_argument("-g",
                        "--generation-count",
                        type=int,
                        required=False,
                        default=MAX_GENERATIONS)
    parser.add_argument("-l",
                        "--capital-l",
                        type=float,
                        required=True,
                        default=DEFAULT_L)
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
    global PLOT_FORMAT
    parser.add_argument("-f",
                        "--plot-format",
                        choices=["pgf", "png"],
                        required=False,
                        default=PLOT_FORMAT)
    global PLOT_SCALE
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

    PLOT_FORMAT = args.plot_format
    PLOT_SCALE = args.plot_scale

    # algos = list(map(lambda x: ALGO_TO_FILE_NAME_PART[x], args.algorithms))
    # algos = list(map(lambda x: ALGO_TO_UPPER[x], args.algorithms))
    algos = args.algorithms
    core_counts = args.core_counts
    test_cases = args.test_cases
    max_generations = args.generation_count
    exc_gens = args.exchange_generations
    population_sizes = args.population_sizes
    plot_kinds = args.plot_kinds
    add_title = not args.no_titles

    # Read all results. Plotting functions will filter and take what's needed.
    all_results = []
    for file in os.listdir(args.directory):
        if file.endswith(".json"):
            filen = os.path.join(args.directory, file)
            print(filen)
            data = read_bench_data(filen)
            all_results.append(data)
            # data.to_dict()
        else:
            print("Unrelated file found in benchmark results directory:\n",
                  file)

    if "cores_diff_test_cases" in plot_kinds:
        for e in exc_gens:
            for p in population_sizes:
                for a in algos:
                    if a == "PACO":
                        plot_paco_cores_diff_from_opt_test_cases(
                            all_results=all_results,
                            core_counts=core_counts,
                            test_cases=test_cases,
                            exc_gens=e,
                            max_gens=max_generations,
                            plot_dir=plot_dir,
                            add_title=add_title)
                    else:
                        plot_abc_cores_diff_from_opt_test_cases(
                            all_results=all_results,
                            core_counts=core_counts,
                            test_cases=test_cases,
                            exc_gens=e,
                            algo=a,
                            max_gens=max_generations,
                            pop_size=p,
                            plot_dir=plot_dir,
                            add_title=add_title)

    # if "cores_diff_gens" in plot_kinds:
    #     for a in algos:
    #         for t in test_cases:
    #             for p in population_sizes:
    #                 plot_cores_diff_from_opt_generations(
    #                     directory=directory,
    #                     test_case=t,
    #                     algo=a,
    #                     core_counts=core_counts,
    #                     exc_gens=exc_gens,
    #                     max_gens=max_generations,
    #                     pop_size=p,
    #                     results_dir=results_dir,
    #                     add_title=add_title)

    # if "gens_diff_popsizes" in plot_kinds:
    #     for a in algos:
    #         for t in test_cases:
    #             for c in core_counts:
    #                 plot_generations_diff_from_opt_pop_sizes(
    #                     directory=directory,
    #                     test_case=t,
    #                     algo=a,
    #                     core_count=c,
    #                     exc_gens=exc_gens,
    #                     max_gens=max_generations,
    #                     pop_sizes=population_sizes,
    #                     results_dir=results_dir,
    #                     add_title=add_title)

    # if "cores_diff_popsizes" in plot_kinds:
    #     for a in algos:
    #         for t in test_cases:
    #             plot_cores_diff_from_opt_pop_sizes(directory=directory,
    #                                                test_case=t,
    #                                                algo=a,
    #                                                core_counts=core_counts,
    #                                                exc_gens=exc_gens,
    #                                                max_gens=max_generations,
    #                                                pop_sizes=population_sizes,
    #                                                results_dir=results_dir,
    #                                                add_title=add_title)

    # if "cores_diff_algos" in plot_kinds:
    #     for t in test_cases:
    #         for p in population_sizes:
    #             plot_cores_diff_from_opt_algos(directory=directory,
    #                                            test_case=t,
    #                                            algos=algos,
    #                                            core_counts=core_counts,
    #                                            exc_gens=exc_gens,
    #                                            max_gens=max_generations,
    #                                            pop_size=p,
    #                                            results_dir=results_dir,
    #                                            add_title=add_title)

    # if "gens_diff_excg" in plot_kinds:
    #     for p in population_sizes:
    #         plot_basic(directory=directory,
    #                    results_dir=results_dir,
    #                    algos=algos,
    #                    test_cases=test_cases,
    #                    core_counts=core_counts,
    #                    pop_size=p,
    #                    add_title=add_title)


def canonicalize_dir(directory):
    if not directory.endswith("/"):
        return directory + "/"
    else:
        return directory


def percent_diff_from_optimal(x, optimal):
    diff = x - optimal
    return (diff / optimal) * 100.0


# Averages tour lengths for each generation over multiple benchmark runs.
def one_exchange_gen_avg(run_results: List[RunResult],
                         bench_config: BenchConfig, max_gens: int):
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
                  x_values,
                  y_values,
                  labels,
                  title,
                  xlabel,
                  ylabel,
                  file_name,
                  add_title,
                  xticks=None,
                  yticks=None,
                  style={
                      "marker": ".",
                      "linestyle": "dashed",
                      "linewidth": 0.75
                  },
                  legend_location=PLOT_LEGEND_LOCATION):
    assert len(labels) == len(y_values)
    if PLOT_FORMAT == "pgf":
        # mpl.use() must be called before importing pyplot
        mpl.use("pgf")
        from matplotlib import pyplot as plt
        plt.rcParams.update({
            "font.family": "serif",  # use serif/main font for text elements
            "text.usetex": True,  # use inline math for ticks
            "pgf.rcfonts": False,  # don't setup fonts from rc parameters
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
            "axes.formatter.use_locale": True  # use decimal separator ','
        })
        # TODO: why is fig not used?
        fig = plt.figure(figsize=set_size(fraction=PLOT_SCALE))
    else:
        from matplotlib import pyplot as plt

    for (y, l) in zip(y_values, labels):
        plt.plot(x_values, y, label=l, **style)
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)
    plt.legend(loc=legend_location)
    if add_title:
        plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(top=Y_TOP, bottom=Y_BOTTOM)
    final_file_name = f"{file_name}.{PLOT_FORMAT}"
    if os.path.exists(final_file_name):
        raise FileExistsError(
            f"Will not overwrite existing plot file:\n{final_file_name}")
    print(f"saving plot: {final_file_name}")
    plt.tight_layout(pad=0.0)
    # dpi is ignored when using pgf
    plt.savefig(final_file_name, format=PLOT_FORMAT, dpi=PLOT_DPI)
    plt.clf()
    plt.close()


# dir must end with '/'
def make_file_name(directory, test_case, algo, cpus, pop_size):
    return f"{directory}bm_{test_case}_{algo}_{cpus}_cpus_p{pop_size}.out"


# Template:
# bm_paco_{problem_name}_{cpu_count}cpus_p{popsize}_q{capital_q_mul}_a{alpha}
# _b{beta}_ro{ro}_intensity{initial_intensity}_k{k}_e{exchange_gens}_lowq{lowercase_q}.json
def make_bm_file_path_paco(*, dir, problem_name, cpu_count, popsize,
                           capital_q_mul, alpha, beta, ro, initial_intensity,
                           k, exchange_gens, lowercase_q):
    return f"{dir}bm_paco_{cpu_count}cpus_p{popsize}_q{capital_q_mul}_a{alpha}_b{beta}_ro{ro}_intensity{initial_intensity}_k{k}_e{exchange_gens}_lowq{lowercase_q}.json"


def make_bm_file_path_abc(*, dir, problem_name, cpu_count, colony_size, nl_max,
                          p_cp, p_rc, p_l, l_min, l_max_mul, r, lowercase_q,
                          exchange_gens, k, capital_l):
    return f"{dir}_{problem_name}_{cpu_count}cpus_cs{colony_size}_nlmax{nl_max}_pcp{p_cp}_prc{p_rc}_pl{p_l}_lmin{l_min}_lmaxm{l_max_mul}_r{r}_q{lowercase_q}_e{exchange_gens}_k{k}_cl{capital_l}"


# Plots the difference from the optimal length.
# x axis - generations
# y axis - diff from optimal
# in one plot: one test case, one thread count, all F_mig (exchange gens)
def plot_basic(*, directory, results_dir, algos, test_cases, core_counts,
               pop_size, add_title):

    x_axis_values = np.arange(1, 501)

    for a in algos:
        for t in test_cases:
            for c in core_counts:
                file_name = make_file_name(directory, t, a, c, pop_size)
                (meta, data) = parse_benchmark_results(file_name)

                problem_name = meta.problem_name
                optimal_length = meta.optimal_length
                algorithm = meta.algorithm
                cpu_count = meta.cpu_count
                plot_file_base_name = file_name.split('/')[-1].split('.')[0]
                y_values = []
                labels = []
                title = f"{ALGO_DISPLAY_NAMES[algorithm]}, \\texttt{{{problem_name}}}, $B = {cpu_count}$, $P = {pop_size}$"
                xlabel = GENERATIONS_AXIS_LABEL
                ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
                file_name = results_dir + plot_file_base_name
                for exc in data:
                    (meta_info, exc_gen_avg) = one_exchange_gen_avg(exc)
                    # Plot the percentage difference from the optimal tour.
                    diff = map(
                        lambda x:
                        (x - optimal_length) / optimal_length * 100.0,
                        exc_gen_avg)
                    y_values.append(list(diff))
                    labels.append(f"$D_m={exc.exc_gens}$")

                plot_and_save(x_values=x_axis_values,
                              y_values=y_values,
                              labels=labels,
                              title=title,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              file_name=file_name,
                              add_title=add_title,
                              style={"linewidth": 1})

def print_file_names(results: List[BenchmarkData]):
    for r in results:
        print(r.results_file_name)

# Core count on X axis, difference from optimal on Y,
# different test cases in one plot.
def plot_paco_cores_diff_from_opt_test_cases(
        *, all_results: List[BenchmarkData], core_counts: List[int],
        test_cases: List[str], exc_gens: int, max_gens: int, #pop_size: int,
        plot_dir: str, add_title: bool):
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    bench_results_algo = list(filter(
        lambda r: r.bench_config.algorithm == "PACO" and r.bench_config.
        algorithm_constants.exchange_generations == exc_gens, all_results))
    labels_all_test_cases = []
    diffs_all_test_cases = []
    for t in test_cases:
        br_t = list(filter(lambda r: r.bench_config.problem.name == t,
                      bench_results_algo))
        print_file_names(br_t)
        labels_all_test_cases.append(f"\\texttt{{{t}}}")
        diffs_all_core_counts = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c, br_t))
            # There must be exactly one file left satisfying all the conditions.
            assert len(br_c) == 1, f"len(br_c) = {len(br_c)}"
            results = br_c[0]
            total = 0
            for rr in results.run_results:
                total += rr.shortest_iteration_tours[max_gens - 1]
            avg = total / results.bench_config.repeat_times
            diff = percent_diff_from_optimal(
                avg, results.bench_config.problem.optimal_length)
            diffs_all_core_counts.append(diff)
        diffs_all_test_cases.append(diffs_all_core_counts)

    # if len(exc_gens) == 1:
    title = f"PACO, $D_m = {exc_gens}$, $G = {max_gens}$"
    plot_file_name = f"cores_diff_from_opt_test_cases_paco_egen_{exc_gens}_m{max_gens}"
    # else:
    # title = f"{ALGO_DISPLAY_NAMES[algo]}, $K = {max_gens}$ kart\\~{{ų}}, $P = {pop_size}$"
    # plot_file_name = f"cores_diff_from_opt_test_cases_{algo}_p{pop_size}"

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_test_cases,
                  labels=labels_all_test_cases,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  add_title=add_title)


def plot_abc_cores_diff_from_opt_test_cases(
        *, all_results: List[BenchmarkData], core_counts: List[int], algo: str,
        test_cases: List[str], exc_gens: int, max_gens: int, pop_size: int,
        plot_dir: str, add_title: bool):
    print(core_counts, algo, test_cases, exc_gens, max_gens, pop_size)
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    bench_results_algo = list(filter(
        lambda r: r.bench_config.algorithm == algo and r.bench_config.
        algorithm_constants.exchange_generations == exc_gens and r.bench_config.algorithm_constants.colony_size == pop_size, all_results))
    print_file_names(bench_results_algo)
    labels_all_test_cases = []
    diffs_all_test_cases = []
    for t in test_cases:
        br_t = list(filter(lambda r: r.bench_config.problem.name == t,
                      bench_results_algo))
        print_file_names(br_t)
        labels_all_test_cases.append(f"\\texttt{{{t}}}")
        diffs_all_core_counts = []
        for c in core_counts:
            br_c = list(
                filter(lambda r: r.bench_config.process_count == c, br_t))
            # There must be exactly one file left satisfying all the conditions.
            assert len(br_c) == 1, f"len(br_c) = {len(br_c)}"
            results = br_c[0]
            total = 0
            for rr in results.run_results:
                total += rr.shortest_iteration_tours[max_gens - 1]
            avg = total / results.bench_config.repeat_times
            diff = percent_diff_from_optimal(
                avg, results.bench_config.problem.optimal_length)
            diffs_all_core_counts.append(diff)
        diffs_all_test_cases.append(diffs_all_core_counts)

    # if len(exc_gens) == 1:
    title = f"{algo}, $D_m = {exc_gens}$, $P = {pop_size}$, $G = {max_gens}$"
    plot_file_name = f"cores_diff_from_opt_test_cases_{algo}_cs{pop_size}_egen_{exc_gens}_m{max_gens}"
    # else:
    # title = f"{ALGO_DISPLAY_NAMES[algo]}, $K = {max_gens}$ kart\\~{{ų}}, $P = {pop_size}$"
    # plot_file_name = f"cores_diff_from_opt_test_cases_{algo}_p{pop_size}"

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_test_cases,
                  labels=labels_all_test_cases,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=plot_dir + plot_file_name,
                  add_title=add_title)


# Core count on X axis, difference from optimal on Y,
# plots single test case, varies generations count.
# plots every 100 gens, up to and including max_gens
def plot_cores_diff_from_opt_generations(*, all_results, test_case, algo,
                                         core_counts, exc_gens, max_gens,
                                         pop_size, results_dir, add_title):

    if len(exc_gens) == 1:
        if GENS_NAME == "kartos":
            title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $P = {pop_size}$"
        elif GENS_NAME == "k":
            title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_gens_{test_case}_{algo}_egen_{exc_gens[0]}_p{pop_size}"
    else:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_gens_{test_case}_{algo}_p{pop_size}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    if DIFF == "percent":
        ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    elif DIFF == "times":
        ylabel = "skirtumas nuo optimalaus, kartai"
    parsed_files = []
    labels_all_gens_counts = []
    diffs_all_gens_counts = []
    # TODO: include generation 0
    for g in range(GENS_START, max_gens, GENS_STEP):
        if GENS_NAME == "k":
            labels_all_gens_counts.append(f"$K = {str(g + 1)}$")
        elif GENS_NAME == "kartos":
            labels_all_gens_counts.append(f"${str(g + 1)}$ kart\\~{{ų}}")
        diffs_single_gens_count = []
        for c in core_counts:
            file_name = make_file_name(directory, test_case, algo, c, pop_size)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens,
                                       data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[g]
            avg = total / (r_group.record_count * required_exc_gens_count)
            if DIFF == "percent":
                diff = percent_diff_from_optimal(avg, meta.optimal_length)
            elif DIFF == "times":
                diff = avg / meta.optimal_length
            diffs_single_gens_count.append(diff)
        diffs_all_gens_counts.append(diffs_single_gens_count)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_gens_counts,
                  labels=labels_all_gens_counts,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=results_dir + plot_file_name,
                  add_title=add_title)


# Core count on X axis, difference from optimal on Y,
# plots multiple algorithms and a single test case.
def plot_cores_diff_from_opt_algos(*, all_results, test_case, algos,
                                   core_counts, exc_gens, max_gens, pop_size,
                                   results_dir, add_title):

    if len(exc_gens) == 1:
        title = f"\\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $K = {max_gens}$ kart\\~{{ų}} $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_algos_{test_case}_mgen_{max_gens}_egen_{exc_gens[0]}_p{pop_size}"
    else:
        title = f"\\texttt{{{test_case}}}, $K = {max_gens}$ kart\\~{{ų}} $P = {pop_size}$"
        plot_file_name = f"cores_diff_from_opt_algos_{test_case}_mgen_{max_gens}_p{pop_size}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_algos = []
    diffs_all_algos = []
    for a in algos:
        labels_all_algos.append(ALGO_DISPLAY_NAMES[a])
        diffs_single_algo = []
        for c in core_counts:
            file_name = make_file_name(directory, test_case, a, c, pop_size)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens,
                                       data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[max_gens - 1]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_single_algo.append(diff)
        diffs_all_algos.append(diffs_single_algo)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_algos,
                  labels=labels_all_algos,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=results_dir + plot_file_name,
                  add_title=add_title)


# Generations on X axis, difference from optimal on Y,
# plots multiple population sizes and a single test case.
def plot_generations_diff_from_opt_pop_sizes(*, all_results, test_case, algo,
                                             core_count, exc_gens, max_gens,
                                             pop_sizes, results_dir,
                                             add_title):

    if len(exc_gens) == 1:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $B = {core_count}$"
        plot_file_name = f"gens_diff_from_opt_pop_sizes_{test_case}_{algo}_cpus_{core_count}_mgen_{max_gens}_egen_{exc_gens[0]}"
    else:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $B = {core_count}$"
        plot_file_name = f"gens_diff_from_opt_pop_sizes_{test_case}_{algo}_cpus_{core_count}_mgen_{max_gens}"
    x_values = np.arange(1, max_gens + 1)
    xlabel = GENERATIONS_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_pop_sizes = []
    diffs_all_pop_sizes = []
    for p in pop_sizes:
        diffs_single_pop_size = []
        labels_all_pop_sizes.append(f"$P = {p}$")
        for g in range(0, max_gens):
            file_name = make_file_name(directory, test_case, algo, core_count,
                                       p)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens,
                                       data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[g]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_single_pop_size.append(diff)
        diffs_all_pop_sizes.append(diffs_single_pop_size)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_pop_sizes,
                  labels=labels_all_pop_sizes,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  file_name=results_dir + plot_file_name,
                  add_title=add_title,
                  style={"linewidth": 1})


# Core count on X axis, difference from optimal on Y,
# plots multiple population sizes and a single test case.
def plot_cores_diff_from_opt_pop_sizes(*, all_results, test_case, algo,
                                       core_counts, exc_gens, max_gens,
                                       pop_sizes, results_dir, add_title):

    popstring = '_'.join(map(str, pop_sizes))
    if len(exc_gens) == 1:
        if GENS_NAME == "kartos":
            title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, ${max_gens}$ kart\\~{{ų}}"
        elif GENS_NAME == "k":
            title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $D_m = {exc_gens[0]}$, $K = {max_gens}$"
        plot_file_name = f"cores_diff_from_opt_pop_sizes_{test_case}_{algo}_mgen_{max_gens}_egen_{exc_gens[0]}_p{popstring}"
    else:
        title = f"{ALGO_DISPLAY_NAMES[algo]}, \\texttt{{{test_case}}}, $K = {max_gens}$"
        plot_file_name = f"cores_diff_from_opt_pop_sizes_{test_case}_{algo}_mgen_{max_gens}_p{popstring}"
    x_values = core_counts
    xlabel = CORE_COUNT_AXIS_LABEL
    ylabel = DIFF_FROM_OPTIMAL_AXIS_LABEL
    parsed_files = []
    labels_all_pop_sizes = []
    diffs_all_pop_sizes = []
    for p in pop_sizes:
        diffs_single_pop_size = []
        labels_all_pop_sizes.append(f"$P = {p}$")
        for c in core_counts:
            file_name = make_file_name(directory, test_case, algo, c, p)
            (meta, data) = parse_benchmark_results(file_name)
            parsed_files.append((meta, data))
            total = 0
            # We are only interested in exc_gens specified.
            required_exc_gens = filter(lambda rg: rg.exc_gens in exc_gens,
                                       data)
            required_exc_gens_count = 0
            for r_group in required_exc_gens:
                required_exc_gens_count += 1
                for rec in r_group.records:
                    total += rec.lengths[max_gens - 1]
            avg = total / (r_group.record_count * required_exc_gens_count)
            diff = percent_diff_from_optimal(avg, meta.optimal_length)
            diffs_single_pop_size.append(diff)
        diffs_all_pop_sizes.append(diffs_single_pop_size)

    plot_and_save(x_values=x_values,
                  y_values=diffs_all_pop_sizes,
                  labels=labels_all_pop_sizes,
                  title=title,
                  xlabel=xlabel,
                  ylabel=ylabel,
                  xticks=core_counts,
                  file_name=results_dir + plot_file_name,
                  add_title=add_title)


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
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * PLOT_ASPECT_RATIO * (subplots[0] /
                                                        subplots[1])

    return (fig_width_in, fig_height_in)


if __name__ == "__main__":
    main()
