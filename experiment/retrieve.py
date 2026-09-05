import csv
import glob
import os
from collections import defaultdict, OrderedDict
from datetime import datetime
import h5py
import numpy as np
import scikit_posthocs as sp
import pandas as pd
from scipy.stats import binomtest, kruskal, f_oneway, rankdata, shapiro, mannwhitneyu
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn

from experiment.aggregation import combine_data
from experiment.globals import BENCHMARK_BASE_SEED
from experiment.problem_identity import create_seeded_problem
from experiment.plotting_utilities import plot_results, plot_results_with_std, plot_box_at_intervals, plot_final_box, \
    plot_final_raincloud, plot_final_petit_prince, plot_results_with_annotations, plot_results_with_average, \
    plot_results_with_annotations_legend
from experiment.setup import setup_experiment, make_dir

# Setup experiment to retrieve settings like algorithm_colors, max_evaluations, etc.
(algorithms, group_of_algorithms, problems, _, number_of_variables, solutions_size,
 max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()



def collect_h5_files_from_paths(paths):
    h5_files = []
    for path in paths:
        files = glob.glob(os.path.join(path, '**', '*.h5'), recursive=True)
        h5_files.extend(files)
    return h5_files


def final_fitness_means(results, algos):
    """Per-algorithm mean of the finite final fitness values.

    Algorithms that are missing, empty, or have no finite finals are omitted.
    """
    means = {}
    for algo in algos:
        a_data = results.get(algo)
        if not a_data:
            continue
        arr = np.asarray(a_data.get('data', np.array([])))
        if arr.size == 0:
            continue
        if arr.ndim == 2:
            arr = arr[:, -1]
        elif arr.ndim != 1:
            continue
        finite = arr[np.isfinite(arr)]
        if finite.size > 0:
            means[algo] = float(np.mean(finite))
    return means


def rank_within_problem(means):
    """Average ranks of the per-problem summaries, 1 = best (minimization).

    Ranks are scale-free, so cross-problem aggregation does not depend on
    per-problem fitness magnitudes or on min-max normalization constants
    (which change whenever the compared set changes).
    """
    algos = list(means)
    ranks = rankdata([means[a] for a in algos])
    return {algo: float(rank) for algo, rank in zip(algos, ranks)}


def match_problem(problem_name, n_vars):
    matched = next(
        (
            prob
            for prob in problems
            if prob.name() == problem_name
            and (n_vars == -1 or prob.number_of_variables() == n_vars)
        ),
        None,
    )
    if matched is not None:
        return matched

    same_name = next((prob for prob in problems if prob.name() == problem_name), None)
    if same_name is None or n_vars == -1:
        return None

    try:
        if hasattr(same_name, "instance_seed"):
            return create_seeded_problem(type(same_name), n_vars, BENCHMARK_BASE_SEED)
        return type(same_name)(n_vars)
    except Exception as e:
        print(f"Could not rebuild problem {problem_name} at dimension {n_vars}: {e}")
        return None


def plot_all_from_h5(file_path):
    if isinstance(file_path, list):
        plot_combined_data_from_h5(file_path)
        return
    loaded_data = load_data(file_path)

    for problem_data in loaded_data:
        problem_name = problem_data['problem']
        n_vars = problem_data['n_vars']
        results = problem_data['results']

        matched_problem = match_problem(problem_name, n_vars)

        if matched_problem is None:
            print(f"Problem {problem_name} (dimension {n_vars}) not found in the setup experiment list.")
            continue

        first_valid_result = next(
            (
                algo_data
                for algo_data in results.values()
                if isinstance(algo_data, dict)
                   and isinstance(algo_data.get("data"), np.ndarray)
                   and algo_data["data"].size > 0
            ),
            None,
        )

        if first_valid_result is None:
            print(
                f"No valid result data found for "
                f"{problem_name}, dimension {n_vars}."
            )
            continue

        no_of_runs = first_valid_result["data"].shape[0]

        # Directory to save plots for the specific problem
        dimensions_dir = f"{results_dir}/dim{n_vars}_runs{no_of_runs}"
        make_dir(dimensions_dir)

        # Plotting all required graphs
        plot_results(results, matched_problem, dimensions_dir, max_evaluations, no_of_runs, algorithm_colors)
        plot_results_with_std(results, matched_problem, dimensions_dir, max_evaluations, no_of_runs, algorithm_colors)
        plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                              no_of_runs=no_of_runs, algorithms_to_compare=algorithms.keys(),
                              results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting box plots for each individual algorithm
        for algorithm in algorithms.keys():
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=no_of_runs, algorithms_to_compare=[algorithm],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # # Plotting box plots comparing PGxHEA algorithms with GA and PSO
        # for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
        #     plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
        #                           no_of_runs=no_of_runs, algorithms_to_compare=[algorithm, 'GA', 'PSO'],
        #                           results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting final box plot comparing all algorithms
        plot_final_box(results, matched_problem, dimensions_dir, algorithm_colors)
        plot_final_raincloud(results, matched_problem, dimensions_dir, algorithm_colors)
        plot_final_petit_prince(results, matched_problem, dimensions_dir, algorithm_colors)





def load_data(filepath, problems=None, algorithms=None):
    """Load experiment data from an HDF5 file.

    Parameters
    ----------
    filepath : str
        Path to the HDF5 file produced by run_all_experiments[_multi].
    problems : list[str] | None
        If given, only load these problem groups (exact names as stored).
    algorithms : list[str] | None
        If given, only load these algorithm groups within each problem.

    Returns a list of problem dicts compatible with combine_data() and all
    statistical / plotting functions:
      [{'problem': str, 'n_vars': int, 'results': {algo: {'data': ndarray, ...}}}]
    """
    result = []
    try:
        with h5py.File(filepath, 'r') as f:
            problem_keys = [k for k in f.keys() if problems is None or k in problems]
            for problem_name in problem_keys:
                problem_grp = f[problem_name]
                n_vars = int(problem_grp.attrs.get('n_vars', -1))
                results = {}
                algo_keys = [k for k in problem_grp.keys() if algorithms is None or k in algorithms]
                for algo_name in algo_keys:
                    grp = problem_grp[algo_name]
                    fitness_curves = grp['fitness_curves'][:]
                    final_fitness = grp['final_fitness'][:]
                    run_times = grp['run_times'][:]
                    seeds = grp['seeds'][:].tolist()
                    valid_final = final_fitness[np.isfinite(final_fitness)]
                    results[algo_name] = {
                        'data': fitness_curves,
                        'final_fitness': final_fitness,
                        'run_times': run_times,
                        'seeds': seeds,
                        'avg_fitness': float(grp.attrs.get('avg_fitness',
                                             np.mean(valid_final) if valid_final.size > 0 else float('inf'))),
                        'std_dev': float(grp.attrs.get('std_dev',
                                         np.std(valid_final) if valid_final.size > 0 else float('nan'))),
                        'avg_time': float(grp.attrs.get('avg_time', np.mean(run_times))),
                    }
                if results:
                    result.append({'problem': problem_name, 'n_vars': n_vars, 'results': results})
        print(f"Successfully loaded data from: {filepath}")
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading file {filepath}: {e}")
        return None
    return result if result else None


def export_h5_subset(src_path, dst_path, problems=None, algorithms=None):
    """Write a filtered copy of an HDF5 experiment file for sharing.

    Parameters
    ----------
    src_path : str
        Source HDF5 file (produced by run_all_experiments[_multi]).
    dst_path : str
        Destination path for the exported file.
    problems : list[str] | None
        Problem groups to include; None means all.
    algorithms : list[str] | None
        Algorithm groups to include; None means all.

    Example
    -------
    # Share only PSO and PGCHEA results for Rastrigin:
    export_h5_subset('dim1000_runs50.h5', 'share_rastrigin_pso_pgchea.h5',
                     problems=['Rastrigin'], algorithms=['PSO', 'PGCHEA'])
    """
    with h5py.File(src_path, 'r') as src, h5py.File(dst_path, 'w') as dst:
        for attr_key, attr_val in src.attrs.items():
            dst.attrs[attr_key] = attr_val

        problem_keys = [k for k in src.keys() if problems is None or k in problems]
        for problem_name in problem_keys:
            src_prob = src[problem_name]
            dst_prob = dst.require_group(problem_name)
            for attr_key, attr_val in src_prob.attrs.items():
                dst_prob.attrs[attr_key] = attr_val

            algo_keys = [k for k in src_prob.keys() if algorithms is None or k in algorithms]
            for algo_name in algo_keys:
                src.copy(f'{problem_name}/{algo_name}', dst_prob, name=algo_name)

    print(f"Exported subset to {dst_path} "
          f"(problems={problems or 'all'}, algorithms={algorithms or 'all'})")


from decimal import Decimal

def incremental_mean(data_iter):
    mean = Decimal(0)  # Initialize mean as a Decimal
    count = 0
    for x in data_iter:
        count += 1
        mean += (x - mean) / count
    return mean



def incremental_std_dev(data_iter, mean):
    M2 = Decimal(0)  # Initialize as Decimal
    count = 0
    for x in data_iter:
        count += 1
        delta = x - mean
        M2 += delta * (x - mean)  # Ensure all operations use Decimal
    variance = M2 / count if count > 1 else Decimal('nan')  # Use Decimal equivalent of nan
    return float(variance ** Decimal(0.5))  # Conve




def calculate_incrementally(arr):
    arr = [Decimal(x) for x in arr]
    mean_dec = incremental_mean(arr)
    std_dev    = incremental_std_dev(arr, mean_dec)

    mean = float(mean_dec)
    return mean, std_dev


def plot_combined_data_from_h5(pickle_files):
    data_list = [
        load_data(file)
        for file in pickle_files
    ]

    combined_data = combine_data(data_list)

    for instance_key, problem_data in combined_data.items():

        problem_name = problem_data["problem"]
        n_vars = problem_data["n_vars"]
        instance_runs = problem_data["runs"]
        results = problem_data["results"]

        # Match both problem identity and dimension.
        matched_problem = match_problem(problem_name, n_vars)

        if matched_problem is None:
            print(
                f"Problem {problem_name} with dimension {n_vars} "
                f"not found in the current setup experiment list."
            )
            continue

        dimensions_dir = (
            f"{results_dir}/dim{n_vars}_runs{instance_runs}/plots"
        )

        make_dir(dimensions_dir)

        # if results:
        #     plot_results_with_annotations_legend(results, matched_problem, dimensions_dir, max_evaluations,
        #                                          instance_runs, algorithm_colors, "All algorithms")

        # # Plotting all required graphs
        # plot_results(results, matched_problem, dimensions_dir, max_evaluations, total_runs, algorithm_colors)
        # plot_results_with_annotations(results, matched_problem, dimensions_dir, max_evaluations, total_runs,
        #                               algorithm_colors)
        # plot_results_with_std(results, matched_problem, dimensions_dir, max_evaluations, total_runs, algorithm_colors)
        # plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
        #                       no_of_runs=total_runs, algorithms_to_compare=algorithms.keys(),
        #                       results_dir=dimensions_dir, algorithm_colors=algorithm_colors)
        #
        # # Plotting box plots for each individual algorithm
        # for algorithm in algorithms.keys():
        #     plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
        #                           no_of_runs=total_runs, algorithms_to_compare=[algorithm],
        #                           results_dir=dimensions_dir, algorithm_colors=algorithm_colors)
        #
        # # # Plotting box plots comparing PGxHEA algorithms with GA and PSO
        # # for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
        # #     plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
        # #                           no_of_runs=total_runs, algorithms_to_compare=[algorithm, 'GA', 'PSO'],
        # #                           results_dir=dimensions_dir, algorithm_colors=algorithm_colors)
        #
        # # Plotting final box plot comparing all algorithms
        # plot_final_box(results, matched_problem, dimensions_dir, algorithm_colors)
        # plot_final_raincloud(results, matched_problem, dimensions_dir, algorithm_colors)
        # plot_final_raincloud(results, matched_problem, dimensions_dir, algorithm_colors, adaptive_height=True)
        # plot_final_petit_prince(results, matched_problem, dimensions_dir, algorithm_colors)
        # plot_final_petit_prince(results, matched_problem, dimensions_dir, algorithm_colors, adaptive_width=True)

        for group_name, algorithm_list in group_of_algorithms.items():
            filtered_results = {algo: problem_data['results'][algo] for algo in algorithm_list if
                                algo in problem_data['results']}

            if not filtered_results or set(filtered_results.keys()) == {'PSO'}:
                print(f"Skipping {group_name}, no valid algorithms found (only 'PSO' or empty).")
                continue

            # plot_results(filtered_results, matched_problem, dimensions_dir, max_evaluations, instance_runs,
            #              algorithm_colors, group_name)

            # plot_results_with_annotations(filtered_results, matched_problem, dimensions_dir, max_evaluations,
            #                               instance_runs, algorithm_colors, group_name)

            plot_results_with_annotations_legend(filtered_results, matched_problem, dimensions_dir, max_evaluations,
                                          instance_runs, algorithm_colors, group_name)

            # plot_results_with_annotations_legend(filtered_results, matched_problem, dimensions_dir, max_evaluations,
            #                                      instance_runs, algorithm_colors, group_name, log_scale=True)

            # plot_results_with_std(filtered_results, matched_problem, dimensions_dir, max_evaluations, instance_runs,
            #                       algorithm_colors, group_name)

            # plot_results_with_average(filtered_results, matched_problem, dimensions_dir, max_evaluations, instance_runs,
            #                       algorithm_colors, group_name)

            # plot_box_at_intervals(filtered_results, matched_problem, max_evaluations=max_evaluations,
            #                       no_of_runs=instance_runs,
            #                       algorithms_to_compare=list(filtered_results.keys()), results_dir=dimensions_dir,
            #                       algorithm_colors=algorithm_colors, group_name=group_name)

            # plot_final_box(filtered_results, matched_problem, dimensions_dir, algorithm_colors, group_name)

            # plot_final_raincloud(filtered_results, matched_problem, dimensions_dir, algorithm_colors, group_name)

            # plot_final_raincloud(filtered_results, matched_problem, dimensions_dir, algorithm_colors, group_name,
            #                      adaptive_height=True)

            # plot_final_petit_prince(filtered_results, matched_problem, dimensions_dir, algorithm_colors)

            plot_final_petit_prince(filtered_results, matched_problem, dimensions_dir, algorithm_colors)

            # plot_final_petit_prince(filtered_results, matched_problem, dimensions_dir, algorithm_colors,
            #                         log_scale=True)

            # plot_final_petit_prince(filtered_results, matched_problem, dimensions_dir, algorithm_colors,
            #                         adaptive_width=True)


def extract_best_algorithms_from_experiment_data(
    aggregated_data_dict
):
    """
    Extract the best algorithm(s) separately for each
    problem/dimension instance.
    """
    best_algorithms_per_problem = {}

    for instance_key, problem_data in aggregated_data_dict.items():

        problem_name = problem_data["problem"]
        dimension = problem_data.get("n_vars", "Unknown")
        results = problem_data.get("results", {})

        if not results:
            continue

        avg_fitness_values = {}

        for algo_name, algo_data in results.items():

            if (
                isinstance(algo_data, dict)
                and "avg_fitness" in algo_data
                and np.isfinite(algo_data["avg_fitness"])
            ):
                avg_fitness_values[algo_name] = (
                    algo_data["avg_fitness"]
                )

        if not avg_fitness_values:
            continue

        try:
            best_value = min(avg_fitness_values.values())

            best_algos = [
                algo
                for algo, fitness
                in avg_fitness_values.items()
                if np.isclose(fitness, best_value)
            ]

            best_algorithms_per_problem.setdefault(
                problem_name,
                {},
            )

            best_algorithms_per_problem[
                problem_name
            ][dimension] = best_algos

        except ValueError:
            print(
                f"Warning: No valid fitness values to compare "
                f"for {problem_name} dim {dimension}."
            )

    return best_algorithms_per_problem


def kruskal_wallis_with_posthoc(pickle_files, perform_shapiro=True, perform_posthoc=True):
    """
    Perform Kruskal-Wallis and posthoc tests using dynamically extracted
    the best algorithms from combined data.
    """
    all_normal = False
    non_significant_dunn = []
    non_significant_tukey = []
    non_significant_vs_best_dunn = []
    non_significant_vs_best_tukey = []

    # 1. Load data from all pickle files
    print("Loading data from pickle files...")
    data_list = [load_data(file) for file in pickle_files]
    valid_data_list = [d for d in data_list if d is not None] # Filter out loading errors

    if not valid_data_list:
        print("Error: No valid data loaded from any pickle files.")
        return

    # 2. Combine data using your existing function
    print("Combining loaded data...")
    # combine_data handles the different list/dict structures internally
    combined_data_dict = combine_data(valid_data_list)

    if not combined_data_dict:
        print("Error: Data combination resulted in empty dictionary.")
        return

    run_counts = sorted({
        problem_data["runs"]
        for problem_data in combined_data_dict.values()
    })

    print(
        f"Data combined successfully. "
        f"Instances: {len(combined_data_dict)}, "
        f"run counts: {run_counts}"
    )

    # 3. Extract best algorithms from the *combined* data
    print("Extracting best algorithms from combined data...")
    # Pass the dictionary output of combine_data
    best_algorithms = extract_best_algorithms_from_experiment_data(combined_data_dict)
    print("Best algorithms extracted:", best_algorithms)

    # 4. Iterate through the COMBINED data for statistical analysis
    for instance_key, problem_data in combined_data_dict.items():

        problem_name = problem_data["problem"]
        dimension = problem_data.get(
            "n_vars",
            "Unknown",
        )
        results = problem_data.get('results', {})

        # Get the best algorithms determined *after* combining all runs
        best_algos = best_algorithms.get(problem_name, {}).get(dimension, [])

        if not results:
            print(f"No results found for Problem: {problem_name}, Dimension: {dimension}. Skipping analysis.")
            continue

        print(f"\nPerforming analysis for Problem: {problem_name}, Dimension: {dimension}")
        if best_algos: print(f"Best algorithm(s) identified: {best_algos}")
        else: print("No best algorithm clearly identified (check extraction logic or data).")

        # Prepare data for statistical tests from the aggregated results
        final_fitness_values = {}
        algorithms_in_analysis = []
        data_for_test = []

        for algo, algo_data in results.items():
            if 'data' in algo_data and isinstance(algo_data['data'], np.ndarray) and algo_data['data'].size > 0:
                 if algo_data['data'].ndim == 2:
                      final_values = algo_data['data'][:, -1]
                 elif algo_data['data'].ndim == 1:
                      final_values = algo_data['data']
                 else: continue # Skip unexpected shape

                 finite_final_values = final_values[np.isfinite(final_values)]
                 if finite_final_values.size > 0:
                      final_fitness_values[algo] = finite_final_values
                      algorithms_in_analysis.append(algo)
                      data_for_test.append(finite_final_values)
                 else: print(f"Warning: No finite final fitness values for {algo}. Skipping.")
            else: print(f"Warning: Missing/invalid 'data' for {algo}. Skipping.")

        if len(final_fitness_values) < 2:
             print("\tNot enough algorithm groups (>1) with valid data for statistical comparison. Skipping tests.")
             continue

        if perform_shapiro:
            normality_results = {algo: shapiro(values) for algo, values in final_fitness_values.items()}
            all_normal = all(p > 0.05 for _, p in normality_results.values())
            print("\tShapiro-Wilk Test Results (p-values):")
            for algo, (stat, p_value) in normality_results.items():
                print(f"{algo}: Statistic={stat:.4f}, P-value={p_value:.4e}")
            if all_normal:
                print("\tAll groups are normally distributed. Performing ANOVA.")
                stat, p = f_oneway(*final_fitness_values.values())
            else:
                print("\tNot all groups are normally distributed. Performing Kruskal-Wallis test.")
                stat, p = kruskal(*final_fitness_values.values())
            print(f"Test Statistic: {stat}")
            print(f"P-value: {p}")
        else:
            stat, p = kruskal(*final_fitness_values.values())
            print(f"\tKruskal-Wallis Statistic: {stat}")
            print(f"P-value: {p}")

        if p < 0.05 and perform_posthoc:
            data = list(final_fitness_values.values())
            algo_names = list(final_fitness_values.keys())

            if perform_shapiro and all_normal:
                combined_data = pd.Series(data[0])
                groups = [algo_names[0]] * len(data[0])
                for i in range(1, len(data)):
                    combined_data = combined_data._append(pd.Series(data[i]))
                    groups.extend([algo_names[i]] * len(data[i]))

                tukey_results = pairwise_tukeyhsd(combined_data, groups)
                print("\nTukey's HSD test pairwise comparison results:")
                print(tukey_results.summary())

                for res in tukey_results._results_table.data[1:]:
                    group1, group2, meandiff, p_adj, lower, upper, reject = res
                    if not reject:
                        non_significant_tukey.append((problem_name, dimension, group1, group2, p_adj))
                        if group1 in best_algos or group2 in best_algos:
                            non_significant_vs_best_tukey.append((problem_name, dimension, group1, group2, p_adj))
            else:
                # Holm for a consistent FWER-control philosophy across this module.
                dunn_results = sp.posthoc_dunn(data, p_adjust='holm')
                dunn_results.index = algo_names
                dunn_results.columns = algo_names
                print("Dunn's test pairwise comparison p-values:")
                print(dunn_results)

                for i, algo1 in enumerate(algo_names):
                    for j, algo2 in enumerate(algo_names):
                        if i < j:
                            p_val = dunn_results.iloc[i, j]
                            if p_val >= 0.05:
                                non_significant_dunn.append((problem_name, dimension, algo1, algo2, p_val))
                                if algo1 in best_algos or algo2 in best_algos:
                                    non_significant_vs_best_dunn.append(
                                        (problem_name, dimension, algo1, algo2, p_val))

    # Output Results
    print("\nNon-Significant Results from Dunn's Test (LaTeX Format):")
    for result in non_significant_dunn:
        problem, dimension, algo1, algo2, p_val = result
        print(f"{problem} & {dimension} & {algo1} vs {algo2} & {p_val:.4f} \\\\")
    print(f"Size: {len(non_significant_dunn)}")

    print("\nNon-Significant Results from Tukey's HSD Test (LaTeX Format):")
    for result in non_significant_tukey:
        problem, dimension, group1, group2, p_adj = result
        print(f"{problem} & {dimension} & {group1} vs {group2} & {p_adj:.4f} \\\\")
    print(f"Size: {len(non_significant_tukey)}")

    print("\nNon-Significant Results vs Best Algorithms from Dunn's Test (LaTeX Format):")
    for result in non_significant_vs_best_dunn:
        problem, dimension, algo1, algo2, p_val = result
        print(f"{problem} & {dimension} & {algo1} vs {algo2} & {p_val:.4f} \\\\")
    print(f"Size: {len(non_significant_vs_best_dunn)}")

    print("\nNon-Significant Results vs Best Algorithms from Tukey's HSD Test (LaTeX Format):")
    for result in non_significant_vs_best_tukey:
        problem, dimension, group1, group2, p_adj = result
        print(f"{problem} & {dimension} & {group1} vs {group2} & {p_adj:.4f} \\\\")
    print(f"Size: {len(non_significant_vs_best_tukey)}")



def get_data_group_key(loaded_data_item):
    """
    Inspects a loaded data item (list or dict) to find the dimension and run count.
    Returns a tuple (dimension, runs) or (None, None) if not found/consistent.
    """
    dimension = None
    runs = None

    if isinstance(loaded_data_item, list):
        if not loaded_data_item: return None, None # Empty list
        # Assume list contains problem dicts, check the first one
        first_problem_data = loaded_data_item[0]
        if not isinstance(first_problem_data, dict): return None, None
    elif isinstance(loaded_data_item, dict):
        # Assumes dict is a single problem dict
        first_problem_data = loaded_data_item
    else:
        return None, None # Unrecognized structure

    # Extract dimension
    dimension = first_problem_data.get("n_vars")

    if isinstance(dimension, np.integer):
        dimension = int(dimension)

    if dimension is None or not isinstance(dimension, int):
        print(
            f"Warning: Could not determine dimension from data item: "
            f"{first_problem_data.get('problem', 'Unknown')}"
        )
        return None, None

    # Extract runs - check first algorithm's data shape
    results = first_problem_data.get('results')
    if not results or not isinstance(results, dict): return dimension, None # Cannot determine runs

    try:
        first_algo_name = next(iter(results))
        first_algo_data = results[first_algo_name]
        if 'data' in first_algo_data and isinstance(first_algo_data['data'], np.ndarray):
            runs = first_algo_data['data'].shape[0] # Number of rows is number of runs
        else:
            return dimension, None # No data array to get runs from
    except StopIteration: # No algorithms in results
         return dimension, None
    except Exception as e:
         print(f"Warning: Error extracting run count from data item: {e}")
         return dimension, None

    # Optional: Add consistency check if data_item is a list with multiple problems
    if isinstance(loaded_data_item, list) and len(loaded_data_item) > 1:
        for other_problem_data in loaded_data_item[1:]:
             other_dim = other_problem_data.get('n_vars')
             # Add similar run check if needed
             if other_dim != dimension:
                  print(f"Warning: Inconsistent dimensions ({dimension} vs {other_dim}) within a single loaded list. Grouping based on first entry.")
                  # Decide how to handle - here we use the first one found
                  break

    return dimension, runs


# --- Refactored Function to Extract Results to CSV ---
def extract_results_to_csv(pickle_files, output_prefix="aggregated_results", base_results_dir=results_dir):
    """
    Loads data, groups by actual dimension/runs found IN the data, combines
    within groups, and writes separate summary CSV files.
    """
    print(f"Processing {len(pickle_files)} pickle files for CSV extraction...")

    # 1. Load all data first, keeping track of original filepath if needed for context
    loaded_data_map = {fp: load_data(fp) for fp in pickle_files}
    valid_loaded_data = {fp: data for fp, data in loaded_data_map.items() if data is not None}

    if not valid_loaded_data:
        print("Error: No valid data loaded from any pickle files.")
        return

    # 2. Group loaded data objects by inspected dimension and runs
    grouped_data_objects = defaultdict(list)
    print("Inspecting loaded data and grouping...")
    for filepath, data_item in valid_loaded_data.items():
        dimension, runs = get_data_group_key(data_item)
        if dimension is not None and runs is not None:
            group_key = (dimension, runs)
            # Append the actual data object to the group list
            grouped_data_objects[group_key].append(data_item)
            # print(f"  Grouped {filepath} into Dim={dimension}, Runs={runs}")
        else:
            print(f"  Skipping data from {filepath} due to missing/inconsistent dimension or run info.")

    if not grouped_data_objects:
        print("Error: Could not group any valid data. No CSVs generated.")
        return

    # 3. Process each group
    for (dimension, runs), data_objects_in_group in grouped_data_objects.items():
        print(f"\n--- Processing Group: Dimension={dimension}, Runs={runs} ---")
        print(f"  Data sources in group: {len(data_objects_in_group)}")

        # 4. Combine data for the current group
        print("  Combining data for this group...")
        # Pass the list of actual data objects (which can be lists or dicts)
        combined_data_dict = combine_data(data_objects_in_group)

        if not combined_data_dict:
            print("  Error: Data combination resulted in empty dictionary for this group. Skipping CSV generation.")
            continue
        # Verify run count consistency
        instance_run_counts = {
            problem_data["runs"]
            for problem_data in combined_data_dict.values()
            if problem_data.get("runs", 0) > 0
        }

        if not instance_run_counts:
            print(
                "  Warning: No valid run counts found "
                "after aggregation."
            )
            continue

        if len(instance_run_counts) == 1:
            effective_runs = next(iter(instance_run_counts))
        else:
            effective_runs = "mixed"

            print(
                f"  Warning: Aggregated benchmark instances "
                f"have different run counts: "
                f"{sorted(instance_run_counts)}"
            )

        print(
            f"  Data combined. "
            f"Run counts found: "
            f"{sorted(instance_run_counts)}"
        )

        # 5. Determine Output Path and Filename
        group_dir = os.path.join(base_results_dir, f"dim{dimension}_runs{effective_runs}")
        make_dir(group_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv_filename = os.path.join(group_dir, f"{output_prefix}_dim{dimension}_runs{effective_runs}_{timestamp}.csv")

        # 6. Prepare and Write CSV
        header = ['Algorithm', 'Problem', 'Variables', 'Runs', 'Average Final Fitness',
                  'Standard deviation', 'Average Computing Time (s)']
        print(f"  Writing aggregated results to: {output_csv_filename}")
        try:
            with open(output_csv_filename, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(header)
                for instance_key in sorted(
                        combined_data_dict.keys(),
                        key=lambda key: (key[0], key[1]),
                ):
                    problem_data = combined_data_dict[instance_key]

                    problem_name = problem_data["problem"]
                    data_dimension = problem_data.get(
                        "n_vars",
                        "N/A",
                    )

                    if (
                            data_dimension != dimension
                            and data_dimension != "N/A"
                    ):
                        print(
                            f"    Internal Warning: Combined data dimension "
                            f"{data_dimension} differs from group key "
                            f"{dimension} for problem '{problem_name}'."
                        )

                    results = problem_data.get("results", {})

                    for algo_name in sorted(results.keys()):
                        algo_data = results[algo_name]

                        runs_count = algo_data.get(
                            "runs",
                            problem_data.get("runs", "N/A"),
                        )

                        avg_fitness = algo_data.get(
                            "avg_fitness",
                            float("nan"),
                        )

                        std_dev = algo_data.get(
                            "std_dev",
                            float("nan"),
                        )

                        avg_time = algo_data.get(
                            "avg_time",
                            float("nan"),
                        )

                        writer.writerow([
                            algo_name,
                            problem_name,
                            data_dimension,
                            runs_count,
                            (
                                f"{avg_fitness}"
                                if np.isfinite(avg_fitness)
                                else "Inf/NaN"
                            ),
                            (
                                f"{std_dev}"
                                if np.isfinite(std_dev)
                                else "NaN"
                            ),
                            (
                                f"{avg_time}"
                                if np.isfinite(avg_time)
                                else "NaN"
                            ),
                        ])
            print(f"  CSV file generation complete for group (Dim={dimension}, Runs={effective_runs}).")
        except IOError as e: print(f"  Error writing CSV file '{output_csv_filename}': {e}")
        except Exception as e: print(f"  An unexpected error occurred during CSV generation for group (Dim={dimension}, Runs={effective_runs}): {e}")

    print("\nFinished processing all groups.")


# ---------------------------------------------
# Wilcoxon (rank-sum) vs PSO baselines – Better/Worse/Similar
# ---------------------------------------------
def wilcoxon_rank_sum_vs_baselines(
    pickle_files,
        baselines_display_to_key=None,
    alpha=0.05,
    lower_is_better=True,
    min_group_size=2,
    output_prefix="wilcoxon_summary",
    base_results_dir=results_dir,
    algo_groups=None,
    print_examples=False,
):

    if baselines_display_to_key is None:
        baselines_display_to_key = OrderedDict([
            ("Canonical PSO", "PSO"),
            ("CMA-ES", "CMAES"),
            ("L-SHADE", "LSHADE"),
        ])
    allowed_set = (set(algo_groups.keys()) if algo_groups else set()) \
                  | set(baselines_display_to_key.values())

    print("Loading data from pickle files...")
    data_list = [load_data(fp) for fp in pickle_files]
    valid_data_list = [d for d in data_list if d is not None]
    if not valid_data_list:
        print("Error: No valid data loaded from any pickle files.")
        return None

    print("Combining loaded data...")
    combined_data_dict = combine_data(valid_data_list)
    if not combined_data_dict:
        print("Error: Data combination resulted in empty dictionary.")
        return None
    run_counts = sorted({
        problem_data["runs"]
        for problem_data in combined_data_dict.values()
    })

    print(
        f"Data combined successfully. "
        f"Instances: {len(combined_data_dict)}, "
        f"run counts: {run_counts}"
    )

    def _final_values_from_algo_data(algo_data):
        if 'data' not in algo_data or not isinstance(algo_data['data'], np.ndarray):
            return None
        arr = algo_data['data']
        if arr.size == 0:
            return None
        if arr.ndim == 2:
            final_vals = arr[:, -1]
        elif arr.ndim == 1:
            final_vals = arr
        else:
            return None
        final_vals = final_vals[np.isfinite(final_vals)]
        return final_vals if final_vals.size > 0 else None


    counts = defaultdict(lambda: {disp: {"better": 0, "worse": 0, "similar": 0}
                                  for disp in baselines_display_to_key.keys()})
    total_test_cases_by_baseline = {disp: 0 for disp in baselines_display_to_key.keys()}

    print("\nStarting Wilcoxon (rank-sum) analysis...")
    for instance_key, problem_data in combined_data_dict.items():

        problem_name = problem_data["problem"]
        dim = problem_data.get(
            "n_vars",
            "Unknown",
        )

        results = problem_data.get(
            "results",
            {},
        )
        if not results:
            continue

        finals = {}
        for algo, a_data in results.items():
            if algo not in allowed_set:
                continue  # pomijamy intruzów
            v = _final_values_from_algo_data(a_data)
            if v is not None and v.size >= min_group_size:
                finals[algo] = v

        baseline_vectors = {}
        for disp, key in baselines_display_to_key.items():
            if key in finals:
                baseline_vectors[disp] = finals[key]

        if not baseline_vectors:
            continue

        for algo, vals in finals.items():
            if algo in baselines_display_to_key.values():
                continue

            for disp, base_vals in baseline_vectors.items():
                total_test_cases_by_baseline[disp] += 1

                try:
                    stat = mannwhitneyu(vals, base_vals, alternative='two-sided', method='auto')
                    p = stat.pvalue
                except Exception as e:
                    print(f"[WARN] mannwhitneyu failed on {problem_name} (dim={dim}) {algo} vs {disp}: {e}")
                    counts[algo][disp]["similar"] += 1
                    continue

                algo_med = np.median(vals)
                base_med = np.median(base_vals)

                if p < alpha:
                    if lower_is_better:
                        if algo_med < base_med:
                            counts[algo][disp]["better"] += 1
                            if print_examples:
                                print(f"[+] {algo} better than {disp} on {problem_name} (dim={dim}) p={p:.3g} med {algo_med:.4g}<{base_med:.4g}")
                        elif algo_med > base_med:
                            counts[algo][disp]["worse"]  += 1
                            if print_examples:
                                print(f"[-] {algo} worse  than {disp} on {problem_name} (dim={dim}) p={p:.3g} med {algo_med:.4g}>{base_med:.4g}")
                        else:
                            counts[algo][disp]["similar"] += 1
                    else:
                        if algo_med > base_med:
                            counts[algo][disp]["better"] += 1
                        elif algo_med < base_med:
                            counts[algo][disp]["worse"]  += 1
                        else:
                            counts[algo][disp]["similar"] += 1
                else:
                    counts[algo][disp]["similar"] += 1

    col_tuples = []
    for disp in baselines_display_to_key.keys():
        col_tuples.extend([
            (f"vs. {disp}", "Better (+)"),
            (f"vs. {disp}", "Worse (-)"),
            (f"vs. {disp}", "Similar (=)"),
        ])
    columns = pd.MultiIndex.from_tuples(col_tuples)

    algos_sorted = sorted(counts.keys())
    data_rows = []
    for algo in algos_sorted:
        row = []
        for disp in baselines_display_to_key.keys():
            c = counts[algo][disp]
            row.extend([c["better"], c["worse"], c["similar"]])
        data_rows.append(row)

    summary_df = pd.DataFrame(data_rows, index=algos_sorted, columns=columns)

    out_dir = os.path.join(base_results_dir, "wilcoxon_vs_baselines")
    make_dir(out_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"{output_prefix}_{timestamp}.csv")
    summary_df.to_csv(csv_path, encoding="utf-8")
    print(f"\nSaved Wilcoxon summary CSV to: {csv_path}")

    latex_str = None
    if algo_groups is not None and len(algo_groups) > 0:
        grouped = OrderedDict()
        for algo in algos_sorted:
            grp = algo_groups.get(algo, "Other")
            grouped.setdefault(grp, []).append(algo)

        left_col_name = r"\textbf{Algorithm}"
        header = []
        header.append(r"\begin{table}[h!]")
        header.append(r"\centering")
        header.append(r"\caption{Summary of Wilcoxon rank-sum test results ($p<%.3g$) for all proposed algorithms against PSO baselines.}" % alpha)
        header.append(r"\label{tab:wilcoxon-summary}")
        header.append(r"\resizebox{\textwidth}{!}{")
        header.append(r"\begin{tabular}{l" + "ccc"*len(baselines_display_to_key) + "}")
        header.append(r"\toprule")

        upper = [""]
        for disp in baselines_display_to_key.keys():
            upper.append(r"\multicolumn{3}{c}{\textbf{vs. %s}}" % disp)
        header.append(" & ".join(upper) + r" \\")
        cmis = []
        start_col = 2
        for i in range(len(baselines_display_to_key)):
            end_col = start_col + 2
            cmis.append(r"\cmidrule(lr){%d-%d}" % (start_col, end_col))
            start_col = end_col + 1
        header.append(" ".join(cmis))

        lower = [left_col_name]
        for _ in baselines_display_to_key.keys():
            lower.extend([r"\textbf{Better (+)}", r"\textbf{Worse (-)}", r"\textbf{Similar (=)}"])
        header.append(" & ".join(lower) + r" \\")
        header.append(r"\midrule")

        rows = []
        for grp, algos in grouped.items():
            rows.append(r"\textbf{%s} \\" % grp)
            for algo in algos:
                fields = [algo]
                for disp in baselines_display_to_key.keys():
                    c = counts[algo][disp]
                    fields.extend([str(c["better"]), str(c["worse"]), str(c["similar"])])
                rows.append(" & ".join(fields) + r" \\")
            rows.append(r"\midrule")

        footer = [r"\bottomrule", r"\end{tabular}", r"}", r"\end{table}"]

        latex_str = "\n".join(header + rows + footer)
        tex_path = os.path.join(out_dir, f"{output_prefix}_{timestamp}.tex")
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(latex_str)
        print(f"Saved LaTeX table to: {tex_path}")

    print("\nTotal test cases per baseline (problem × dimension where both had data):")
    for disp, n in total_test_cases_by_baseline.items():
        print(f"  {disp}: {n}")

    return {
        "summary_df": summary_df,
        "csv_path": csv_path,
        "latex": latex_str
    }


from scipy.stats import friedmanchisquare, wilcoxon
import numpy as np
from itertools import combinations


def friedman_wilcoxon_algorithm_groups(pickle_files, algo_groups):
    """
    Min-max normalizes fitness per problem, averages them by the provided
    algorithm taxonomy groups, and runs a Friedman test + pairwise Wilcoxons.
    """
    # 1. Dynamically extract unique group names (preserving order)
    group_names = list(dict.fromkeys(algo_groups.values()))

    # 2. Load and combine data
    print("Loading data from pickle files...")
    data_list = [load_data(fp) for fp in pickle_files]
    valid_data_list = [d for d in data_list if d is not None]
    if not valid_data_list:
        print("Error: No valid data loaded.")
        return

    combined_data_dict = combine_data(valid_data_list)
    if not combined_data_dict:
        print("Error: Combined data is empty.")
        return

    # Initialize storage for group means per problem
    group_scores_per_problem = {g: [] for g in group_names}

    # 3. Per problem: rank the compared algorithms (scale-free), then average
    #    member ranks per group. Requires the full compared set per problem so
    #    the rank blocks are comparable.
    compared_algos = list(algo_groups)
    for instance_key, problem_data in combined_data_dict.items():
        results = problem_data.get('results', {})

        algo_means = final_fitness_means(results, compared_algos)
        if len(algo_means) != len(compared_algos):
            continue

        ranks = rank_within_problem(algo_means)

        for g in group_names:
            g_algos = [a for a, grp in algo_groups.items() if grp == g]
            group_scores_per_problem[g].append(float(np.mean([ranks[a] for a in g_algos])))

    n_cases = len(group_scores_per_problem[group_names[0]])
    if n_cases < 2:
        print("Not enough complete matched data across all groups to perform statistical tests.")
        return

    # 4. Perform Friedman Test
    stat, p_friedman = friedmanchisquare(*[group_scores_per_problem[g] for g in group_names])

    print(f"\n=== Taxonomy Group Analysis (Per-problem mean ranks, N={n_cases} paired benchmarks) ===")
    for g in group_names:
        print(f"  {g} Global Mean Rank: {np.mean(group_scores_per_problem[g]):.4f}")

    print(f"\nFriedman Test Statistic: {stat:.4f}, p-value: {p_friedman:.5e}")

    # 5. Perform Pairwise Wilcoxon if Friedman is significant
    if p_friedman < 0.05:
        print("\nFriedman test is SIGNIFICANT (p < 0.05). Proceeding with pairwise Wilcoxon tests:")
        for g1, g2 in combinations(group_names, 2):
            try:
                w_stat, p_wilc = wilcoxon(group_scores_per_problem[g1], group_scores_per_problem[g2])
                mean_diff = np.mean(group_scores_per_problem[g1]) - np.mean(group_scores_per_problem[g2])
                winner = g1 if mean_diff < 0 else g2  # Lower mean rank is better
                sig_star = "*" if p_wilc < 0.05 else ""
                print(f"  {g1} vs {g2}: p={p_wilc:.4f}{sig_star} --> {winner} trended better")
            except Exception as e:
                print(f"  {g1} vs {g2}: Could not perform Wilcoxon test ({e})")
    else:
        print("\nNo significant difference among the groups overall (Friedman p >= 0.05).")
        print(
            "This implies that specific topographical advantages of individual algorithms balance out across the benchmark suite (No Free Lunch).")

def friedman_wilcoxon_algorithm_groups_with_holm(pickle_files, algo_groups):
    """
    Ranks the compared algorithms within each problem (1 = best, scale-free),
    averages the ranks by the provided algorithm taxonomy groups, and runs a
    Friedman test followed by pairwise Wilcoxon signed-rank tests with Holm
    correction.

    Parameters
    ----------
    pickle_files : list
        List of pickle files containing experimental results.

    algo_groups : dict
        Mapping from algorithm name to group name. If each algorithm should be
        compared individually, map each algorithm to itself.

    Returns
    -------
    dict
        Dictionary containing group scores, Friedman result, and Holm-corrected
        pairwise Wilcoxon post-hoc results.
    """

    # 1. Dynamically extract unique group names, preserving order
    group_names = list(dict.fromkeys(algo_groups.values()))

    # 2. Load and combine data
    print("Loading data from pickle files...")
    data_list = [load_data(fp) for fp in pickle_files]
    valid_data_list = [d for d in data_list if d is not None]

    if not valid_data_list:
        print("Error: No valid data loaded.")
        return None

    combined_data_dict = combine_data(valid_data_list)

    if not combined_data_dict:
        print("Error: Combined data is empty.")
        return None

    runs_by_instance = {
        instance_key: problem_data["runs"]
        for instance_key, problem_data
        in combined_data_dict.items()
    }

    # Initialize storage for group means per problem
    group_scores_per_problem = {g: [] for g in group_names}

    excluded_empty = 0
    excluded_incomplete = 0

    # 3. Per problem: rank the compared algorithms (scale-free), then average
    #    member ranks per group. Requires the full compared set per problem so
    #    the rank blocks are comparable.
    compared_algos = list(algo_groups)
    for instance_key, problem_data in combined_data_dict.items():
        results = problem_data.get("results", {})

        algo_means = final_fitness_means(results, compared_algos)

        if not algo_means:
            excluded_empty += 1
            continue

        if len(algo_means) != len(compared_algos):
            excluded_incomplete += 1
            continue

        ranks = rank_within_problem(algo_means)

        for g in group_names:
            g_algos = [a for a, grp in algo_groups.items() if grp == g]
            group_scores_per_problem[g].append(float(np.mean([ranks[a] for a in g_algos])))

    n_cases = len(group_scores_per_problem[group_names[0]])

    if n_cases < 2:
        print("Not enough complete matched data across all groups to perform statistical tests.")
        return None

    # Convert to arrays
    for g in group_names:
        group_scores_per_problem[g] = np.asarray(group_scores_per_problem[g], dtype=float)

    # 4. Friedman test
    stat, p_friedman = friedmanchisquare(
        *[group_scores_per_problem[g] for g in group_names]
    )

    print(f"\n=== Taxonomy Group Analysis (Per-problem mean ranks, N={n_cases} paired benchmarks) ===")
    for g in group_names:
        print(f"  {g} Global Mean Rank: {np.mean(group_scores_per_problem[g]):.4f}")

    print("\n=== Instance filtering summary ===")
    print(f"Total instances available: {len(combined_data_dict)}")
    print(f"Valid complete instances used: {n_cases}")
    print(f"Excluded empty/no relevant algos: {excluded_empty}")
    print(f"Excluded incomplete across compared algorithms: {excluded_incomplete}")

    print(f"\nFriedman Test Statistic: {stat:.4f}, p-value: {p_friedman:.5e}")

    pairwise_results = []

    # 5. Pairwise Wilcoxon post-hoc tests with Holm correction
    if p_friedman < 0.05:
        print(
            "\nFriedman test is SIGNIFICANT (p < 0.05). "
            "Proceeding with pairwise Wilcoxon signed-rank tests with Holm correction:"
        )

        raw_results = []
        raw_p_values = []

        for g1, g2 in combinations(group_names, 2):
            try:
                w_stat, p_raw = wilcoxon(
                    group_scores_per_problem[g1],
                    group_scores_per_problem[g2]
                )

                mean_g1 = float(np.mean(group_scores_per_problem[g1]))
                mean_g2 = float(np.mean(group_scores_per_problem[g2]))
                mean_diff = mean_g1 - mean_g2

                winner = g1 if mean_diff < 0 else g2  # lower mean rank is better

                raw_results.append({
                    "group_1": g1,
                    "group_2": g2,
                    "wilcoxon_stat": float(w_stat),
                    "p_raw": float(p_raw),
                    "mean_group_1": mean_g1,
                    "mean_group_2": mean_g2,
                    "mean_diff_group_1_minus_group_2": float(mean_diff),
                    "lower_mean_group": winner
                })

                raw_p_values.append(p_raw)

            except Exception as e:
                print(f"  {g1} vs {g2}: Could not perform Wilcoxon test ({e})")

        if raw_results:
            reject, p_adjusted, _, _ = multipletests(
                raw_p_values,
                alpha=0.05,
                method="holm"
            )

            for result, p_adj, is_significant in zip(raw_results, p_adjusted, reject):
                result["p_holm"] = float(p_adj)
                result["significant_holm"] = bool(is_significant)

                pairwise_results.append(result)

                sig_star = "*" if is_significant else ""

                if is_significant:
                    conclusion = f"{result['lower_mean_group']} significantly better"
                else:
                    conclusion = f"{result['lower_mean_group']} lower mean, not significant"

                print(
                    f"  {result['group_1']} vs {result['group_2']}: "
                    f"raw p={result['p_raw']:.4f}, "
                    f"Holm-adjusted p={result['p_holm']:.4f}{sig_star} "
                    f"--> {conclusion}"
                )

    else:
        print("\nNo significant difference among the groups overall (Friedman p >= 0.05).")
        print(
            "This result is consistent with the possibility that topographical advantages "
            "of individual algorithms balance out across the benchmark suite, but it does "
            "not prove a No Free Lunch effect."
        )

    return {
        "group_names": group_names,
        "group_scores_per_problem": group_scores_per_problem,
        "n_cases": n_cases,
        "friedman_statistic": float(stat),
        "friedman_p_value": float(p_friedman),
        "friedman_significant": bool(p_friedman < 0.05),
        "pairwise_results": pairwise_results,
        "excluded_empty": excluded_empty,
        "excluded_incomplete": excluded_incomplete,
        "runs_by_instance": runs_by_instance,
    }

def head_to_head_champions(pickle_files, algo_A="PSO", algo_B="ContrarianDefeatistPSO"):
    """Paired head-to-head comparison of two algorithms across the suite.

    Per problem, the winner is the algorithm with the lower mean final
    fitness; the sign test (exact binomial, ties dropped) then asks whether
    either wins on significantly more problems. Scale-free by construction -
    no cross-problem normalization is involved.
    """
    data_list = [load_data(fp) for fp in pickle_files]
    valid_data_list = [d for d in data_list if d is not None]
    combined_data_dict = combine_data(valid_data_list)

    wins_A = wins_B = ties = 0

    for instance_key, problem_data in combined_data_dict.items():
        results = problem_data.get('results', {})
        means = final_fitness_means(results, [algo_A, algo_B])
        if len(means) != 2:
            continue
        if means[algo_A] < means[algo_B]:
            wins_A += 1
        elif means[algo_B] < means[algo_A]:
            wins_B += 1
        else:
            ties += 1

    n_decisive = wins_A + wins_B
    print(f"\n=== Head-to-Head: {algo_A} vs {algo_B} "
          f"(N={n_decisive + ties} paired benchmarks) ===")
    print(f"Wins {algo_A}: {wins_A} | Wins {algo_B}: {wins_B} | Ties: {ties}")

    if n_decisive < 1:
        print("No decisive problems - nothing to test.")
        return

    p = binomtest(wins_A, n_decisive, 0.5).pvalue
    print(f"Sign test (exact binomial) p-value: {p:.4e}")
    if p < 0.05:
        winner = algo_A if wins_A > wins_B else algo_B
        print(f"Result: {winner} wins on significantly more problems (p < 0.05).")
    else:
        print("Result: Statistical TIE. Both are equally powerful but on different landscapes.")


import numpy as np
from scipy.stats import friedmanchisquare
import scikit_posthocs as sp
import pandas as pd


def all_vs_all_algorithm_stats(pickle_files, algos_to_compare=None, baseline="PSO"):
    """All-vs-all comparison over the benchmark suite.

    Blocked design: problems are blocks, algorithms are treatments, the
    per-problem summary is the mean final fitness. The Friedman omnibus and
    the Nemenyi-Friedman post-hoc both operate on within-problem ranks, so no
    cross-problem normalization is involved. Mean ranks are reported as the
    effect size.
    """
    # 1. Load and combine data
    valid_data = [d for d in [load_data(fp) for fp in pickle_files] if d is not None]
    if not valid_data:
        raise ValueError("all_vs_all_algorithm_stats: no valid data loaded.")
    combined_data_dict = combine_data(valid_data)

    # 2. Roster: explicit, or every algorithm present in ALL loaded problems.
    if algos_to_compare is None:
        rosters = [set(pd.get('results', {})) for pd in combined_data_dict.values()]
        algos_to_compare = sorted(set.intersection(*rosters)) if rosters else []
    if len(algos_to_compare) < 2:
        raise ValueError(
            f"all_vs_all_algorithm_stats: need at least 2 algorithms present in "
            f"every problem, got {algos_to_compare}."
        )

    # 3. Blocked matrix of per-problem mean final fitness.
    rows = []
    for instance_key, problem_data in combined_data_dict.items():
        means = final_fitness_means(problem_data.get('results', {}), algos_to_compare)
        if len(means) == len(algos_to_compare):
            rows.append([means[a] for a in algos_to_compare])

    n_cases = len(rows)
    if n_cases < 3:
        raise ValueError(
            f"all_vs_all_algorithm_stats: only {n_cases} complete problems for "
            f"roster {algos_to_compare}; need at least 3 blocks for Friedman."
        )

    matrix = pd.DataFrame(rows, columns=algos_to_compare)

    print(f"\n=== Algorithm-to-Algorithm Analysis (N={n_cases} paired benchmarks) ===")

    # 4. Mean ranks (1 = best) as the effect size.
    ranks = matrix.rank(axis=1, method='average')
    mean_ranks = ranks.mean(axis=0).sort_values()
    print("Mean ranks (1 = best):")
    for algo, mr in mean_ranks.items():
        print(f"  {algo:40s} {mr:.3f}")

    # 5. Friedman omnibus (ranks within blocks internally).
    stat, p_friedman = friedmanchisquare(*[matrix[a] for a in algos_to_compare])
    print(f"Friedman Test Statistic: {stat:.4f}, p-value: {p_friedman:.5e}")

    nemenyi = None
    if p_friedman < 0.05:
        print("\nFriedman test is SIGNIFICANT. Nemenyi-Friedman post-hoc (blocked design):")
        nemenyi = sp.posthoc_nemenyi_friedman(matrix.values)
        nemenyi.columns = algos_to_compare
        nemenyi.index = algos_to_compare

        if baseline in algos_to_compare:
            print(f"\n--- Significant differences vs {baseline} (p < 0.05) ---")
            base_col = nemenyi[baseline]
            for algo in algos_to_compare:
                if algo != baseline and base_col[algo] < 0.05:
                    direction = "BETTER than" if mean_ranks[algo] < mean_ranks[baseline] else "WORSE than"
                    print(f"{algo}: p={base_col[algo]:.4f} ({direction} {baseline})")
    else:
        print("Friedman test not significant across individual algorithms.")

    return {
        "algorithms": list(algos_to_compare),
        "n_cases": n_cases,
        "mean_ranks": mean_ranks.to_dict(),
        "friedman_statistic": float(stat),
        "friedman_p_value": float(p_friedman),
        "nemenyi_p_values": nemenyi,
    }


from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
import numpy as np


import numpy as np
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests


def many_to_one_vs_baseline(pickle_files, algos_to_compare, baseline="PSO", alpha=0.05):
    """
    Performs many-to-one comparison with a control (baseline) using paired Wilcoxon signed-rank tests
    and Holm–Bonferroni correction.

    Properly aligns paired samples per instance and reports N per algorithm.
    """

    # --------------------------
    # 1. Load and combine data
    # --------------------------
    valid_data = []
    for fp in pickle_files:
        d = load_data(fp)
        if d is not None:
            valid_data.append(d)

    if not valid_data:
        print("No valid data loaded.")
        return

    combined_data_dict = combine_data(valid_data)

    # --------------------------
    # 2. Compute within-instance ranks (1 = best) - scale-free paired scores
    # --------------------------
    # instance_scores[instance_key][algo] = rank among baseline + compared algos
    instance_scores = {}
    instance_means = {}

    excluded_missing_baseline = 0
    excluded_insufficient_algos = 0

    for instance_key, problem_data in combined_data_dict.items():

        results = problem_data.get("results", {})

        # Must have baseline
        if baseline not in results:
            excluded_missing_baseline += 1
            continue

        algo_means = final_fitness_means(results, algos_to_compare + [baseline])

        # Need baseline and at least one comparison algorithm
        if baseline not in algo_means or len(algo_means) < 2:
            excluded_insufficient_algos += 1
            continue

        instance_scores[instance_key] = rank_within_problem(algo_means)
        instance_means[instance_key] = algo_means

    total_instances = len(combined_data_dict)
    valid_instances = len(instance_scores)

    print("\n=== Instance filtering summary ===")
    print(f"Total instances available: {total_instances}")
    print(f"Valid instances used:      {valid_instances}")
    print(f"Excluded missing baseline: {excluded_missing_baseline}")
    print(f"Excluded insufficient algos: {excluded_insufficient_algos}")

    # --------------------------
    # 3. Perform paired tests per algorithm
    # --------------------------
    raw_results = []
    p_values = []

    for algo in algos_to_compare:

        paired_keys = [
            key for key in instance_scores
            if baseline in instance_scores[key] and algo in instance_scores[key]
        ]

        N = len(paired_keys)

        if N < 5:
            continue

        x = np.array([instance_scores[k][algo] for k in paired_keys])
        y = np.array([instance_scores[k][baseline] for k in paired_keys])

        try:
            stat, p = wilcoxon(x, y)
        except Exception:
            continue

        mean_diff = np.mean(x - y)

        direction = "BETTER" if mean_diff < 0 else "WORSE"

        # Win/tie/loss counts vs the baseline (raw per-problem means).
        wins = sum(1 for k in paired_keys
                   if instance_means[k][algo] < instance_means[k][baseline])
        losses = sum(1 for k in paired_keys
                     if instance_means[k][algo] > instance_means[k][baseline])
        ties = N - wins - losses

        raw_results.append({
            "algo": algo,
            "p": p,
            "N": N,
            "mean_rank_diff": mean_diff,
            "wins": wins,
            "ties": ties,
            "losses": losses,
            "direction": direction
        })

        p_values.append(p)

    if not raw_results:
        print("No valid comparisons possible.")
        return

    # --------------------------
    # 4. Holm–Bonferroni correction
    # --------------------------
    reject, pvals_corrected, _, _ = multipletests(
        p_values,
        alpha=alpha,
        method="holm"
    )

    # --------------------------
    # 5. Print results
    # --------------------------
    print(f"\n=== Many-to-One Comparison with Control ({baseline}) ===")

    for i, result in enumerate(raw_results):

        sig = "*" if reject[i] else ""

        print(
            f"{result['algo']:15s} | "
            f"N={result['N']:3d} | "
            f"W/T/L={result['wins']}/{result['ties']}/{result['losses']} | "
            f"Adjusted p={pvals_corrected[i]:.12f} {sig} | "
            f"{result['direction']}"
        )

    # --------------------------
    # 6. Return structured results
    # --------------------------
    return {
        "results": raw_results,
        "corrected_p": pvals_corrected,
        "reject": reject,
        "instance_scores": instance_scores
    }