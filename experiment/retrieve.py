import pickle
import numpy as np
import scikit_posthocs as sp
import pandas as pd
from scipy.stats import kruskal, f_oneway, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn

from experiment.plotting_utilities import plot_results, plot_results_with_std, plot_box_at_intervals, plot_final_box
from experiment.setup import setup_experiment, make_dir

# Setup experiment to retrieve settings like algorithm_colors, max_evaluations, etc.
(algorithms, group_of_algorithms, problems, _, number_of_variables, solutions_size,
 max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()


def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def plot_all_from_pickle(file_path):
    loaded_data = load_data_from_pickle(file_path)

    for problem_data in loaded_data:
        problem_name = problem_data['problem']
        n_vars = problem_data['n_vars']
        results = problem_data['results']

        # Matching problem instance
        matched_problem = next((prob for prob in problems if prob.name() == problem_name), None)

        if matched_problem is None:
            print(f"Problem {problem_name} not found in the setup experiment list.")
            continue

        no_of_runs = problem_data['results']['GA']['data'].shape[0]

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

        # Plotting box plots comparing PGxHEA algorithms with GA and PSO
        for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=no_of_runs, algorithms_to_compare=[algorithm, 'GA', 'PSO'],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting final box plot comparing all algorithms
        plot_final_box(results, matched_problem, dimensions_dir, algorithm_colors)


def combine_data(data_list):
    combined_data = {}
    total_runs = 0

    for data in data_list:
        # Accumulate the number of runs from each data set
        total_runs += data[0]['results']['GA']['data'].shape[0]  # You can change 'GA' to any consistently run algorithm

        for problem_data in data:
            problem_name = problem_data['problem']
            n_vars = problem_data['n_vars']
            results = problem_data['results']

            if problem_name not in combined_data:
                combined_data[problem_name] = {
                    'n_vars': n_vars,
                    'results': {algo: {'data': [], 'avg_fitness': [], 'std_dev': [], 'avg_time': []}
                                for algo in results.keys()}
                }

            for algo, algo_data in results.items():
                combined_data[problem_name]['results'][algo]['data'].append(algo_data['data'])
                combined_data[problem_name]['results'][algo]['avg_fitness'].append(algo_data['avg_fitness'])
                combined_data[problem_name]['results'][algo]['std_dev'].append(algo_data['std_dev'])
                combined_data[problem_name]['results'][algo]['avg_time'].append(algo_data['avg_time'])

    # Aggregating the data
    for problem_name, problem_data in combined_data.items():
        for algo, algo_data in problem_data['results'].items():
            # Concatenate the list of arrays into a single array
            algo_data['data'] = np.concatenate(algo_data['data'], axis=0)
            algo_data['avg_fitness'] = np.mean(algo_data['avg_fitness'])
            algo_data['std_dev'] = np.std(algo_data['avg_fitness'])
            algo_data['avg_time'] = np.mean(algo_data['avg_time'])

    return combined_data, total_runs


def plot_combined_data_from_pickles(pickle_files):
    data_list = [load_data_from_pickle(file) for file in pickle_files]
    combined_data, total_runs = combine_data(data_list)

    for problem_name, problem_data in combined_data.items():
        n_vars = problem_data['n_vars']
        results = problem_data['results']

        # Matching problem instance
        matched_problem = next((prob for prob in problems if prob.name() == problem_name), None)

        if matched_problem is None:
            print(f"Problem {problem_name} not found in the setup experiment list.")
            continue

        # Directory to save plots for the specific problem
        dimensions_dir = f"{results_dir}/dim{n_vars}_runs{total_runs}"
        make_dir(dimensions_dir)

        # Plotting all required graphs
        plot_results(results, matched_problem, dimensions_dir, max_evaluations, total_runs, algorithm_colors)
        plot_results_with_std(results, matched_problem, dimensions_dir, max_evaluations, total_runs, algorithm_colors)
        plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                              no_of_runs=total_runs, algorithms_to_compare=algorithms.keys(),
                              results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting box plots for each individual algorithm
        for algorithm in algorithms.keys():
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=total_runs, algorithms_to_compare=[algorithm],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting box plots comparing PGxHEA algorithms with GA and PSO
        for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=total_runs, algorithms_to_compare=[algorithm, 'GA', 'PSO'],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting final box plot comparing all algorithms
        plot_final_box(results, matched_problem, dimensions_dir, algorithm_colors)


def kruskal_wallis_with_posthoc(pickle_files, perform_shapiro=False, perform_posthoc=True):
    best_algorithms = {
        "Ackley": {10: "PSO", 50: "PGPHEA", 100: "PGPHEA", 500: "PGPHEA", 1000: "PGPHEA"},
        "Griewank": {10: "PSO", 50: "PSO", 100: "PGPHEA", 500: "PGPHEA", 1000: "PGPHEA"},
        "Levy": {10: ["PSO", "PGPHEA"], 50: "PGSHEA", 100: "PGPHEA", 500: "PGPHEA", 1000: "PGPHEA"},
        "Michalewicz": {10: "GA", 50: "PGPHEA", 100: "PGPHEA", 500: "PGPHEA", 1000: "PGPHEA"},
        "Rastrigin": {10: "GA", 50: "PGPHEA", 100: "PGPHEA", 500: "PGPHEA", 1000: "PGPHEA"},
        "Schwefel": {10: "PGCHEA", 50: "GA", 100: "GA", 500: "PGPHEA", 1000: "PGPHEA"},
        "Shifted Rotated Weierstrass": {10: "GA", 50: "PGCHEA", 100: "PGSHEA", 500: "PGCHEA", 1000: "PGCHEA"},
    }

    non_significant_dunn = []
    non_significant_tukey = []
    non_significant_vs_best_dunn = []
    non_significant_vs_best_tukey = []

    for pickle_file in pickle_files:
        with open(pickle_file, 'rb') as f:
            loaded_data = pickle.load(f)

        for problem_data in loaded_data:
            problem_name = problem_data['problem']
            dimension = problem_data['n_vars']
            best_algos = best_algorithms.get(problem_name, {}).get(dimension, [])

            if not best_algos:
                continue

            if isinstance(best_algos, str):
                best_algos = [best_algos]

            print(f"\nPerforming analysis for Problem: {problem_name}, Dimension: {dimension}")

            final_fitness_values = {
                algo: [run[-1] for run in algo_data['data']]
                for algo, algo_data in problem_data['results'].items()
            }

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
                    dunn_results = sp.posthoc_dunn(data, p_adjust='fdr_bh')
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





