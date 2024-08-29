import csv
import pickle
from datetime import datetime

import numpy as np

from experiment.plotting_utilities import plot_results, plot_results_with_std, plot_box_at_intervals, plot_final_box
from experiment.setup import setup_experiment, initialize_algorithms
from observer.fitness_observer import FitnessObserver

# Configuration
(algorithms, problems, no_of_runs, number_of_variables, solutions_size,
 max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()


def run_all_experiments():
    csv_filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_results.csv'
    with (open(csv_filename, mode='w', newline='') as file):
        writer = csv.writer(file)
        writer.writerow(['Algorithm', 'Problem', 'Variables', 'Runs', 'Average Final Fitness',
                         'Standard deviation', 'Average Computing Time (s)'])

        all_data = []
        for problem in problems:
            initialized_algorithms = initialize_algorithms(algorithms, problem)
            problem_data = {'problem': problem.name(), 'n_vars': problem.number_of_variables(), 'results': {}}
            for name, algorithm in initialized_algorithms.items():
                fitness_data, avg_fitness, std_dev, avg_time = run_experiment(algorithm, no_of_runs, frequency)
                problem_data['results'][name] = {'data': fitness_data, 'avg_fitness': avg_fitness, 'std_dev': std_dev,
                                                 'avg_time': avg_time}

                print(f"Algorithm: {name}, Problem: {problem.name()}, Variables: {problem.number_of_variables()}, "
                      f"Runs: {no_of_runs}, Average Final Fitness: {avg_fitness},"
                      f"Standard deviation: {std_dev}, Average Time: {avg_time}")

                writer.writerow([name, problem.name(), problem.number_of_variables(), no_of_runs, avg_fitness,
                                 std_dev, avg_time])

            all_data.append(problem_data)

            # plot results
            plot_results(problem_data['results'], problem, results_dir, max_evaluations, no_of_runs,
                         algorithm_colors)
            plot_results_with_std(problem_data['results'], problem, results_dir, max_evaluations,
                                  no_of_runs, algorithm_colors)
            plot_box_at_intervals(problem_data['results'], problem, max_evaluations=max_evaluations,
                                  no_of_runs=no_of_runs, algorithms_to_compare=algorithms.keys(),
                                  results_dir=results_dir,
                                  algorithm_colors=algorithm_colors)

            for algorithm in algorithms.keys():
                plot_box_at_intervals(problem_data['results'], problem, max_evaluations=max_evaluations,
                                      no_of_runs=no_of_runs, algorithms_to_compare=[algorithm],
                                      results_dir=results_dir,
                                      algorithm_colors=algorithm_colors)
            for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
                plot_box_at_intervals(problem_data['results'], problem, max_evaluations=max_evaluations,
                                      no_of_runs=no_of_runs, algorithms_to_compare=[algorithm] + ['GA', 'PSO'],
                                      results_dir=results_dir,
                                      algorithm_colors=algorithm_colors)
            plot_final_box(problem_data['results'], problem, results_dir, algorithm_colors)

    with open(f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_experiment_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)


def run_experiment(algorithm, runs, interval):
    all_fitness_data = []
    total_times = []

    for _ in range(runs):
        observer = FitnessObserver(interval=interval)
        algorithm.observable.register(observer)

        start_time = datetime.now()
        algorithm.run()
        end_time = datetime.now()
        total_time = (end_time - start_time).total_seconds()

        last_fitness = observer.best_fitness_history[-1] if observer.best_fitness_history else float('nan')
        filled_fitness = (observer.best_fitness_history +
                          [last_fitness] * (max_evaluations // interval - len(observer.best_fitness_history)))
        all_fitness_data.append(filled_fitness)
        total_times.append(total_time)

    average_final_fitness = np.mean([data[-1] for data in all_fitness_data])
    standard_deviation = np.std([data[-1] for data in all_fitness_data])
    average_time = np.mean(total_times)

    return np.array(all_fitness_data), average_final_fitness, standard_deviation, average_time
