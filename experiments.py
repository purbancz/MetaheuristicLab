import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.single_objective_PSO import SingleObjectivePSO
from observer.fitness_observer import FitnessObserver
from problem.fixed_varaibles.de_joung import DeJoung
from problem.n_variables.ackley import Ackley


# Function to run the experiment
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
    average_time = np.mean(total_times)

    return np.array(all_fitness_data), average_final_fitness, average_time


# Function to plot results
# Function to plot results
def plot_results(data_dict, problem):
    plt.figure(figsize=(12, 6))
    for label, fitness_data in data_dict.items():
        average_fitness = np.mean(fitness_data['data'], axis=0)
        plt.plot(average_fitness, label=label)

    plt.title(f'{problem.name()} ({problem.number_of_variables()} variables)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Average Best Fitness over {no_of_runs} runs')

    # Add legend with proper background
    plt.legend(frameon=True, facecolor='white', framealpha=1)

    # Add grid
    plt.grid()

    # Save the plot to a file
    plt.savefig(f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem.name()}.png')

    # Adjust layout to make sure everything fits correctly
    plt.tight_layout()

    # Show the plot
    plt.show()


no_of_runs = 3
number_of_variables = 10
solutions_size = 10
max_evaluations = 2500
frequency = solutions_size  # Snapshot each generation

results_dir = 'experiment_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

csv_filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_results.csv'

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Algorithm', 'Problem', 'Variables', 'Runs', 'Average Final Fitness',
                     'Average Computing Time (s)'])

    problems = [Rastrigin(number_of_variables),
                DeJoung()]

    for problem in problems:
        algorithms = {
            'GA': GeneticAlgorithm(
                problem=problem,
                population_size=solutions_size,
                offspring_population_size=solutions_size,
                mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
                crossover=SBXCrossover(0.9, 5.0),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            ),
            'PSO': SingleObjectivePSO(
                problem=problem,
                swarm_size=solutions_size,
                c1=1.97,
                c2=0.94,
                w=0.56,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            )
        }

        results = {}
        for name, algorithm in algorithms.items():
            fitness_data, average_final_fitness, average_time = run_experiment(algorithm, no_of_runs, frequency)
            results[name] = {'data': fitness_data, 'avg_fitness': average_final_fitness, 'avg_time': average_time}

            # Print data for debugging
            print(f"Algorithm: {name}, Problem: {problem.name()}, Variables: {problem.number_of_variables()},"
                  f"Runs: {no_of_runs}, Average Final Fitness: {average_final_fitness}, Average Time: {average_time}")

            writer.writerow([name, problem.name(), problem.number_of_variables(), no_of_runs, average_final_fitness,
                             average_time])

        plot_results(results, problem)
