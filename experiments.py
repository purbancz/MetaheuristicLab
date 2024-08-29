import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import csv
import pickle
from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.PGCHEA import PGCHEA
from algorithm.PGPHEA import PGPHEA
from algorithm.PGSHEA import PGSHEA
from algorithm.single_objective_PSO import SingleObjectivePSO
from observer.fitness_observer import FitnessObserver
from problem.fixed_varaibles.branin import BraninRCOC
from problem.fixed_varaibles.de_joung import DeJoung
from problem.fixed_varaibles.easom import Easom
from problem.fixed_varaibles.goldstein_price import GoldsteinPrice
from problem.fixed_varaibles.hartmann import Hartmann
from problem.fixed_varaibles.schaffer import SchafferN2
from problem.fixed_varaibles.shekel import Shekel
from problem.fixed_varaibles.shubert import Shubert
from problem.n_variables.ackley import Ackley
from problem.n_variables.griewank import Griewank
from problem.n_variables.levy import Levy
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.rosenbrock import Rosenbrock
from problem.n_variables.schwefel import Schwefel
from problem.n_variables.weierstrass import ShiftedRotatedWeierstrass
from problem.n_variables.zakharov import Zakharov


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
    standard_deviation = np.std([data[-1] for data in all_fitness_data])
    average_time = np.mean(total_times)

    return np.array(all_fitness_data), average_final_fitness, standard_deviation, average_time


# Function to plot results
def plot_results(data_dict, problem):
    plt.figure(figsize=(12, 6))
    for label, fitness_data in data_dict.items():
        average_fitness = np.mean(fitness_data['data'], axis=0)
        color = ALGORITHM_COLORS.get(label, 'black')  # Use the global color dictionary
        plt.plot(average_fitness, label=label, color=color)

    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Average Best Fitness over {no_of_runs} runs')
    plt.legend(frameon=True, facecolor='white', framealpha=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem.name()}.png')
    plt.show()

def plot_results_with_std(data_dict, problem):
    plt.figure(figsize=(12, 6))
    for label, fitness_data in data_dict.items():
        average_fitness = np.mean(fitness_data['data'], axis=0)
        std_dev_fitness = np.std(fitness_data['data'], axis=0)
        color = ALGORITHM_COLORS.get(label, 'black')  # Use the global color dictionary
        plt.plot(average_fitness, label=label, color=color)
        plt.fill_between(range(len(average_fitness)),
                         average_fitness - std_dev_fitness,
                         average_fitness + std_dev_fitness,
                         color=color, alpha=0.2)

    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Average Best Fitness over {no_of_runs} runs')
    plt.legend(frameon=True, facecolor='white', framealpha=1)
    plt.grid()
    plt.tight_layout()
    plt.savefig(f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem.name()}_with_stddev.png')
    plt.show()

def plot_box_at_intervals(data_dict, problem, interval=10, algorithms_to_compare=None):
    if algorithms_to_compare is None:
        algorithms_to_compare = data_dict.keys()

    plt.figure(figsize=(12, 6))

    # Prepare legend entries
    legend_handles = []

    # Determine the maximum number of evaluations across all runs
    max_evaluations_index = max([len(fitness_data['data'][0]) for fitness_data in data_dict.values()])

    for label, fitness_data in data_dict.items():
        if label not in algorithms_to_compare:
            continue

        box_data = []
        positions = []
        for i in range(0, max_evaluations_index, interval):
            box_data.append([run_data[i] for run_data in fitness_data['data']])
            positions.append(i)

        # Add the final evaluation data to the box plot if not already included
        if max_evaluations_index - 1 not in positions:
            box_data.append([run_data[-1] for run_data in fitness_data['data']])
            positions.append(max_evaluations_index - 1)

        # Use the global color dictionary
        color = ALGORITHM_COLORS.get(label, 'black')  # Default to black if not specified
        bp = plt.boxplot(box_data,
                         positions=positions,
                         widths=5,
                         patch_artist=True,
                         boxprops=dict(facecolor=color, color=color),
                         whiskerprops=dict(color=color),
                         capprops=dict(color=color),
                         medianprops=dict(color='yellow'),
                         flierprops=dict(marker='o', color=color, markersize=5, alpha=0.5))

        # Append to legend handles
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))

    # Add legend manually
    plt.legend(handles=legend_handles, frameon=True, facecolor='white', framealpha=1)

    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Fitness Distribution over {no_of_runs} runs')
    plt.grid()

    # Adjust x-axis range and labels
    plt.xlim([-5, max_evaluations_index + 5])  # Add padding to avoid cropping
    plt.xticks(np.arange(0, max_evaluations_index + 1, interval),
               labels=np.arange(0, max_evaluations_index + 1, interval))

    plt.tight_layout()

    # Join algorithm names with underscores, removing any special characters that could cause issues in filenames
    algorithm_names = '_'.join([algo.replace(' ', '_').replace('-', '_') for algo in algorithms_to_compare])

    # Save the figure with the algorithm names included in the filename
    plt.savefig(
        f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem.name()}_{algorithm_names}_box_intervals.png')
    plt.show()


def plot_final_box(data_dict, problem):
    plt.figure(figsize=(12, 6))
    box_data = [fitness_data['data'][:, -1] for fitness_data in data_dict.values()]
    labels = [label for label in data_dict.keys()]
    colors = [ALGORITHM_COLORS.get(label, 'black') for label in labels]
    bp = plt.boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.ylabel('Final Fitness Distribution')

    # Remove the top, right, and bottom spines (the frame around the boxes)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)

    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    plt.savefig(f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem.name()}_final_box.png')
    plt.show()




no_of_runs = 5
number_of_variables = 50
solutions_size = 100
max_evaluations = 25000
frequency = solutions_size  # Snapshot each generation
all_data = []

# Global color mapping for algorithms
ALGORITHM_COLORS = {
    'GA': 'blue',
    'PSO': 'orange',
    'PGPHEA': 'purple',
    'PGSHEA': 'green',
    'PGCHEA': 'red'
}

results_dir = 'experiment_results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

csv_filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_results.csv'

with (open(csv_filename, mode='w', newline='') as file):
    writer = csv.writer(file)
    writer.writerow(['Algorithm', 'Problem', 'Variables', 'Runs', 'Average Final Fitness', 'Standard deviation'
                     'Average Computing Time (s)'])

    n_variables_problems = [
        # Zakharov(number_of_variables),
        # Rosenbrock(number_of_variables),
        ###
        # Rastrigin(number_of_variables),
        Ackley(number_of_variables),
        # Griewank(number_of_variables),
        # Levy(number_of_variables),
        # Michalewicz(number_of_variables),
        # Schwefel(number_of_variables),
        # ShiftedRotatedWeierstrass(number_of_variables),
    ]

    fixed_variables_problems = [
        # BraninRCOC(),
        # DeJoung(),
        # GoldsteinPrice(),
        # Hartmann(),
        # Shubert()
        ###
        # SchafferN2(),
        # Shekel(),
        # Easom(),
    ]
    problems = n_variables_problems + fixed_variables_problems

    for problem in problems:
        problem_data = {'problem': problem.name(), 'n_vars': problem.number_of_variables(), 'results': {}}
        algorithms = {
            'GA': GeneticAlgorithm(
                problem=problem,
                population_size=solutions_size,
                offspring_population_size=solutions_size,
                mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
                crossover=SBXCrossover(0.75, 5.0),
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
            ),
            'PSO': SingleObjectivePSO(
                problem=problem,
                swarm_size=solutions_size,
                c1=1.97,
                c2=0.94,
                w=0.56,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            'PGSHEA': PGSHEA(
                problem=problem,
                solutions_size=solutions_size,
                mutation=PolynomialMutation(0.38 / problem.number_of_variables(), 20.0),
                crossover=SBXCrossover(1, 5.0),
                swap_interval=13, #int(max_evaluations/(2 * solutions_size))
                c1=2.63,
                c2=0.21,
                w=0.01,
                starting_algorithm='PSO',
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            'PGPHEA': PGPHEA(
                problem=problem,
                solutions_size=solutions_size,
                mutation=PolynomialMutation(0.37 / problem.number_of_variables(), 20.0),
                crossover=SBXCrossover(1, 5.0),
                exchange_interval=13,
                exchange_number=7, #11
                c1=0.00001,
                c2=0.26,
                w=0.17,
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
            'PGCHEA': PGCHEA(
                problem=problem,
                solutions_size=solutions_size,
                mutation=PolynomialMutation(0.61 / problem.number_of_variables(), 20.0),
                crossover=SBXCrossover(1, 5.0),
                c1=1.85,
                c2=0.5,
                w=1.53,
                starting_algorithm='PSO',
                termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
            ),
        }

        results = {}
        for name, algorithm in algorithms.items():
            fitness_data, avg_fitness, std_dev, avg_time = run_experiment(algorithm, no_of_runs, frequency)
            problem_data['results'][name] = {'data': fitness_data, 'avg_fitness': avg_fitness, 'std_dev': std_dev,
                                             'avg_time': avg_time}

            # Print data for debugging
            print(f"Algorithm: {name}, Problem: {problem.name()}, Variables: {problem.number_of_variables()}, "
                  f"Runs: {no_of_runs}, Average Final Fitness: {avg_fitness},"
                  f"Standard deviation: {std_dev}, Average Time: {avg_time}"
                  )

            writer.writerow([name, problem.name(), problem.number_of_variables(), no_of_runs, avg_fitness,
                             std_dev, avg_time])

        all_data.append(problem_data)

        plot_results(problem_data['results'], problem)
        plot_results_with_std(problem_data['results'], problem)
        plot_box_at_intervals(problem_data['results'], problem)
        plot_final_box(problem_data['results'], problem)

with open(f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_experiment_data.pkl', 'wb') as f:
    pickle.dump(all_data, f)

