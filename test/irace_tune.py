import os
import json
import numpy as np
from datetime import datetime

# Import benchmark problems from jMetalPy
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

# Import your PSO algorithm classes
from problem.n_variables.ackley import Ackley
from problem.n_variables.griewank import Griewank
from algorithm.single_objective_PSO import SingleObjectivePSO, RebelPSO, EscapistPSO, RebelEscapistPSO, REAPSO
from algorithm.FAPSO import FAPSO
from algorithm.NPSO import NPSO
from algorithm.QTPSO import QTPSO
from algorithm.SPPPSO import SPPPSO
from algorithm.TDPSO import TDPSO

# Import iracepy-tiny components
from irace import irace, ParameterSpace, Scenario, Experiment, Real, Integer

# Fix encoding issues by enforcing UTF-8 globally
import rpy2.robjects as robjects
os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["R_DEFAULT_ENCODING"] = "UTF-8"
robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')

# Define constants
number_of_variables = 100
solutions_size = 100
max_evaluations = 25000
num_runs = 5   # Number of independent runs per problem
budget = 1000    # Total number of configurations to try in irace

# Define benchmark problems for tuning
problems = [
    Rastrigin(number_of_variables),
    Ackley(number_of_variables),
    Griewank(number_of_variables)
]

# Define parameter spaces for each algorithm using irace's Real and Integer objects.
# (Each parameter space is defined as a list of parameters; later we pass it to ParameterSpace.)
parameter_spaces = {
    'SingleObjectivePSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
    ],
    'REAPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("base_inertia", 0.01, 10),
        Real("min_inertia", 0.01, 10),
        Real("max_inertia", 0.01, 10),
        Real("rebel_ratio", 0.05, 0.8),
        Real("escapist_ratio", 0.05, 0.8),
    ],
    'RebelPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Real("rebel_fraction", 0.05, 0.8),
    ],
    'EscapistPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Real("escapist_fraction", 0.05, 0.8),
    ],
    'RebelEscapistPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Real("rebel_fraction", 0.05, 0.8),
        Real("escapist_fraction", 0.05, 0.8),
    ],
    'QTPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Real("quantum_prob", 0.01, 1.0),
        Real("chaos_strength", 0.01, 1.0),
    ],
    'SPPPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Real("predator_ratio", 0.01, 0.5),
        Real("scavenger_ratio", 0.01, 0.5),
    ],
    'TDPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Real("temperature", 0.1, 5.0),
        Real("cooling_rate", 0.9, 1.0),
    ],
    'NPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Real("spike_threshold", 0.5, 1.0),
    ],
    'FAPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Integer("fractal_depth", 1, 5),
    ],
}

# Global variable to indicate the current algorithm (used in the target_runner)
current_algorithm = None

def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    """
    The target runner function that irace calls. It receives an Experiment object
    (with a candidate configuration in experiment.configuration) and a Scenario.
    It then evaluates the candidate configuration over all benchmark problems and runs.
    """
    config = experiment.configuration
    results = []
    AlgorithmClass = globals()[current_algorithm]
    for problem in problems:
        for _ in range(num_runs):
            algorithm = AlgorithmClass(
                problem=problem,
                swarm_size=solutions_size,
                termination_criterion=StoppingByEvaluations(max_evaluations),
                **config
            )
            algorithm.run()
            results.append(algorithm.result().objectives[0])
    avg_result = np.mean(results)
    print(f"Evaluated config: {config} with average objective: {avg_result}")
    return avg_result

if __name__ == "__main__":
    best_configurations = {}
    # Loop over each algorithm in our parameter_spaces
    for algo_name, space_list in parameter_spaces.items():
        current_algorithm = algo_name  # Set the current algorithm for the target runner
        print(f"Optimizing parameters for {algo_name} ...")
        # Create a ParameterSpace instance from the list of parameters
        parameter_space = ParameterSpace(params=space_list)
        # Create a Scenario instance with the desired settings
        scenario = Scenario(max_experiments=budget, seed=42, n_jobs=4)
        # Run irace using the target_runner, parameter_space, and scenario.
        result = irace(target_runner, parameter_space, scenario, return_df=True, remove_metadata=True)
        best_configurations[algo_name] = result
        print(f"Best configuration for {algo_name}: {result}")

    # Save the best configurations to a JSON file for later reference.
    with open("irace_best_configurations.json", "w") as f:
        json.dump(best_configurations, f, indent=4)
