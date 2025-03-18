import os
import json
import numpy as np
from datetime import datetime
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from problem.n_variables.ackley import Ackley
from problem.n_variables.griewank import Griewank
from algorithm.AdaptivePSO import GlobalAdaptivePSO, LocalAdaptivePSO
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO
from algorithm.single_objective_PSO import SingleObjectivePSO
from algorithm.FAPSO import FAPSO
from algorithm.NPSO import NPSO
from algorithm.QTPSO import QTPSO
from algorithm.SPPPSO import SPPPSO
from algorithm.TDPSO import TDPSO
from irace import irace, ParameterSpace, Scenario, Experiment, Real, Integer
import rpy2.robjects as robjects

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["R_DEFAULT_ENCODING"] = "UTF-8"
robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
# robjects.r('library(iraceplot)')



# number_of_variables = 10
# solutions_size = 10
# max_evaluations = 1000
# num_runs = 2
# budget = 60


number_of_variables = 100
solutions_size = 100
max_evaluations = 25000
num_runs = 5   # Number of independent runs per problem
budget = 1000    # Total number of configurations to try per parameter

problems = [
    Rastrigin(number_of_variables),
    Ackley(number_of_variables),
    Griewank(number_of_variables)
]

parameter_spaces = {
    'SingleObjectivePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("w", 0.01, 2),
    ],
    'ReverseLearningGlobalAttractorPSO': [
        Real("a", 0.01, 6),
        Real("b1", 0.01, 6),
        Real("b2", 0.01, 6),
        Real("w", 0.01, 2),
    ],
    'ReverseLearningPersonalAttractorPSO': [
        Real("a", 0.01, 6),
        Real("b1", 0.01, 6),
        Real("b2", 0.01, 6),
        Real("w", 0.01, 2),
    ],
    'EschewerPSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("ac2", 0.01, 6),
        Real("w", 0.01, 2),
        Real("eschewer_fraction", 0.05, 0.8),
    ],
    'EscapistPSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("ac1", 0.01, 6),
        Real("w", 0.01, 2),
        Real("escapist_fraction", 0.05, 0.8),
    ],
    'EschewerEscapistPSO': [ # 7
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("ac1", 0.01, 6),
        Real("ac2", 0.01, 6),
        Real("w", 0.01, 2),
        Real("eschewer_fraction", 0.05, 0.8),
        Real("escapist_fraction", 0.05, 0.8),
    ],
    'GlobalAdaptivePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("max_c1", 4, 20),
        Real("max_c2", 4, 20),
        Real("w", 0.01, 2),

    ],
    'LocalAdaptivePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("max_c1", 4, 20),
        Real("max_c2", 4, 20),
        Real("w", 0.01, 2),
    ],
}

current_algorithm = None

def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    print(f"Running experiment with configuration: {experiment.configuration}")
    config = experiment.configuration
    results = []
    AlgorithmClass = globals()[current_algorithm]
    for problem in problems:
        for _ in range(num_runs):
            # print(f"Running {AlgorithmClass.__name__} on {problem.__class__.__name__}")
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
    output_file = "irace_best_configurations.json"

    for algo_name, space_list in parameter_spaces.items():
        current_algorithm = algo_name
        print(f"Optimizing parameters for {algo_name} ...")

        parameter_space = ParameterSpace(params=space_list)
        scenario = Scenario(max_experiments=budget * len(space_list), seed=42, n_jobs=8)

        result = irace(target_runner, parameter_space, scenario, return_df=True, remove_metadata=True)
        best_configurations[algo_name] = result

        # Save results **after each algorithm**
        with open(output_file, "w") as f:
            json.dump({k: v.to_json() for k, v in best_configurations.items()}, f, indent=4)

        print(f"Saved best configuration for {algo_name} to {output_file}")
