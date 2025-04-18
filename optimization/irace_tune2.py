import os
import json
import numpy as np
from datetime import datetime

from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.AdaptivePSO import GlobalAdaptivePSO, PersonalAdaptivePSO
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO, \
    ReverseLearningPSO
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

from problem.n_variables.ackley import Ackley


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
    Sphere(number_of_variables),
    Rastrigin(number_of_variables),
    Ackley(number_of_variables)
]

parameter_spaces = {
    # # first batch
    # 'RebelPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac2", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("rebel_fraction", 0.05, 0.8),
    # ],
    # 'RejectorPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac1", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("rejector_fraction", 0.05, 0.8),
    # ],
    # 'RebelRejectorPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac1", 0.01, 6),
    #     Real("ac2", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("rebel_fraction", 0.05, 0.8),
    #     Real("rejector_fraction", 0.05, 0.8),
    # ],
    # 'ContrarianPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac2", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("contrarian_fraction", 0.05, 0.8),
    # ],
    # 'DefeatistPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac1", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("defeatist_fraction", 0.05, 0.8),
    # ],
    # 'ContrarianDefeatistPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac1", 0.01, 6),
    #     Real("ac2", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("contrarian_fraction", 0.05, 0.8),
    #     Real("defeatist_fraction", 0.05, 0.8),
    # ],
    # 'EschewerPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac2", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("eschewer_fraction", 0.05, 0.8),
    # ],

    # second batch
    'EscapistPSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("ac1", 0.01, 6),
        Real("w", 0.01, 2),
        Real("escapist_fraction", 0.05, 0.8),
    ],
    'EschewerEscapistPSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("ac1", 0.01, 6),
        Real("ac2", 0.01, 6),
        Real("w", 0.01, 2),
        Real("eschewer_fraction", 0.05, 0.8),
        Real("escapist_fraction", 0.05, 0.8),
    ],
    'SingleObjectivePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("w", 0.01, 2),
    ],
    'GlobalAdaptivePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("max_c1", 4, 20),
        Real("max_c2", 4, 20),
        Real("w", 0.01, 2),
    ],
    'PersonalAdaptivePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("max_c1", 4, 20),
        Real("max_c2", 4, 20),
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
    'CombinedLearningPSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("b1", 0.01, 6),
        Real("b2", 0.01, 6),
        Real("w", 0.01, 2),
    ],
    'ReverseLearningPSO': [
        Real("b1", 0.01, 6),
        Real("b2", 0.01, 6),
        Real("w", 0.01, 2),
    ],
    'FAPSO': [
        Real("c1", 0.01, 10),
        Real("c2", 0.01, 10),
        Real("w", 0.01, 10),
        Integer("fractal_depth", 1, 5),
        Real("convergence_threshold", 0.0001, 0.2),
    ],

#     # Separate run
#     'RRAPSO': [
#     Real("c1", 0.01, 6),  # Cognitive coefficient
#     Real("c2", 0.01, 6),  # Social coefficient
#     Real("ac1", 0.01, 6),  # Adaptive cognitive coefficient
#     Real("ac2", 0.01, 6),  # Adaptive social coefficient
#     Real("base_inertia", 0.01, 1),  # Base inertia weight
#     Real("min_inertia", 0.01, 1),  # Minimum inertia weight
#     Real("max_inertia", 0.01, 1),  # Maximum inertia weight
#     Real("rebel_fraction", 0.05, 0.8),  # Fraction of rebel particles
#     Real("rejector_fraction", 0.05, 0.8),  # Fraction of rejector particles
#     Integer("window_size", 10, 50),  # Window size for convergence
#     # Real("perturbation_probability", 0.01, 1),  # Probability of perturbation
#     # Real("perturbation_scale", 0.01, 1),  # Scale of perturbation
#     Real("max_rebel_fraction", 0.1, 0.98),  # Max limit for rebel fraction
#     Real("max_rejector_fraction", 0.1, 0.98),  # Max limit for a rejector fraction
#     Real("diversity_threshold", 0.001, 0.3),  # Threshold for diversity
#     Real("improvement_threshold", 0.0001, 0.1),  # Threshold for improvement rate
# ],

}

current_algorithm = None

def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    print(f"Running experiment with configuration: {experiment.configuration}")
    config = experiment.configuration

    # # Constraints check
    # if not (config["min_inertia"] < config["base_inertia"] < config["max_inertia"]):
    #     print("Inertia constraints violated; applying penalty.")
    #     return 3973
    #
    # if config["max_rebel_fraction"] < config["rebel_fraction"]:
    #     print("Rebel fraction constraint violated; applying penalty.")
    #     return 3973
    #
    # if config["max_rejector_fraction"] < config["rejector_fraction"]:
    #     print("Rejector fraction constraint violated; applying penalty.")
    #     return 3973

    # if config["max_c1"]:
    #     if config["max_c1"] < config["c1"]:
    #         print("Rebel fraction constraint violated; applying penalty.")
    #         return 3973
    #
    #     if config["max_c2"] < config["c2"]:
    #         print("Rejector fraction constraint violated; applying penalty.")
    #         return 3973

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
        scenario = Scenario(max_experiments=budget * len(space_list), seed=42, n_jobs=16)

        result = irace(target_runner, parameter_space, scenario, return_df=True, remove_metadata=True)
        best_configurations[algo_name] = result

        # Save results **after each algorithm**
        with open(output_file, "w") as f:
            json.dump({k: v.to_json() for k, v in best_configurations.items()}, f, indent=4)

        print(f"Saved best configuration for {algo_name} to {output_file}")
