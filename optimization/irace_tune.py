import os
import json
import numpy as np
from datetime import datetime

from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.hybrid_diverse import HybridPartialDisjointPSO, HybridFullDisjointPSO, HybridAdditivePSO
from algorithm.AdaptivePSO import CoAdaptativePSO, IndividualAdaptivePSO
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO, \
    ReverseLearningPSO
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO, CDAPSO, EEAPSO, AnarchicPSO, AmnesiacPSO, \
    WandererPSO, NoisyPSO
from algorithm.single_objective_PSO import SingleObjectivePSO, PerturbationPSO
from algorithm.reinitialized_PSO import FAPSO
from algorithm.NPSO import NPSO
from algorithm.QTPSO import QTPSO
from algorithm.SPPPSO import SPPPSO
from algorithm.TDPSO import TDPSO
from irace import irace, ParameterSpace, Scenario, Experiment, Real, Integer, Bool, Categorical
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
num_runs = 5  # Number of independent runs per problem
budget = 1000  # Total number of configurations to try per parameter

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

    # # second batch
    # 'EscapistPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac1", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("escapist_fraction", 0.05, 0.8),
    # ],
    # 'EschewerEscapistPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("ac1", 0.01, 6),
    #     Real("ac2", 0.01, 6),
    #     Real("w", 0.01, 2),
    #     Real("eschewer_fraction", 0.05, 0.8),
    #     Real("escapist_fraction", 0.05, 0.8),
    # ],
    # 'SingleObjectivePSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("w", 0.01, 2),
    # ],
    # 'GlobalAdaptivePSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("max_c1", 4, 20),
    #     Real("max_c2", 4, 20),
    #     Real("w", 0.01, 2),
    # ],
    # 'PersonalAdaptivePSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("max_c1", 4, 20),
    #     Real("max_c2", 4, 20),
    #     Real("w", 0.01, 2),
    # ],
    # 'ReverseLearningGlobalAttractorPSO': [
    #     Real("a", 0.01, 6),
    #     Real("b1", 0.01, 6),
    #     Real("b2", 0.01, 6),
    #     Real("w", 0.01, 2),
    # ],
    # 'ReverseLearningPersonalAttractorPSO': [
    #     Real("a", 0.01, 6),
    #     Real("b1", 0.01, 6),
    #     Real("b2", 0.01, 6),
    #     Real("w", 0.01, 2),
    # ],
    # 'CombinedLearningPSO': [
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("b1", 0.01, 6),
    #     Real("b2", 0.01, 6),
    #     Real("w", 0.01, 2),
    # ],
    # 'ReverseLearningPSO': [
    #     Real("b1", 0.01, 6),
    #     Real("b2", 0.01, 6),
    #     Real("w", 0.01, 2),
    # ],
    # 'FAPSO': [
    #     Real("c1", 0.01, 10),
    #     Real("c2", 0.01, 10),
    #     Real("w", 0.01, 10),
    #     Integer("fractal_depth", 1, 5),
    #     Real("convergence_threshold", 0.0001, 0.2),
    # ],

    #     # Separate run
    # 'HybridPartialDisjointPSO': [
    #     # Core PSO Params
    #     Real("w", 0.01, 1.0),  # Inertia weight
    #     Real("c1", 0.01, 6.0),  # Standard cognitive coefficient
    #     Real("c2", 0.01, 6.0),  # Standard social coefficient
    #     # Special Role Coefficients
    #     Real("rejector_c", 0.01, 6.0),
    #     Real("defeatist_c", 0.01, 6.0),
    #     Real("escapist_c", 0.01, 6.0),
    #     Real("rebel_c", 0.01, 6.0),
    #     Real("contrarian_c", 0.01, 6.0),
    #     Real("eschewer_c", 0.01, 6.0),
    #     # Special Role Fractions (Require normalization in target-runner)
    #     # Range [0.0, 1.0] allows exploring full range, normalization ensures validity.
    #     Real("rejector_fraction", 0.01, 0.78),  # Max value should allow sum > 1 before normalization
    #     Real("defeatist_fraction", 0.01, 0.78),
    #     Real("escapist_fraction", 0.01, 0.78),
    #     # --- Cognitive sum constraint handled in runner ---
    #     Real("rebel_fraction", 0.01, 0.78),
    #     Real("contrarian_fraction", 0.01, 0.78),
    #     Real("eschewer_fraction", 0.01, 0.78),
    #     # --- Social sum constraint handled in runner ---
    #     # Behavior Flags
    #     Bool("assign_roles_every_iteration"),
    # ],
    # 'HybridFullDisjointPSO': [  # Assuming the version with individual fraction parameters
    #     # Core PSO Params
    #     Real("w", 0.01, 1.0),  # Inertia weight
    #     Real("c1", 0.01, 6.0),  # Standard cognitive coefficient (used if role is 'std_cognitive')
    #     Real("c2", 0.01, 6.0),  # Standard social coefficient (used if role is 'std_social')
    #     # Special Role Coefficients
    #     Real("rejector_c", 0.01, 6.0),
    #     Real("defeatist_c", 0.01, 6.0),
    #     Real("escapist_c", 0.01, 6.0),
    #     Real("rebel_c", 0.01, 6.0),
    #     Real("contrarian_c", 0.01, 6.0),
    #     Real("eschewer_c", 0.01, 6.0),
    #     # Special Role Fractions (Require normalization in target-runner)
    #     # Range [0.0, 1.0] allows exploring full range, normalization ensures sum <= 1.
    #     Real("rejector_fraction", 0.01, 0.75),
    #     Real("defeatist_fraction", 0.01, 0.75),
    #     Real("escapist_fraction", 0.01, 0.75),
    #     Real("rebel_fraction", 0.01, 0.75),
    #     Real("contrarian_fraction", 0.01, 0.75),
    #     Real("eschewer_fraction", 0.01, 0.75),
    #     # --- Sum constraint (sum <= 1.0) handled in runner ---
    #     # Behavior Flags
    #     Bool("assign_roles_every_iteration"),
    # ],
    # 'HybridAdditivePSO': [ # Assuming the version with default-to-standard logic
    #     # Core PSO Params
    #     Real("w", 0.01, 1.0),               # Inertia weight
    #     Real("c1", 0.01, 6.0),             # Standard cognitive coefficient
    #     Real("c2", 0.01, 6.0),             # Standard social coefficient
    #     # Special Role Coefficients
    #     Real("rejector_c", 0.01, 6.0),
    #     Real("defeatist_c", 0.01, 6.0),
    #     Real("escapist_c", 0.01, 6.0),
    #     Real("rebel_c", 0.01, 6.0),
    #     Real("contrarian_c", 0.01, 6.0),
    #     Real("eschewer_c", 0.01, 6.0),
    #     # Role Activation Probabilities (Independent)
    #     Real("std_cognitive_prob", 0.01, 1.0), # Prob of activating std cognitive explicitly
    #     Real("rejector_prob", 0.01, 1.0),
    #     Real("defeatist_prob", 0.01, 1.0),
    #     Real("escapist_prob", 0.01, 1.0),
    #     Real("std_social_prob", 0.01, 1.0),    # Prob of activating std social explicitly
    #     Real("rebel_prob", 0.01, 1.0),
    #     Real("contrarian_prob", 0.01, 1.0),
    #     Real("eschewer_prob", 0.01, 1.0),
    #     # Behavior Flags
    #     # Bool("assign_flags_every_iteration"),
    # ],
    # 'AnarchicPSO': [
    #     Real("w", 0.01, 1),
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("random_strength", 0.01, 3),
    #     Real("anarchic_fraction", 0.01, 0.8),
    # ],
    # 'AmnesiacPSO': [
    #     Real("w", 0.01, 1),
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("random_strength", 0.01, 3),
    #     Real("amnesiac_fraction", 0.01, 0.8),
    # ],
    # 'WandererPSO': [
    #     Real("w", 0.01, 1),
    #     Real("c1", 0.01, 6),
    #     Real("c2", 0.01, 6),
    #     Real("random_strength", 0.01, 3),
    #     Real("wanderer_fraction", 0.01, 0.8),
    # ],
    'NoisyPSO': [
        Real("w", 0.01, 1),
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("noise_strength", 0.01, 3),
        Real("noisy_fraction", 0.01, 0.8),
    ],
    'PerturbationPSO': [
        Real("w", 0.01, 1),
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("perturbation_scale", 0.001, 1.0),
        Categorical("perturbation_method", ["gaussian", "cauchy"]),
    ]
}

current_algorithm = None


def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    print(f"Running experiment with configuration: {experiment.configuration}")
    config = experiment.configuration

    # # --- Cognitive Normalization ---
    # cog_fractions = {
    #     "rejector": config["rejector_fraction"],
    #     "defeatist": config["defeatist_fraction"],
    #     "escapist": config["escapist_fraction"],
    # }
    # sum_cog = sum(cog_fractions.values())
    # if sum_cog > 1.0:  # Normalize if sum exceeds 1.0
    #     for role in cog_fractions: cog_fractions[role] /= sum_cog
    # # else: use original fractions (sum <= 1.0)
    #
    # # --- Social Normalization ---
    # soc_fractions = {
    #     "rebel": config["rebel_fraction"],
    #     "contrarian": config["contrarian_fraction"],
    #     "eschewer": config["eschewer_fraction"],
    # }
    # sum_soc = sum(soc_fractions.values())
    # if sum_soc > 1.0:  # Normalize if sum exceeds 1.0
    #     for role in soc_fractions: soc_fractions[role] /= sum_soc
    # # else: use original fractions (sum <= 1.0)

    # # Constraints check
    # if config["rejector_fraction"] + config["defeatist_fraction"] + config["escapist_fraction"] + config[
    #     "rebel_fraction"] + config["contrarian_fraction"] + config["eschewer_fraction"] > 0.8:
    #     print("Inertia constraints violated; applying penalty.")
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
        scenario = Scenario(max_experiments=budget * len(space_list), seed=42, n_jobs=64)

        result = irace(target_runner, parameter_space, scenario, return_df=True, remove_metadata=True)
        best_configurations[algo_name] = result

        # Save results **after each algorithm**
        with open(output_file, "w") as f:
            json.dump({k: v.to_json() for k, v in best_configurations.items()}, f, indent=4)

        print(f"Saved best configuration for {algo_name} to {output_file}")
