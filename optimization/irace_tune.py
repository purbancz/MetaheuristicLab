import os
import json
import traceback

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
    ContrarianDefeatistPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO, CDAPSO, EEAPSO
from algorithm.single_objective_PSO import SingleObjectivePSO
from algorithm.FAPSO import FAPSO
from algorithm.NPSO import NPSO
from algorithm.QTPSO import QTPSO
from algorithm.SPPPSO import SPPPSO
from algorithm.TDPSO import TDPSO
from irace import irace, ParameterSpace, Scenario, Experiment, Real, Integer, Bool
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
    'HybridPartialDisjointPSO': [
        # Core PSO Params
        Real("w", 0.01, 1.0),  # Inertia weight
        Real("c1", 0.01, 6.0),  # Standard cognitive coefficient
        Real("c2", 0.01, 6.0),  # Standard social coefficient
        # Special Role Coefficients
        Real("rejector_c", 0.01, 6.0),
        Real("defeatist_c", 0.01, 6.0),
        Real("escapist_c", 0.01, 6.0),
        Real("rebel_c", 0.01, 6.0),
        Real("contrarian_c", 0.01, 6.0),
        Real("eschewer_c", 0.01, 6.0),
        # Special Role Fractions (Require normalization in target-runner)
        # Range [0.0, 1.0] allows exploring full range, normalization ensures validity.
        Real("rejector_fraction", 0.01, 0.78),  # Max value should allow sum > 1 before normalization
        Real("defeatist_fraction", 0.01, 0.78),
        Real("escapist_fraction", 0.01, 0.78),
        # --- Cognitive sum constraint handled in runner ---
        Real("rebel_fraction", 0.01, 0.78),
        Real("contrarian_fraction", 0.01, 0.78),
        Real("eschewer_fraction", 0.01, 0.78),
        # --- Social sum constraint handled in runner ---
        # Behavior Flags
        Bool("assign_roles_every_iteration"),
    ],
    'HybridFullDisjointPSO': [  # Assuming the version with individual fraction parameters
        # Core PSO Params
        Real("w", 0.01, 1.0),  # Inertia weight
        Real("c1", 0.01, 6.0),  # Standard cognitive coefficient (used if role is 'std_cognitive')
        Real("c2", 0.01, 6.0),  # Standard social coefficient (used if role is 'std_social')
        # Special Role Coefficients
        Real("rejector_c", 0.01, 6.0),
        Real("defeatist_c", 0.01, 6.0),
        Real("escapist_c", 0.01, 6.0),
        Real("rebel_c", 0.01, 6.0),
        Real("contrarian_c", 0.01, 6.0),
        Real("eschewer_c", 0.01, 6.0),
        # Special Role Fractions (Require normalization in target-runner)
        # Range [0.0, 1.0] allows exploring full range, normalization ensures sum <= 1.
        Real("rejector_fraction", 0.01, 0.75),
        Real("defeatist_fraction", 0.01, 0.75),
        Real("escapist_fraction", 0.01, 0.75),
        Real("rebel_fraction", 0.01, 0.75),
        Real("contrarian_fraction", 0.01, 0.75),
        Real("eschewer_fraction", 0.01, 0.75),
        # --- Sum constraint (sum <= 1.0) handled in runner ---
        # Behavior Flags
        Bool("assign_roles_every_iteration"),
    ],
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
    #     Bool("assign_flags_every_iteration"),
    # ],

}

current_algorithm = None


def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    """
    Target runner function called by irace.
    - Receives a configuration from irace.
    - Normalizes fraction parameters for specific algorithms if needed.
    - Instantiates and runs the specified algorithm multiple times on multiple problems.
    - Calculates the average performance cost.
    - Returns the cost to irace.
    """
    global current_algorithm, problems, num_runs, solutions_size, max_evaluations # Access globals

    if current_algorithm is None:
         print("ERROR: current_algorithm global variable is not set!")
         return float('inf') # Cannot proceed without knowing which algorithm to run

    config = experiment.configuration
    print(f"\n--- Evaluating Algorithm: {current_algorithm} ---")
    print(f"Received Config: {config}")

    # Initialize normalized fractions with defaults (original values)
    rejector_fraction_norm = config.get("rejector_fraction", 0.0)
    defeatist_fraction_norm = config.get("defeatist_fraction", 0.0)
    escapist_fraction_norm = config.get("escapist_fraction", 0.0)
    rebel_fraction_norm = config.get("rebel_fraction", 0.0)
    contrarian_fraction_norm = config.get("contrarian_fraction", 0.0)
    eschewer_fraction_norm = config.get("eschewer_fraction", 0.0)

    # ---  specific to HybridPartialDisjointPSO ---
    if current_algorithm == 'HybridPartialDisjointPSO':
        cog_fractions_orig = {
            "rejector": config.get("rejector_fraction", 0.0),
            "defeatist": config.get("defeatist_fraction", 0.0),
            "escapist": config.get("escapist_fraction", 0.0),
        }
        sum_cog = sum(cog_fractions_orig.values())
        if sum_cog > 1.0 + 1e-9:
            print(f"Normalizing cognitive fractions (Sum: {sum_cog:.4f})")
            if sum_cog > 1e-9:
                rejector_fraction_norm = cog_fractions_orig["rejector"] / sum_cog
                defeatist_fraction_norm = cog_fractions_orig["defeatist"] / sum_cog
                escapist_fraction_norm = cog_fractions_orig["escapist"] / sum_cog
            else: # Should not happen if sum_cog > 1.0, but for safety
                 rejector_fraction_norm = 0.0
                 defeatist_fraction_norm = 0.0
                 escapist_fraction_norm = 0.0
        # No 'else' needed, initial values are already set from config if sum <= 1.0

        soc_fractions_orig = {
            "rebel": config.get("rebel_fraction", 0.0),
            "contrarian": config.get("contrarian_fraction", 0.0),
            "eschewer": config.get("eschewer_fraction", 0.0),
        }
        sum_soc = sum(soc_fractions_orig.values())
        if sum_soc > 1.0 + 1e-9: # Add tolerance
            print(f"Normalizing social fractions (Sum: {sum_soc:.4f})")
            if sum_soc > 1e-9:
                rebel_fraction_norm = soc_fractions_orig["rebel"] / sum_soc
                contrarian_fraction_norm = soc_fractions_orig["contrarian"] / sum_soc
                eschewer_fraction_norm = soc_fractions_orig["eschewer"] / sum_soc
            else:
                rebel_fraction_norm = 0.0
                contrarian_fraction_norm = 0.0
                eschewer_fraction_norm = 0.0
        # No 'else' needed

        print(f"Using Normalized Fractions: Rej={rejector_fraction_norm:.3f}, Def={defeatist_fraction_norm:.3f}, Esc={escapist_fraction_norm:.3f} | "
              f"Reb={rebel_fraction_norm:.3f}, Con={contrarian_fraction_norm:.3f}, Esh={eschewer_fraction_norm:.3f}")

    elif current_algorithm == 'HybridFullDisjointPSO':
        # Extract all special fractions for this algorithm
        all_special_fractions_orig = {
            "rejector": config.get("rejector_fraction", 0.0),
            "defeatist": config.get("defeatist_fraction", 0.0),
            "escapist": config.get("escapist_fraction", 0.0),
            "rebel": config.get("rebel_fraction", 0.0),
            "contrarian": config.get("contrarian_fraction", 0.0),
            "eschewer": config.get("eschewer_fraction", 0.0),
        }
        sum_all = sum(all_special_fractions_orig.values())

        # Normalize if the sum of *all* special fractions exceeds 1.0
        if sum_all > 1.0 + 1e-9:
            print(f"Normalizing ALL special fractions (Sum: {sum_all:.4f}) for {current_algorithm}")
            if sum_all > 1e-9:
                rejector_fraction_norm = all_special_fractions_orig["rejector"] / sum_all
                defeatist_fraction_norm = all_special_fractions_orig["defeatist"] / sum_all
                escapist_fraction_norm = all_special_fractions_orig["escapist"] / sum_all
                rebel_fraction_norm = all_special_fractions_orig["rebel"] / sum_all
                contrarian_fraction_norm = all_special_fractions_orig["contrarian"] / sum_all
                eschewer_fraction_norm = all_special_fractions_orig["eschewer"] / sum_all
            else:
                rejector_fraction_norm = 0.0
                defeatist_fraction_norm = 0.0
                escapist_fraction_norm = 0.0
                rebel_fraction_norm = 0.0
                contrarian_fraction_norm = 0.0
                eschewer_fraction_norm = 0.0

        print(
            f"Using Normalized Fractions: Rej={rejector_fraction_norm:.3f}, Def={defeatist_fraction_norm:.3f}, Esc={escapist_fraction_norm:.3f}, "
            f"Reb={rebel_fraction_norm:.3f}, Con={contrarian_fraction_norm:.3f}, Esh={eschewer_fraction_norm:.3f}")


    try:
        AlgorithmClass = globals()[current_algorithm]
    except KeyError:
        print(f"ERROR: Algorithm class '{current_algorithm}' not found in global scope.")
        return float('inf')

    results = []
    for problem in problems:
        problem_name = problem.get_name() if hasattr(problem, 'get_name') else problem.__class__.__name__
        print(f"  Problem: {problem_name}")
        for run_index in range(num_runs):
            print(f"    Run: {run_index + 1}/{num_runs}")
            try:
                # Instantiate the algorithm with potentially normalized fractions
                # Ensure ALL required parameters for the specific AlgorithmClass are provided
                # Using .get() for safety in case a parameter is somehow missing from config
                algo_params = {
                    "problem": problem,
                    "swarm_size": solutions_size,
                    "termination_criterion": StoppingByEvaluations(max_evaluations),
                    "w": config.get("w"), # Example, add all relevant params
                    "c1": config.get("c1"),
                    "c2": config.get("c2"),
                    # Add coefficients - use .get()
                    "rejector_c": config.get("rejector_c"),
                    "defeatist_c": config.get("defeatist_c"),
                    "escapist_c": config.get("escapist_c"),
                    "rebel_c": config.get("rebel_c"),
                    "contrarian_c": config.get("contrarian_c"),
                    "eschewer_c": config.get("eschewer_c"),
                    # Add boolean flags - use .get()
                    "assign_roles_every_iteration": config.get("assign_roles_every_iteration"),
                    # Add probability params if tuning HybridAdditivePSO
                    "std_cognitive_prob": config.get("std_cognitive_prob"),
                    "rejector_prob": config.get("rejector_prob"),
                    "defeatist_prob": config.get("defeatist_prob"),
                    "escapist_prob": config.get("escapist_prob"),
                    "std_social_prob": config.get("std_social_prob"),
                    "rebel_prob": config.get("rebel_prob"),
                    "contrarian_prob": config.get("contrarian_prob"),
                    "eschewer_prob": config.get("eschewer_prob"),
                    "assign_flags_every_iteration": config.get("assign_flags_every_iteration"),
                    # Pass NORMALIZED fractions
                    "rejector_fraction": rejector_fraction_norm,
                    "defeatist_fraction": defeatist_fraction_norm,
                    "escapist_fraction": escapist_fraction_norm,
                    "rebel_fraction": rebel_fraction_norm,
                    "contrarian_fraction": contrarian_fraction_norm,
                    "eschewer_fraction": eschewer_fraction_norm,
                    # Add any other parameters specific to other algorithms if needed
                    # "b1": config.get("b1"), ...
                }

                # Filter out None values (parameters not relevant to the current algorithm)
                filtered_params = {k: v for k, v in algo_params.items() if v is not None}

                # Instantiate with relevant parameters
                algorithm = AlgorithmClass(**filtered_params)

                algorithm.run()
                result = algorithm.result()

                # Check if result is valid
                if result is None or not hasattr(result, 'objectives') or not result.objectives:
                     print(f"      Warning: Run {run_index + 1} returned no valid result/objectives.")
                     results.append(float('inf')) # Penalize failed runs
                else:
                    # Append the first objective value (assuming single-objective)
                    results.append(result.objectives[0])
                    print(f"      Run {run_index + 1} result: {result.objectives[0]:.4f}")


            except Exception as e:
                print(f"      ERROR during run {run_index + 1}: {e}")
                print(f"      Algorithm: {current_algorithm}, Problem: {problem_name}")
                print(f"      Config (Original): {config}")
                print(f"      Passed Params (Filtered): {filtered_params}")
                # Print traceback for detailed debugging
                traceback.print_exc()
                results.append(float('inf')) # Penalize crashed runs


    print(f"--- Finished runs for config. Results: {results} ---")

    if not results:
        print(f"Warning: No successful runs completed for config: {config}")
        return float('inf') # Return infinite cost if no results obtained

    try:
        numeric_results = [r for r in results if isinstance(r, (int, float))]
        if not numeric_results:
             print(f"Warning: No numeric results found for config: {config}. Results: {results}")
             return float('inf')
        avg_result = np.mean(numeric_results)

    except Exception as e:
        print(f"Error calculating mean of results: {results}. Error: {e}")
        return float('inf') # Return inf if mean calculation fails


    if np.isnan(avg_result) or np.isinf(avg_result):
        cost = float('inf')
        print(f"Evaluated config: {config} -> Avg Cost resulted in NaN/Inf ({avg_result}), returning Inf")
    else:
        cost = float(avg_result)
        print(f"Evaluated config: {config} -> Final Avg Cost: {cost:.4f}")

    return cost




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
