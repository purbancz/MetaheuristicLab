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
    # constraints batch
    'EEAPSO': [
        Real("c1", 0.01, 6),  # Cognitive coefficient
        Real("c2", 0.01, 6),  # Social coefficient
        Real("ac1", 0.01, 6),  # Adaptive cognitive coefficient
        Real("ac2", 0.01, 6),  # Adaptive social coefficient
        Real("base_inertia", 0.01, 1),  # Base inertia weight
        Real("min_inertia", 0.01, 1),  # Minimum inertia weight
        Real("max_inertia", 0.01, 1),  # Maximum inertia weight
        Real("eschewer_fraction", 0.05, 0.8),  # Fraction of rebel particles
        Real("escapist_fraction", 0.05, 0.8),  # Fraction of rejector particles
        Integer("window_size", 10, 50),  # Window size for convergence
        Real("max_eschewer_fraction", 0.1, 0.98),  # Max limit for rebel fraction
        Real("max_escapist_fraction", 0.1, 0.98),  # Max limit for a rejector fraction
        Real("diversity_threshold", 0.001, 0.3),  # Threshold for diversity
        Real("improvement_threshold", 0.0001, 0.1),  # Threshold for improvement rate
    ],
    'CDAPSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("ac1", 0.01, 6),
        Real("ac2", 0.01, 6),
        Real("base_inertia", 0.01, 1),
        Real("min_inertia", 0.01, 1),
        Real("max_inertia", 0.01, 1),
        Real("contrarian_fraction", 0.05, 0.8),
        Real("defeatist_fraction", 0.05, 0.8),
        Integer("window_size", 10, 50),
        Real("max_contrarian_fraction", 0.1, 0.98),
        Real("max_defeatist_fraction", 0.1, 0.98),
        Real("diversity_threshold", 0.001, 0.3),
        Real("improvement_threshold", 0.0001, 0.1),
    ],
    'RRAPSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("ac1", 0.01, 6),
        Real("ac2", 0.01, 6),
        Real("base_inertia", 0.01, 1),
        Real("min_inertia", 0.01, 1),
        Real("max_inertia", 0.01, 1),
        Real("rebel_fraction", 0.05, 0.8),
        Real("rejector_fraction", 0.05, 0.8),
        Integer("window_size", 10, 50),
        Real("max_rebel_fraction", 0.1, 0.98),
        Real("max_rejector_fraction", 0.1, 0.98),
        Real("diversity_threshold", 0.001, 0.3),
        Real("improvement_threshold", 0.0001, 0.1),
    ],
    'CoAdaptativePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("max_c1", 4, 20),
        Real("max_c2", 4, 20),
        Real("w", 0.01, 2),
    ],
    'IndividualAdaptivePSO': [
        Real("c1", 0.01, 6),
        Real("c2", 0.01, 6),
        Real("max_c1", 4, 20),
        Real("max_c2", 4, 20),
        Real("w", 0.01, 2),
    ],

    # #     # Separate run
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
    #     # Bool("assign_roles_every_iteration"),
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
    #     # Bool("assign_roles_every_iteration"),
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
}

current_algorithm = None


def normalize_fraction_sum(fractions_dict: dict, max_sum: float = 1.0) -> dict:
    """Normalizes fractions in a dictionary if their sum exceeds max_sum."""
    current_sum = sum(fractions_dict.values())
    normalized_fractions = fractions_dict.copy() # Avoid modifying original dict directly if passed from config

    if current_sum > max_sum + 1e-9: # Use tolerance
        print(f"Normalizing fractions (Sum: {current_sum:.4f} > {max_sum})")
        if current_sum > 1e-9: # Avoid division by zero
            for role in normalized_fractions:
                normalized_fractions[role] /= current_sum
            # Optional: Rescale to max_sum if you strictly want sum == max_sum
            # factor = max_sum / sum(normalized_fractions.values()) # Should be close to 1
            # for role in normalized_fractions: normalized_fractions[role] *= factor
        else: # Sum is zero or negative (shouldn't happen with positive fractions)
            for role in normalized_fractions: normalized_fractions[role] = 0.0
    # else: sum is valid, return original (copied) fractions
    return normalized_fractions

def repair_max_param_constraints(config: dict) -> dict:
    """Repairs constraints like max_param >= param by setting max_param = param if violated."""
    repaired_config = config.copy() # Work on a copy

    constraints_to_check = [
        ("c1", "max_c1"),
        ("c2", "max_c2"),
        ("min_inertia", "base_inertia"), # base must be >= min
        ("base_inertia", "max_inertia"),  # max must be >= base
        ("min_inertia", "max_inertia"),  # max must be >= min (implied by above two)
        # Fractions for EEAPSO, CDAPSO, RRAPSO
        ("eschewer_fraction", "max_eschewer_fraction"),
        ("escapist_fraction", "max_escapist_fraction"),
        ("contrarian_fraction", "max_contrarian_fraction"),
        ("defeatist_fraction", "max_defeatist_fraction"),
        ("rebel_fraction", "max_rebel_fraction"),
        ("rejector_fraction", "max_rejector_fraction"),
    ]

    for param, max_param in constraints_to_check:
        # Check if both parameters exist in the configuration for the current algorithm
        if param in repaired_config and max_param in repaired_config:
            param_val = repaired_config[param]
            max_param_val = repaired_config[max_param]

            if max_param_val < param_val:
                print(f"Repairing constraint: {max_param} ({max_param_val:.4f}) < {param} ({param_val:.4f}). Setting {max_param} = {param_val:.4f}")
                repaired_config[max_param] = param_val

    # Special check for min/base/max inertia order
    if "min_inertia" in repaired_config and "base_inertia" in repaired_config and "max_inertia" in repaired_config:
         if repaired_config["base_inertia"] < repaired_config["min_inertia"]:
             repaired_config["base_inertia"] = repaired_config["min_inertia"]
             print(f"Repairing inertia: base < min. Set base = min ({repaired_config['min_inertia']:.4f})")
         if repaired_config["max_inertia"] < repaired_config["base_inertia"]:
              repaired_config["max_inertia"] = repaired_config["base_inertia"]
              print(f"Repairing inertia: max < base. Set max = base ({repaired_config['base_inertia']:.4f})")

    return repaired_config

# ==============================================================================
# Main Target Runner Function
# ==============================================================================

def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    """
    Universal target runner with constraint handling.
    """
    global current_algorithm, problems, num_runs, solutions_size, max_evaluations

    if current_algorithm is None:
         print("ERROR: current_algorithm global variable is not set!")
         return float('inf')

    config = experiment.configuration
    print(f"\n--- Evaluating Algorithm: {current_algorithm} ---")
    print(f"Received Config: {config}")

    # --- Step 1: Repair Dependent Max Parameter Constraints ---
    # Apply this universally, as it won't affect algos without these params
    repaired_config = repair_max_param_constraints(config)
    if repaired_config != config:
        print(f"Config after repair: {repaired_config}")


    # --- Step 2: Handle Fraction Normalization (Algorithm Specific) ---
    # Initialize potentially normalized fractions with values from the *repaired* config
    final_params = repaired_config.copy() # Start building the final params to pass

    if current_algorithm == 'HybridPartialDisjointPSO':
        cog_fractions = {"rejector": repaired_config.get("rejector_fraction", 0.0), "defeatist": repaired_config.get("defeatist_fraction", 0.0), "escapist": repaired_config.get("escapist_fraction", 0.0)}
        norm_cog = normalize_fraction_sum(cog_fractions, 1.0)
        final_params["rejector_fraction"] = norm_cog["rejector"]
        final_params["defeatist_fraction"] = norm_cog["defeatist"]
        final_params["escapist_fraction"] = norm_cog["escapist"]

        soc_fractions = {"rebel": repaired_config.get("rebel_fraction", 0.0), "contrarian": repaired_config.get("contrarian_fraction", 0.0), "eschewer": repaired_config.get("eschewer_fraction", 0.0)}
        norm_soc = normalize_fraction_sum(soc_fractions, 1.0)
        final_params["rebel_fraction"] = norm_soc["rebel"]
        final_params["contrarian_fraction"] = norm_soc["contrarian"]
        final_params["eschewer_fraction"] = norm_soc["eschewer"]

        print(f"Using Normalized Fractions: Rej={final_params['rejector_fraction']:.3f}, Def={final_params['defeatist_fraction']:.3f}, Esc={final_params['escapist_fraction']:.3f} | "
              f"Reb={final_params['rebel_fraction']:.3f}, Con={final_params['contrarian_fraction']:.3f}, Esh={final_params['eschewer_fraction']:.3f}")


    elif current_algorithm == 'HybridFullDisjointPSO':
        special_keys = ["rejector", "defeatist", "escapist", "rebel", "contrarian", "eschewer"]
        all_special_fractions = {k: repaired_config.get(f"{k}_fraction", 0.0) for k in special_keys}
        norm_special = normalize_fraction_sum(all_special_fractions, 1.0)
        # Update final_params with normalized values
        for k, v in norm_special.items():
             final_params[f"{k}_fraction"] = v

        print(f"Using Normalized Fractions: Rej={final_params['rejector_fraction']:.3f}, Def={final_params['defeatist_fraction']:.3f}, Esc={final_params['escapist_fraction']:.3f}, "
              f"Reb={final_params['rebel_fraction']:.3f}, Con={final_params['contrarian_fraction']:.3f}, Esh={final_params['eschewer_fraction']:.3f}")


    # --- Step 3: Get Algorithm Class ---
    try:
        AlgorithmClass = globals()[current_algorithm]
    except KeyError:
        print(f"ERROR: Algorithm class '{current_algorithm}' not found in global scope.")
        return float('inf')

    # --- Step 4: Run Experiments ---
    results = []
    for problem in problems:
        problem_name = problem.get_name() if hasattr(problem, 'get_name') else problem.__class__.__name__
        print(f"  Problem: {problem_name}")
        for run_index in range(num_runs):
            # print(f"    Run: {run_index + 1}/{num_runs}") # Make logging less verbose
            try:
                # Build parameter dictionary for constructor
                # We pass the potentially repaired/normalized `final_params`
                constructor_params = {
                    "problem": problem,
                    "swarm_size": solutions_size,
                    "termination_criterion": StoppingByEvaluations(max_evaluations),
                }
                # Add only the parameters relevant to this algorithm's constructor
                # Inspect AlgorithmClass.__init__ signature or use a predefined list per algo
                # For now, assume all keys in final_params might be relevant and filter later if needed
                constructor_params.update(final_params)

                # Filter out parameters the specific AlgorithmClass constructor doesn't accept
                import inspect
                sig = inspect.signature(AlgorithmClass.__init__)
                allowed_params = {k for k in sig.parameters if k != 'self'}
                filtered_constructor_params = {k: v for k, v in constructor_params.items() if k in allowed_params}


                # Instantiate with relevant parameters
                # print(f"      Instantiating {current_algorithm} with params: {filtered_constructor_params}") # Debug
                algorithm = AlgorithmClass(**filtered_constructor_params)

                algorithm.run()
                result = algorithm.result()

                if result is None or not hasattr(result, 'objectives') or not result.objectives:
                     # print(f"      Warning: Run {run_index + 1} returned no valid result/objectives.")
                     results.append(float('inf'))
                else:
                    obj_value = result.objectives[0]
                    results.append(obj_value)
                    # print(f"      Run {run_index + 1} result: {obj_value:.4f}") # Less verbose

            except Exception as e:
                print(f"      ERROR during run {run_index + 1} on {problem_name}: {e}")
                print(f"      Algorithm: {current_algorithm}")
                print(f"      Config (Original): {config}")
                print(f"      Passed Params: {filtered_constructor_params}")
                traceback.print_exc()
                results.append(float('inf'))

    # --- Step 5: Process Results ---
    print(f"--- Finished runs for config. Num results: {len(results)} ---")
    # (Keep the robust result processing from the previous version)
    if not results: return float('inf')
    try:
        numeric_results = [r for r in results if isinstance(r, (int, float)) and np.isfinite(r)] # Filter Inf/NaN
        if not numeric_results: return float('inf')
        avg_result = np.mean(numeric_results)
    except Exception as e: print(f"Error calculating mean: {e}"); return float('inf')
    if np.isnan(avg_result) or np.isinf(avg_result): cost = float('inf')
    else: cost = float(avg_result)
    print(f"Evaluated config: {config} -> Final Avg Cost: {cost:.4f}")
    return cost




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
