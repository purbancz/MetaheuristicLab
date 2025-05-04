import os
import json
import random
import traceback

import numpy as np
from datetime import datetime
import inspect  # Need inspect for filtering params

from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.hybrid_diverse import HybridPartialDisjointPSO, HybridFullDisjointPSO, HybridAdditivePSO, \
    HybridFullDisjointPSO_WithRandom, HybridPartialDisjointPSO_WithRandom, HybridAdditivePSO_WithRandom
from algorithm.AdaptivePSO import CoAdaptativePSO, IndividualAdaptivePSO
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO, \
    ReverseLearningPSO
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO, CDAPSO, EEAPSO, AAAPSO, NAPSO
from algorithm.single_objective_PSO import SingleObjectivePSO
from algorithm.reinitialized_PSO import FRAPSO
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
    'AAAPSO': [
        Real("c1", 0.01, 6.0),
        Real("c2", 0.01, 6.0),
        Real("base_inertia", 0.01, 1.0),
        Real("min_inertia", 0.01, 1.0),
        Real("max_inertia", 0.01, 1.0),
        Real("random_strength", 0.01, 1.0),
        Real("anarchic_fraction", 0.01, 0.8),
        Real("amnesiac_fraction", 0.01, 0.8),
        Integer("window_size", 10, 50),
        Real("max_anarchic_fraction", 0.01, 0.98),
        Real("max_amnesiac_fraction", 0.01, 0.98),
        Real("diversity_threshold", 0.001, 0.3),
        Real("improvement_threshold", 0.0001, 0.1),
    ],
    'NAPSO': [
        Real("c1", 0.01, 6.0),
        Real("c2", 0.01, 6.0),
        Real("base_inertia", 0.01, 1.0),
        Real("min_inertia", 0.01, 1.0),
        Real("max_inertia", 0.01, 1.0),
        Real("noise_strength", 0.01, 1.0),
        Real("noisy_fraction", 0.01, 0.8),
        Real("max_noisy_fraction", 0.01, 0.98),
        Integer("window_size", 10, 50),
        Real("diversity_threshold", 0.001, 0.3),
        Real("improvement_threshold", 0.0001, 0.1),
    ],
    'HybridFullDisjointPSO_WithRandom': [
        Real("w", 0.01, 1.0),
        Real("c1", 0.01, 6.0),
        Real("c2", 0.01, 6.0),
        Real("rejector_c", 0.01, 6.0),
        Real("defeatist_c", 0.01, 6.0),
        Real("escapist_c", 0.01, 6.0),
        Real("amnesiac_c", 0.01, 6.0),
        Real("rebel_c", 0.01, 6.0),
        Real("contrarian_c", 0.01, 6.0),
        Real("eschewer_c", 0.01, 6.0),
        Real("anarchic_c", 0.01, 6.0),
        Real("rejector_fraction", 0.01, 0.73),
        Real("defeatist_fraction", 0.01, 0.73),
        Real("escapist_fraction", 0.01, 0.73),
        Real("amnesiac_fraction", 0.01, 0.73),
        Real("rebel_fraction", 0.01, 0.73),
        Real("contrarian_fraction", 0.01, 0.73),
        Real("eschewer_fraction", 0.01, 0.73),
        Real("anarchic_fraction", 0.01, 0.73),
        Bool("assign_roles_every_iteration"),
    ],
    'HybridPartialDisjointPSO_WithRandom': [
        Real("w", 0.01, 1.0),
        Real("c1", 0.01, 6.0),
        Real("c2", 0.01, 6.0),
        Real("rejector_c", 0.01, 6.0),
        Real("defeatist_c", 0.01, 6.0),
        Real("escapist_c", 0.01, 6.0),
        Real("rebel_c", 0.01, 6.0),
        Real("contrarian_c", 0.01, 6.0),
        Real("eschewer_c", 0.01, 6.0),
        Real("amnesiac_c", 0.01, 6.0),
        Real("anarchic_c", 0.01, 6.0),
        Real("rejector_fraction", 0.01, 0.77),
        Real("defeatist_fraction", 0.01, 0.77),
        Real("escapist_fraction", 0.01, 0.77),
        Real("amnesiac_fraction", 0.01, 0.77),
        Real("rebel_fraction", 0.01, 0.77),
        Real("contrarian_fraction", 0.01, 0.77),
        Real("eschewer_fraction", 0.01, 0.77),
        Real("anarchic_fraction", 0.01, 0.77),
        Bool("assign_roles_every_iteration"),
    ],
    'HybridAdditivePSO_WithRandom': [
        Real("w", 0.01, 1.0),
        Real("c1", 0.01, 6.0),
        Real("c2", 0.01, 6.0),
        Real("rejector_c", 0.01, 6.0),
        Real("defeatist_c", 0.01, 6.0),
        Real("escapist_c", 0.01, 6.0),
        Real("rebel_c", 0.01, 6.0),
        Real("contrarian_c", 0.01, 6.0),
        Real("eschewer_c", 0.01, 6.0),
        Real("anarchic_c", 0.01, 6.0),
        Real("amnesiac_c", 0.01, 6.0),
        Real("std_cognitive_prob", 0.01, 1.0),
        Real("rejector_prob", 0.01, 1.0),
        Real("defeatist_prob", 0.01, 1.0),
        Real("escapist_prob", 0.01, 1.0),
        Real("amnesiac_prob", 0.01, 1.0),
        Real("std_social_prob", 0.01, 1.0),
        Real("rebel_prob", 0.01, 1.0),
        Real("contrarian_prob", 0.01, 1.0),
        Real("eschewer_prob", 0.01, 1.0),
        Real("anarchic_prob", 0.01, 1.0),
        Bool("assign_flags_every_iteration"),
    ],
}

current_algorithm = None


# ==============================================================================
# Constraint Handling Functions
# ==============================================================================
def normalize_fraction_sum(fractions_dict: dict, max_sum: float = 1.0) -> dict:
    """Normalizes fractions in a dictionary if their sum exceeds max_sum."""
    current_sum = sum(v for v in fractions_dict.values() if isinstance(v, (int, float)))  # Ensure numeric sum
    normalized_fractions = fractions_dict.copy()

    if current_sum > max_sum + 1e-9:
        print(f"Normalizing fractions (Sum: {current_sum:.4f} > {max_sum})")
        if current_sum > 1e-9:
            factor = max_sum / current_sum
            for role in normalized_fractions:
                if isinstance(normalized_fractions[role], (int, float)):
                    normalized_fractions[role] *= factor
        else:
            for role in normalized_fractions: normalized_fractions[role] = 0.0
    return normalized_fractions


def repair_max_param_constraints_random(config: dict) -> dict:
    """
    Repairs constraints like max_param >= param.
    - For simple max_param >= param, clamps max_param = param if violated.
    - For min <= base <= max, SWAPS min and max if min > max, then repairs base
      by assigning a random value within the valid [min, max] range.
    """
    repaired_config = config.copy()

    # --- Standard Param <= Max_Param Constraints ---
    # (Clamping remains the best approach here)
    constraints_to_check = [
        ("c1", "max_c1"), ("c2", "max_c2"),
        ("eschewer_fraction", "max_eschewer_fraction"), ("escapist_fraction", "max_escapist_fraction"),
        ("contrarian_fraction", "max_contrarian_fraction"), ("defeatist_fraction", "max_defeatist_fraction"),
        ("rebel_fraction", "max_rebel_fraction"), ("rejector_fraction", "max_rejector_fraction"),
        ("anarchic_fraction", "max_anarchic_fraction"), ("amnesiac_fraction", "max_amnesiac_fraction"),
        ("noisy_fraction", "max_noisy_fraction"),
    ]
    for param, max_param in constraints_to_check:
        if param in repaired_config and max_param in repaired_config:
            param_val = repaired_config[param]; max_param_val = repaired_config[max_param]
            if isinstance(param_val, (int, float)) and isinstance(max_param_val, (int, float)):
                if max_param_val < param_val:
                    # Clamp max = param
                    print(f"Repairing constraint: {max_param} ({max_param_val:.4f}) < {param} ({param_val:.4f}). Clamping {max_param} = {param_val:.4f}")
                    repaired_config[max_param] = param_val

    # --- Min <= Base <= Max Constraints (e.g., Inertia) ---
    min_base_max_triplets = [
        ("min_inertia", "base_inertia", "max_inertia")
        # Add others here, e.g., ("min_fraction", "base_fraction", "max_fraction")
    ]

    for min_key, base_key, max_key in min_base_max_triplets:
        if min_key in repaired_config and base_key in repaired_config and max_key in repaired_config:
            min_val = repaired_config[min_key]
            base_val = repaired_config[base_key]
            max_val = repaired_config[max_key]

            # Ensure numeric values
            if not (isinstance(min_val, (int, float)) and isinstance(base_val, (int, float)) and isinstance(max_val, (int, float))):
                print(f"Warning: Non-numeric values encountered for {min_key}/{base_key}/{max_key}. Skipping repair.")
                continue

            if max_val < min_val:
                print(f"Repairing bounds: {max_key} ({max_val:.4f}) < {min_key} ({min_val:.4f}). Swapping values.")
                repaired_config[min_key], repaired_config[max_key] = max_val, min_val # Swap in dict
                # Update local variables for subsequent checks
                min_val, max_val = max_val, min_val

            if base_val < min_val:
                new_base = random.uniform(min_val, max_val)
                print(f"Repairing value: {base_key} ({base_val:.4f}) < {min_key} ({min_val:.4f}). Setting {base_key} to random in [{min_val:.4f}, {max_val:.4f}]: {new_base:.4f}")
                repaired_config[base_key] = new_base
            elif base_val > max_val:
                new_base = random.uniform(min_val, max_val)
                print(f"Repairing value: {base_key} ({base_val:.4f}) > {max_key} ({max_val:.4f}). Setting {base_key} to random in [{min_val:.4f}, {max_val:.4f}]: {new_base:.4f}")
                repaired_config[base_key] = new_base

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
    repaired_config = repair_max_param_constraints_random(config)
    if repaired_config != config:
        print(f"Config after repair: {repaired_config}")


    # --- Step 2: Handle Fraction Normalization (Algorithm Specific) ---
    # Initialize potentially normalized fractions with values from the *repaired* config
    final_params = repaired_config.copy()  # Start building the final params to pass

    # Define parameter groups for clarity
    cognitive_fraction_keys = ["rejector_fraction", "defeatist_fraction", "escapist_fraction", "amnesiac_fraction"]
    social_fraction_keys = ["rebel_fraction", "contrarian_fraction", "eschewer_fraction", "anarchic_fraction"]
    all_special_fraction_keys = cognitive_fraction_keys + social_fraction_keys + ["wanderer_fraction"]

    if current_algorithm == 'HybridPartialDisjointPSO' or current_algorithm == 'HybridPartialDisjointPSO_WithRandom':
        cog_fractions = {k: repaired_config.get(k, 0.0) for k in cognitive_fraction_keys if k in repaired_config}
        if cog_fractions:  # Only normalize if relevant keys exist
            norm_cog = normalize_fraction_sum(cog_fractions, 1.0)
            final_params.update(norm_cog)  # Update final_params with normalized values

        soc_fractions = {k: repaired_config.get(k, 0.0) for k in social_fraction_keys if k in repaired_config}
        if soc_fractions:
            norm_soc = normalize_fraction_sum(soc_fractions, 1.0)
            final_params.update(norm_soc)

        print(
            f"Using Normalized Fractions: Rej={final_params['rejector_fraction']:.3f}, Def={final_params['defeatist_fraction']:.3f}, Esc={final_params['escapist_fraction']:.3f} | "
            f"Reb={final_params['rebel_fraction']:.3f}, Con={final_params['contrarian_fraction']:.3f}, Esh={final_params['eschewer_fraction']:.3f}")



    elif current_algorithm == 'HybridFullDisjointPSO' or current_algorithm == 'HybridFullDisjointPSO_WithRandom':
        all_special_fractions = {k: repaired_config.get(k, 0.0) for k in all_special_fraction_keys if
                                 k in repaired_config}
        if all_special_fractions:
            norm_special = normalize_fraction_sum(all_special_fractions, 1.0)
            final_params.update(norm_special)

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
        # print(f"  Problem: {problem_name}") # Verbose
        for run_index in range(num_runs):
            try:
                # Prepare parameters for this specific algorithm's constructor
                constructor_params = {"problem": problem, "swarm_size": solutions_size,
                                      "termination_criterion": StoppingByEvaluations(max_evaluations)}
                # Add parameters from the (potentially repaired/normalized) final_params dict
                constructor_params.update(final_params)

                # Filter parameters to only those accepted by the specific constructor
                sig = inspect.signature(AlgorithmClass.__init__)
                allowed_params = {k for k in sig.parameters if k != 'self'}
                # Include **kwargs if the constructor accepts them
                if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
                    # If **kwargs exists, we don't need strict filtering, but it's safer to filter known ones
                    filtered_constructor_params = {k: v for k, v in constructor_params.items() if
                                                   k in allowed_params or k in ['problem', 'swarm_size',
                                                                                'termination_criterion']}
                    # Add remaining items from final_params if they are not standard keys already included
                    for k, v in final_params.items():
                        if k not in filtered_constructor_params:
                            filtered_constructor_params[k] = v  # Assume they are handled by **kwargs
                else:
                    # Strict filtering if no **kwargs
                    filtered_constructor_params = {k: v for k, v in constructor_params.items() if
                                                   k in allowed_params}


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
    print(f"Evaluated config: {repaired_config} -> Final Avg Cost: {cost:.4f}")
    return cost


if __name__ == "__main__":
    best_configurations = {}
    output_file = "irace_best_configurations.json"

    for algo_name, space_list in parameter_spaces.items():
        current_algorithm = algo_name
        print(f"Optimizing parameters for {algo_name} ...")

        parameter_space = ParameterSpace(params=space_list)
        scenario = Scenario(max_experiments=budget * len(space_list), seed=42, n_jobs=48)

        result = irace(target_runner, parameter_space, scenario, return_df=True, remove_metadata=True)
        best_configurations[algo_name] = result

        # Save results **after each algorithm**
        with open(output_file, "w") as f:
            json.dump({k: v.to_json() for k, v in best_configurations.items()}, f, indent=4)

        print(f"Saved best configuration for {algo_name} to {output_file}")
