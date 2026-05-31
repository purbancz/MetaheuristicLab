import os
import json
from skopt.space import Real, Integer
from skopt.utils import use_named_args, dump, load
from skopt import gp_minimize
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from algorithm.NPSO import NPSO
from algorithm.QTPSO import QTPSO
from algorithm.SPPPSO import SPPPSO
from algorithm.TDPSO import TDPSO

n_calls = 500

ALGORITHMS = {
    # 'SingleObjectivePSO': [
    #     Real(0.01, 10, name='b1'),
    #     Real(0.01, 10, name='b2'),
    #     Real(0.01, 10, name='w'),
    # ],
    # 'RRAPSO': [
    #     Real(0.01, 10, name='b1'),
    #     Real(0.01, 10, name='b2'),
    #     Real(0.01, 10, name='base_inertia'),
    #     Real(0.01, 10, name='min_inertia'),
    #     Real(0.01, 10, name='max_inertia'),
    #     Real(0.05, 0.8, name='rebel_fraction'),
    #     Real(0.05, 0.8, name='rejector_fraction'),
    # ],
    'RebelPSO': [
        Real(0.9, 4, name='b1'),
        Real(0.7, 3, name='b2'),
        Real(0.01, 0.3, name='w'),
        Real(0.05, 0.8, name='rebel_fraction'),
    ],
    'RejectorPSO': [
        Real(0.8, 1.7, name='b1'),
        Real(0.8, 4.5, name='b2'),
        Real(0.6, 0.8, name='w'),
        Real(0.05, 0.8, name='rejector_fraction'),
    ],
    'RebelRejectorPSO': [
        Real(0.6, 2, name='b1'),
        Real(0.9, 5, name='b2'),
        Real(0.06, 0.3, name='w'),
        Real(0.05, 0.8, name='rebel_fraction'),
        Real(0.05, 0.8, name='rejector_fraction'),
    ],
    # 'QTPSO': [
    #     Real(0.01, 10, name='b1'),
    #     Real(0.01, 10, name='b2'),
    #     Real(0.01, 10, name='w'),
    #     Real(0.01, 1.0, name='quantum_prob'),
    #     Real(0.01, 1.0, name='chaos_strength'),
    # ],
    'SPPPSO': [
        Real(0.6, 2.5, name='b1'),
        Real(0.8, 5, name='b2'),
        Real(0.08, 0.5, name='w'),
        Real(0.01, 0.5, name='predator_ratio'),
        Real(0.01, 0.5, name='scavenger_ratio'),
    ],
    # 'TDPSO': [
    #     Real(0.01, 10, name='b1'),
    #     Real(0.01, 10, name='b2'),
    #     Real(0.01, 10, name='w'),
    #     Real(0.1, 5.0, name='temperature'),
    #     Real(0.9, 1.0, name='cooling_rate'),
    # ],
    'NPSO': [
        Real(0.2, 1.5, name='b1'),
        Real(0.7, 4, name='b2'),
        Real(0.01, 0.22, name='w'),
        Real(0.9, 1.0, name='spike_threshold'),
    ],
    'FAPSO': [
        Real(0.9, 2.3, name='b1'),
        Real(0.9, 4.2, name='b2'),
        Real(0.01, 0.5, name='w'),
        Integer(3, 5, name='fractal_depth'),
    ],
}


def run_optimization(algorithm_name, space):
    results_file = f'{algorithm_name}_results.json'
    pickle_file = f'{algorithm_name}_results.pkl'
    prev_results = load(pickle_file) if os.path.exists(pickle_file) else None
    run_count = 0

    @use_named_args(space)
    def objective(**params):
        nonlocal run_count
        problem = Rastrigin(100)
        num_runs = 5
        results = []

        AlgorithmClass = globals()[algorithm_name]
        for _ in range(num_runs):

            algorithm = AlgorithmClass(
                problem=problem,
                swarm_size=100,
                termination_criterion=StoppingByEvaluations(max_evaluations=25000),
                **params
            )
            algorithm.run()
            result = algorithm.result()
            results.append(result.objectives[0])

        run_count += 1
        avg_result = sum(results) / num_runs

        data = {
            'run_number': run_count,
            'params': params,
            'average_result': avg_result
        }
        with open(results_file, 'a') as f:
            f.write(json.dumps(data) + '\n')
        return avg_result

    x0, y0 = (prev_results.x_iters, list(prev_results.func_vals)) if prev_results else (None, None)
    new_results = gp_minimize(objective, space, n_calls=n_calls, x0=x0, y0=y0, random_state=0)
    dump(new_results, pickle_file, store_objective=False)

    best_data = {
        'run_number': 'best',
        'best_params': new_results.x,
        'best_objective': new_results.fun
    }
    with open(results_file, 'a') as f:
        f.write(json.dumps(best_data) + '\n')

    return new_results

if __name__ == "__main__":
    for algo_name, param_space in ALGORITHMS.items():
        print(f"Running optimization for {algo_name}...")
        run_optimization(algo_name, param_space)
