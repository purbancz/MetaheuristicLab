import os
import time
import json
import sqlite3
from joblib import Parallel, delayed
from mpi4py import MPI
from skopt.space import Real
from skopt.utils import use_named_args, dump, load
from skopt import gp_minimize
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from algorithm.role_based.roles import RRAPSO, RebelPSO, RebelRejectorPSO, RejectorPSO

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

ALGORITHMS = {
    'RRAPSO': [
        Real(0.2, 2.5, name='b1'),
        Real(0.2, 2.5, name='b2'),
        Real(0.4, 1.4, name='base_inertia'),
        Real(0.1, 0.6, name='min_inertia'),
        Real(0.6, 2, name='max_inertia'),
        Real(0.05, 0.6, name='rebel_fraction'),
        Real(0.05, 0.6, name='rejector_fraction'),
    ],
    'RebelPSO': [
        Real(0.5, 2.5, name='b1'),
        Real(0.5, 2.5, name='b2'),
        Real(0.1, 1.4, name='w'),
        Real(0.05, 0.4, name='rebel_fraction'),
    ],
    'RejectorPSO': [
        Real(0.5, 2.5, name='b1'),
        Real(0.5, 2.5, name='b2'),
        Real(0.1, 1.4, name='w'),
        Real(0.05, 0.4, name='rejector_fraction'),
    ],
    'RebelRejectorPSO': [
        Real(0.5, 2.5, name='b1'),
        Real(0.5, 2.5, name='b2'),
        Real(0.1, 1.4, name='w'),
        Real(0.05, 0.4, name='rebel_fraction'),
        Real(0.05, 0.4, name='rejector_fraction'),
    ],
}


def save_result_to_db(algorithm_name, params, avg_result, run_number):
    conn = sqlite3.connect('optimization_results.db')
    cursor = conn.cursor()
    cursor.execute(f'''
        CREATE TABLE IF NOT EXISTS {algorithm_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_number TEXT,
            params TEXT,
            avg_result REAL
        )
    ''')
    cursor.execute(f'INSERT INTO {algorithm_name} (run_number, params, avg_result) VALUES (?, ?, ?)',
                   (run_number, json.dumps(params), avg_result))
    conn.commit()
    conn.close()


def parallel_objective(params, algorithm_name):
    problem = Rastrigin(100)
    algorithm_class = globals()[algorithm_name]
    algorithm = algorithm_class(
        problem=problem,
        swarm_size=100,
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
        **params
    )
    algorithm.run()
    return algorithm.result().objectives[0]


def run_optimization(algorithm_name, space):
    pickle_file = f'{algorithm_name}_results.pkl'
    prev_results = load(pickle_file) if os.path.exists(pickle_file) else None
    run_count = 0

    @use_named_args(space)
    def objective(**params):
        nonlocal run_count
        num_runs = 5
        results = Parallel(n_jobs=num_runs)(
            delayed(parallel_objective)(params, algorithm_name) for _ in range(num_runs))
        avg_result = sum(results) / num_runs
        run_count += 1
        save_result_to_db(algorithm_name, params, avg_result, run_count)
        return avg_result

    x0, y0 = (prev_results.x_iters, list(prev_results.func_vals)) if prev_results else (None, None)
    new_results = gp_minimize(objective, space, n_calls=10, x0=x0, y0=y0, random_state=0)
    dump(new_results, pickle_file, store_objective=False)

    best_data = {
        'run_number': 'best',
        'best_params': new_results.x,
        'best_objective': new_results.fun
    }
    save_result_to_db(algorithm_name, new_results.x, new_results.fun, 'best')

    return new_results


if rank == 0:
    for algo_name, param_space in ALGORITHMS.items():
        print(f"Running optimization for {algo_name} on {size} nodes...")
        run_optimization(algo_name, param_space)
