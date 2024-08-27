import os
import time
from skopt.space import Real, Integer
from skopt.utils import use_named_args, dump, load
from skopt import gp_minimize
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import PolynomialMutation, SBXCrossover
from algorithm.PGSHEA import PGSHEA

space = [
    Real(1e-3, 5, name='c1'),
    Real(1e-3, 5, name='c2'),
    Real(1e-5, 5, name='w'),
    Real(1e-3, 5, "log-uniform", name='mutation_factor'),
    Integer(1, 250, name='exchange_interval')
]

start = time.time()
run_count = 0
n_calls = 10
results_gp = None


@use_named_args(space)
def objective(c1, c2, w, mutation_factor, exchange_interval):
    global run_count
    problem = Rastrigin(100)
    mutation = PolynomialMutation(mutation_factor / problem.number_of_variables(), 20.0)
    crossover = SBXCrossover(1.0, 5.0)
    num_runs = 5
    results = []

    for _ in range(num_runs):
        algorithm = PGSHEA(
            problem=problem,
            solutions_size=100,
            c1=c1,
            c2=c2,
            w=w,
            mutation=mutation,
            crossover=crossover,
            swap_interval=exchange_interval,
            starting_algorithm='PSO',
            termination_criterion=StoppingByEvaluations(max_evaluations=25000)
        )
        algorithm.run()
        result = algorithm.result()
        results.append(result.objectives[0])

        run_count += 1
        print(f'\033[93mRun {run_count}/{num_runs * n_calls}\033[0m')
        print(f'\033[93mTime: {time.time() - start}\033[0m')

    average_result = sum(results) / len(results)
    print(f'\033[93mc1: {c1}, c2: {c2}, w: {w}, mutation_factor: {mutation_factor},\033[0m'
          f'\033[93m exchange_interval {exchange_interval},\033[0m'
          )
    print(f'\033[93m Average result: {average_result}\033[0m')
    return average_result


if os.path.exists('previous_results.pkl'):
    results_gp = load('previous_results.pkl')


def run_optimization():
    global results_gp
    x0, y0 = None, None
    additional_calls = n_calls

    if results_gp:
        # Load previous x and y values if results_gp already exists
        x0 = results_gp.x_iters
        y0 = list(results_gp.func_vals)

    new_results_gp = gp_minimize(
        objective,
        space,
        n_calls=additional_calls,
        x0=x0,
        y0=y0,
        random_state=0
    )

    # Manually combine the results if previous results exist
    if results_gp:
        new_x_iters = results_gp.x_iters + new_results_gp.x_iters
        new_func_vals = list(results_gp.func_vals) + list(new_results_gp.func_vals)
        new_results_gp.x_iters = new_x_iters
        new_results_gp.func_vals = new_func_vals

    # Save the updated results
    dump(new_results_gp, 'previous_results.pkl', store_objective=False)
    return new_results_gp


res_gp = run_optimization()
print("Best parameters:", res_gp.x)
print("Best objective:", res_gp.fun)
