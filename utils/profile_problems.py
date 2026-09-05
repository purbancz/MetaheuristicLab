"""Per-problem evaluation-cost profile of the active benchmark suite.

Times problem.evaluate() for every problem returned by setup_experiment()
and projects the campaign cost per algorithm-problem pair under the
EVALUATIONS_PER_DIMENSION budget with NO_OF_RUNS runs.

Run from the repository root:  python utils/profile_problems.py
"""

import os
import sys
import time

# Make the script runnable as `python utils/profile_problems.py` (Python puts
# the script's own directory on sys.path, not the repository root).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from jmetal.core.solution import FloatSolution

from experiment.globals import EVALUATIONS_PER_DIMENSION, NO_OF_RUNS
from experiment.setup import setup_experiment


def time_evaluate(problem, min_seconds=0.2, max_reps=1000):
    rng = np.random.default_rng(42)
    lower = np.asarray(problem.lower_bound, dtype=float)
    upper = np.asarray(problem.upper_bound, dtype=float)
    solution = FloatSolution(problem.lower_bound, problem.upper_bound, 1, 0)
    solution.objectives = [0.0]
    solution.variables = list(rng.uniform(lower, upper))

    problem.evaluate(solution)  # warm-up

    reps = 0
    start = time.perf_counter()
    while True:
        problem.evaluate(solution)
        reps += 1
        elapsed = time.perf_counter() - start
        if elapsed >= min_seconds or reps >= max_reps:
            return elapsed / reps


def main():
    (_, _, problems, *_rest) = setup_experiment()

    rows = []
    for problem in problems:
        dim = problem.number_of_variables()
        seconds = time_evaluate(problem)
        budget = EVALUATIONS_PER_DIMENSION * dim
        cpu_hours = seconds * budget * NO_OF_RUNS / 3600.0
        rows.append((problem.name(), dim, seconds * 1000.0, cpu_hours))

    rows.sort(key=lambda r: r[2], reverse=True)

    total = sum(r[3] for r in rows)
    print(f"\n{'problem':45s} {'dim':>5s} {'ms/eval':>9s} {'CPU-h/algorithm':>16s}")
    print("-" * 80)
    for name, dim, ms, hours in rows:
        print(f"{name:45s} {dim:5d} {ms:9.3f} {hours:16.1f}")
    print("-" * 80)
    print(f"{'SUITE TOTAL (per algorithm, '+str(NO_OF_RUNS)+' runs, 10^4*D)':45s} "
          f"{'':5s} {'':9s} {total:16.1f}")
    print(f"x active algorithms and / cores for wall-time estimates.")


if __name__ == "__main__":
    main()
