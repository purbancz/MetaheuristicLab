import os
import logging
import datetime
import traceback
from typing import Any, Dict

import numpy as np
import pandas as pd
import skopt.plots
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args, dump, load
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt

from algorithm.single_objective_PSO import RebelPSO, EscapistPSO, RebelEscapistPSO, REAPSO, SingleObjectivePSO
from problem.n_variables.ackley import Ackley
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.schwefel import Schwefel

# Configure logging and output
logging.basicConfig(level=logging.INFO)
RESULTS_DIR = "optimization_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

current_run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

BENCHMARK_PROBLEMS = {
    "Rastrigin": Rastrigin(100),
    "Michalewicz": Michalewicz(100),
    "Ackley": Ackley(100),
    "Schwefel": Schwefel(100),
}
PROBLEM_WEIGHTS = {
    "Rastrigin": 0.4,
    "Michalewicz": 0.3,
    "Ackley": 0.2,
    "Schwefel": 0.1
}

# Algorithm parameter spaces
ALGO_SPACES = {
    'RebelPSO': [
        Real(0.05, 0.4, name='rebel_fraction'),
        Real(0.5, 2.5, name='c1'),
        Real(0.5, 2.5, name='c2'),
        Real(0.1, 1.4, name='w')
    ],
    'EscapistPSO': [
        Real(0.05, 0.4, name='escapist_fraction'),
        Real(0.5, 2.5, name='c1'),
        Real(0.5, 2.5, name='c2'),
        Real(0.1, 1.4, name='w')
    ],
    'RebelEscapistPSO': [
        Real(0.05, 0.4, name='rebel_fraction'),
        Real(0.05, 0.4, name='escapist_fraction'),
        Real(0.5, 2.5, name='c1'),
        Real(0.5, 2.5, name='c2'),
        Real(0.1, 1.4, name='w')
    ],
    'REAPSO': [
        Real(0.5, 2.5, name='c1'),
        Real(0.5, 2.5, name='c2'),
        Real(0.1, 1.4, name='w'),
        Real(0.05, 0.4, name='rebel_ratio'),
        Real(0.05, 0.4, name='escapist_ratio'),
        Real(0.4, 1.4, name='base_inertia'),
        Real(0.1, 0.6, name='min_inertia'),
        Real(0.6, 2, name='max_inertia')
    ]
}


class OptimizationRunner:
    def __init__(self, algorithm_name: str, max_evals: int = 25000, n_runs: int = 1):
        """
        Initializes the optimization runner.
        :param algorithm_name: Name of the algorithm to optimize.
        :param max_evals: Maximum evaluations for each run.
        :param n_runs: Number of runs per benchmark.
        """
        self.algorithm_name = algorithm_name
        self.max_evals = max_evals
        self.n_runs = n_runs
        self.space = ALGO_SPACES[algorithm_name]
        self.baseline_scores = {}
        self.best_params_per_problem = {problem: {'score': float('inf'), 'params': None}
                                        for problem in BENCHMARK_PROBLEMS}
        self.results: Any = None

        # Precompute baseline PSO values (in main process to avoid multiprocessing issues)
        for problem_name, problem in BENCHMARK_PROBLEMS.items():
            try:
                logging.info(f"Precomputing baseline PSO for {problem_name}...")
                bs = self.baseline_pso(problem)
                # If baseline is inf, set a large finite penalty instead.
                if np.isinf(bs):
                    bs = 1e6
                self.baseline_scores[problem_name] = bs
                logging.info(f"Baseline PSO score for {problem_name}: {self.baseline_scores[problem_name]}")
            except Exception as e:
                logging.error(f"Failed to compute baseline PSO for {problem_name}: {e}")
                self.baseline_scores[problem_name] = 1e6

    def objective(self, *args) -> float:
        """
        Objective function that evaluates a given set of parameters.
        """
        named_objective = use_named_args(self.space)(self._objective_function)
        return named_objective(*args)

    def _objective_function(self, **params: Dict) -> float:
        with ProcessPoolExecutor() as executor:
            futures = []
            for problem_name, problem in BENCHMARK_PROBLEMS.items():
                try:
                    future = executor.submit(
                        self.evaluate_problem,
                        problem,
                        params,
                        PROBLEM_WEIGHTS[problem_name]
                    )
                    if future is not None:
                        futures.append(future)
                except Exception as e:
                    logging.error(f"Failed to submit task for problem {problem_name}: {e}")

            weighted_results = []
            for future in futures:
                try:
                    res = future.result()
                    # In case res is infinite or nan, assign a high penalty.
                    if np.isinf(res) or np.isnan(res):
                        res = 1e6
                    weighted_results.append(res)
                except Exception as e:
                    logging.error(f"Error while fetching result from future: {e}")
                    weighted_results.append(1e6)

            total_score = np.sum(weighted_results)

        self.save_iteration_results(params, total_score)
        return total_score

    def create_algorithm(self, problem: Any, params: Dict) -> Any:
        """
        Constructs an algorithm instance with common and specific parameters.
        """
        common_params = {
            'problem': problem,
            'swarm_size': 100,
            'termination_criterion': StoppingByEvaluations(self.max_evals)
        }
        constructor = globals().get(self.algorithm_name)
        if constructor is None:
            raise ValueError(f"Algorithm {self.algorithm_name} not found.")
        return constructor(**{**common_params, **params})

    def evaluate_problem(self, problem: Any, params: Dict, weight: float) -> float:
        """
        Evaluates the algorithm on a given problem multiple times and returns a weighted score.
        """
        problem_name = problem.name()
        results = []
        for _ in range(self.n_runs):
            algo = self.create_algorithm(problem, params)
            algo.run()  # run() must return self
            # Ensure the result is finite; if not, assign a penalty.
            val = algo.result().objectives[0]
            if np.isinf(val) or np.isnan(val):
                val = 1e6
            results.append(val)

        # Retrieve cached baseline
        baseline = self.baseline_scores[problem_name]
        # Use Mann-Whitney U test to determine if the new run is significantly different.
        try:
            _, p_value = mannwhitneyu(results, [baseline] * len(results))
        except Exception as e:
            logging.error(f"Exception in Mann-Whitney U test for {problem_name}: {e}")
            p_value = 1.0

        penalty = 0 if p_value < 0.05 else 1000
        problem_score = weight * (np.median(results) + penalty)
        # If the computed score is inf or nan, assign a large finite penalty.
        if np.isinf(problem_score) or np.isnan(problem_score):
            problem_score = 1e6

        # Store best parameters if current run is better.
        if problem_score < self.best_params_per_problem[problem_name]['score']:
            self.best_params_per_problem[problem_name]['score'] = problem_score
            self.best_params_per_problem[problem_name]['params'] = params

        return problem_score

    @staticmethod
    def baseline_pso(problem: Any) -> float:
        """
        Runs a baseline PSO on the given problem and returns its objective value.
        This is executed in the main process.
        """
        logging.debug(f"🔍 Running baseline PSO for problem: {problem.name()}")
        try:
            pso = SingleObjectivePSO(
                problem=problem,
                swarm_size=100,
                c1=1.97,
                c2=1.97,  # note: use appropriate values; adjust as needed
                w=0.56,
                termination_criterion=StoppingByEvaluations(25000)
            )
            initial_solutions = pso.create_initial_solutions()
            if not initial_solutions:
                logging.error(f"🚨 Problem {problem.name()} failed to generate initial solutions!")
                return float('inf')
            baseline_algo = pso.run()
            if baseline_algo is None:
                logging.error(f"🚨 Baseline PSO run() returned None for {problem.name()}.")
                return float('inf')
            result = baseline_algo.result().objectives[0]
            logging.debug(f"✅ Baseline PSO result for {problem.name()}: {result}")
            return result
        except Exception as e:
            logging.error(f"🚨 Exception in baseline_pso() for {problem.name()}: {e}")
            logging.error(traceback.format_exc())
            return float('inf')

    def save_iteration_results(self, params: Dict, score: float) -> None:
        """
        Saves the result of each iteration to a CSV file.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{current_run_id}_{self.algorithm_name}_intermediate.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        result_entry = {
            "timestamp": timestamp,
            "algorithm": self.algorithm_name,
            "score": score,
            **params
        }
        df = pd.DataFrame([result_entry])
        df.to_csv(filepath, mode='a', header=not os.path.exists(filepath), index=False)

    def save_problem_specific_results(self):
        """
        Saves the best parameters for each algorithm on each benchmark problem.
        """
        for problem_name, data in self.best_params_per_problem.items():
            if data['params'] is None:
                continue
            filename = f"{current_run_id}_{self.algorithm_name}_{problem_name}_best_params.csv"
            filepath = os.path.join(RESULTS_DIR, filename)
            result_entry = {
                "algorithm": self.algorithm_name,
                "problem": problem_name,
                "best_score": data["score"],
                **data["params"]
            }
            df = pd.DataFrame([result_entry])
            df.to_csv(filepath, mode='w', header=True, index=False)
            logging.info(f"✅ Saved best parameters for {self.algorithm_name} on {problem_name}")

    def run_optimization(self, n_calls: int = 50, resume: bool = True) -> Any:
        """
        Runs the optimization process with checkpointing.
        """
        checkpoint_file = os.path.join(RESULTS_DIR, f"{self.algorithm_name}_checkpoint.pkl")
        if resume and os.path.exists(checkpoint_file):
            self.results = load(checkpoint_file)
            logging.info(f"Resuming optimization from {checkpoint_file}")
        else:
            self.results = gp_minimize(
                self.objective,
                self.space,
                n_calls=n_calls,
                random_state=42
            )
            dump(self.results, checkpoint_file)
        self.save_final_results()
        self.save_problem_specific_results()
        return self.results

    def save_final_results(self) -> None:
        """
        Saves convergence and objective plots as well as the parameter history.
        """
        plt.figure(figsize=(10, 6))
        skopt.plots.plot_convergence(self.results)
        plot_path = os.path.join(RESULTS_DIR, f"{current_run_id}_{self.algorithm_name}_convergence.png")
        plt.savefig(plot_path)
        plt.close()

        plt.figure(figsize=(12, 8))
        skopt.plots.plot_objective(self.results)
        importance_path = os.path.join(RESULTS_DIR, f"{current_run_id}_{self.algorithm_name}_importance.png")
        plt.savefig(importance_path)
        plt.close()

        params_df = pd.DataFrame(self.results.x_iters, columns=[dim.name for dim in self.space])
        params_df['score'] = self.results.func_vals
        csv_path = os.path.join(RESULTS_DIR, f"{current_run_id}_{self.algorithm_name}_params.csv")
        params_df.to_csv(csv_path, index=False)





class MetaOptimizer:
    def __init__(self, algorithm_class: Any):
        """
        Initializes the meta-optimizer.
        :param algorithm_class: The algorithm class to be tuned.
        """
        self.algorithm_class = algorithm_class
        self.global_params = None

    def global_stage(self) -> Any:
        """
        Runs a global optimization stage for the algorithm.
        """
        global_runner = OptimizationRunner(self.algorithm_class.__name__)
        self.global_params = global_runner.run_optimization()
        return self.global_params

    def problem_specific_stage(self) -> Dict[str, Any]:
        """
        Runs problem-specific optimization stages for each benchmark problem.
        """
        problem_params = {}
        for problem_name in BENCHMARK_PROBLEMS:
            runner = self.ProblemSpecificOptimizer(self.algorithm_class, problem_name)
            problem_params[problem_name] = runner.tune()
        return problem_params

    class ProblemSpecificOptimizer:
        def __init__(self, algorithm_class: Any, problem_name: str):
            """
            Initializes the problem-specific optimizer.
            """
            self.algorithm_class = algorithm_class
            self.problem = BENCHMARK_PROBLEMS[problem_name]
            self.space = ALGO_SPACES[algorithm_class.__name__]

        def tune(self) -> Any:
            """
            Tunes the parameters for a specific problem using gp_minimize.
            """
            return gp_minimize(
                lambda p: self.evaluate(p),
                self.space,
                n_calls=30,
                random_state=42
            )

        def evaluate(self, params: Dict) -> float:
            """
            Evaluates the algorithm on the problem with the given parameters.
            """
            algo = self.algorithm_class(self.problem, **params)
            algo.run()
            return algo.result().objectives[0]


# Usage example
if __name__ == "__main__":
    algorithms_to_tune = [
        'RebelPSO',
        'EscapistPSO',
        'RebelEscapistPSO',
        'REAPSO'
    ]

    for algo_name in algorithms_to_tune:
        logging.info(f"Starting optimization for {algo_name}")
        runner = OptimizationRunner(algo_name)
        result = runner.run_optimization(n_calls=100)

        logging.info(f"Best parameters for {algo_name}: {result.x}")
        logging.info(f"Best score for {algo_name}: {result.fun}")



