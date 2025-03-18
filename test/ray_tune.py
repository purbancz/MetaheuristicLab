import os
import time
import json
import sqlite3
import ray
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO
from algorithm.single_objective_PSO import SingleObjectivePSO


ALGORITHMS = {
    'RebelPSO': {
        "c1": tune.uniform(0.2, 3),
        "c2": tune.uniform(0.2, 3),
        "ac2": tune.uniform(0.2, 3),
        "w": tune.uniform(0.1, 1.4),
        "rebel_fraction": tune.uniform(0.05, 0.6),
    },
    'RejectorPSO': {
        "c1": tune.uniform(0.2, 3),
        "c2": tune.uniform(0.2, 3),
        "ac1": tune.uniform(0.2, 3),
        "w": tune.uniform(0.1, 1.4),
        "escapist_fraction": tune.uniform(0.05, 0.6),
    },
    'RebelRejectorPSO': {
        "c1": tune.uniform(0.2, 3),
        "c2": tune.uniform(0.2, 3),
        "ac1": tune.uniform(0.2, 3),
        "ac2": tune.uniform(0.2, 3),
        "w": tune.uniform(0.1, 1.4),
        "rebel_fraction": tune.uniform(0.05, 0.6),
        "escapist_fraction": tune.uniform(0.05, 0.6),
    },
}


def objective(config, algorithm_name):
    problem = Rastrigin(100)
    num_runs = 5
    results = []
    AlgorithmClass = globals()[algorithm_name]

    for _ in range(num_runs):
        algorithm = AlgorithmClass(
            problem=problem,
            swarm_size=100,
            termination_criterion=StoppingByEvaluations(max_evaluations=25000),
            **config
        )
        algorithm.run()
        result = algorithm.result()
        results.append(result.objectives[0])

    avg_result = sum(results) / num_runs
    tune.report(average_result=avg_result)


def run_ray_optimization(algorithm_name, param_space):
    scheduler = PopulationBasedTraining(
        time_attr="training_iteration",
        metric="average_result",
        mode="min",
        perturbation_interval=5,
        hyperparam_mutations={
            "b1": tune.uniform(0.2, 3),
            "b2": tune.uniform(0.2, 3),
            "base_inertia": tune.uniform(0.4, 1.4),
            "min_inertia": tune.uniform(0.1, 0.6),
            "max_inertia": tune.uniform(0.6, 2),
            "rebel_fraction": tune.uniform(0.05, 0.6),
            "escapist_fraction": tune.uniform(0.05, 0.6),
            "w": tune.uniform(0.1, 1.4),
            "rebel_fraction": tune.uniform(0.05, 0.6),
            "escapist_fraction": tune.uniform(0.05, 0.6),
        },
    )

    analysis = tune.run(
        tune.with_parameters(objective, algorithm_name=algorithm_name),
        config=param_space,
        num_samples=10,
        scheduler=scheduler,
        resources_per_trial={"cpu": 1},
    )

    best_trial = analysis.get_best_trial("average_result", mode="min")
    best_params = best_trial.config
    best_result = best_trial.last_result["average_result"]

    with open(f"{algorithm_name}_ray_results.json", "a") as f:
        json.dump({"run_number": "best", "best_params": best_params, "best_objective": best_result}, f)
        f.write("\n")

    return best_params, best_result


if __name__ == "__main__":
    ray.init()
    for algo_name, param_space in ALGORITHMS.items():
        print(f"Running Ray Tune optimization for {algo_name}...")
        run_ray_optimization(algo_name, param_space)
