import copy
import numpy as np
import experiment.runner as runner

from jmetal.problem import Sphere
from jmetal.util.termination_criterion import (
    StoppingByEvaluations,
)

from algorithm.basic.single_objective_pso import (
    SingleObjectivePSO,
)


def test_run_experiment_creates_fresh_algorithm_for_each_run(
    monkeypatch,
):
    no_of_runs = 3
    swarm_size = 4
    max_evaluations = 8

    monkeypatch.setattr(
        runner,
        "max_evaluations",
        max_evaluations,
    )

    created_algorithms = []

    def algorithm_factory(problem):
        algorithm = SingleObjectivePSO(
            problem=problem,
            swarm_size=swarm_size,
            w=0.5,
            c1=1.5,
            c2=1.5,
            termination_criterion=StoppingByEvaluations(
                max_evaluations=max_evaluations
            ),
        )

        created_algorithms.append(algorithm)
        return algorithm

    problem = Sphere(3)

    runner.run_experiment(
        algorithm_factory,
        problem,
        runs=no_of_runs,
        interval=swarm_size,
    )

    assert len(created_algorithms) == no_of_runs
    assert len({id(algorithm) for algorithm in created_algorithms}) == no_of_runs
    assert all(algorithm.evaluations == max_evaluations for algorithm in created_algorithms)


def test_run_experiment_is_reproducible_with_same_seeds(
    monkeypatch,
):
    no_of_runs = 3
    swarm_size = 4
    max_evaluations = 8
    run_seeds = [101, 102, 103]

    monkeypatch.setattr(
        runner,
        "max_evaluations",
        max_evaluations,
    )

    def algorithm_factory(problem):
        return SingleObjectivePSO(
            problem=problem,
            swarm_size=swarm_size,
            w=0.5,
            c1=1.5,
            c2=1.5,
            termination_criterion=StoppingByEvaluations(
                max_evaluations=max_evaluations
            ),
        )

    problem = Sphere(3)

    first_result = runner.run_experiment(
        algorithm_factory,
        problem,
        runs=no_of_runs,
        interval=swarm_size,
        run_seeds=run_seeds,
    )

    second_result = runner.run_experiment(
        algorithm_factory,
        problem,
        runs=no_of_runs,
        interval=swarm_size,
        run_seeds=run_seeds,
    )

    np.testing.assert_array_equal(
        first_result[0],
        second_result[0],
    )

    assert first_result[1] == second_result[1]
    assert first_result[2] == second_result[2]

def test_sequential_and_worker_use_seed_consistently(
    monkeypatch,
):
    swarm_size = 4
    max_evaluations = 8
    seed = 123

    monkeypatch.setattr(
        runner,
        "max_evaluations",
        max_evaluations,
    )

    def algorithm_factory(problem):
        return SingleObjectivePSO(
            problem=problem,
            swarm_size=swarm_size,
            w=0.5,
            c1=1.5,
            c2=1.5,
            termination_criterion=StoppingByEvaluations(
                max_evaluations=max_evaluations
            ),
        )

    problem = Sphere(3)

    sequential = runner.run_experiment(
        algorithm_factory,
        problem,
        runs=1,
        interval=swarm_size,
        run_seeds=[seed],
    )

    worker = runner.run_single_instance(
        (
            "PSO",
            copy.deepcopy(problem),
            algorithm_factory,
            1,
            seed,
            max_evaluations,
            swarm_size,
        )
    )

    np.testing.assert_array_equal(
        sequential[0][0],
        worker["fitness_history"],
    )

    assert (
        sequential[0][0][-1]
        == worker["final_fitness"]
    )