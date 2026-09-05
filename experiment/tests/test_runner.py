import copy
import random

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
        first_result["data"],
        second_result["data"],
    )
    np.testing.assert_array_equal(
        first_result["final_fitness"],
        second_result["final_fitness"],
    )

    assert first_result["seeds"] == second_result["seeds"]
    assert first_result["avg_fitness"] == second_result["avg_fitness"]
    assert first_result["std_dev"] == second_result["std_dev"]

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
        sequential["data"][0],
        worker["fitness_history"],
    )

    assert (
        sequential["final_fitness"][0]
        == worker["final_fitness"]
    )


def test_multi_runner_stamps_problem_own_dimension(
    monkeypatch,
    tmp_path,
):
    import h5py

    swarm_size = 4
    max_evaluations = 8

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

    class SerialPool:
        def __init__(self, processes=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, iterable):
            return [func(item) for item in iterable]

    monkeypatch.setattr(runner, "Pool", SerialPool)
    monkeypatch.setattr(runner, "problems", [Sphere(3)])
    monkeypatch.setattr(runner, "algorithms", {"PSO": algorithm_factory})
    monkeypatch.setattr(runner, "no_of_runs", 2)
    monkeypatch.setattr(runner, "max_evaluations", max_evaluations)
    monkeypatch.setattr(runner, "frequency", swarm_size)
    # Deliberately mismatched campaign dimension: files must carry the
    # problem's own dimension (3), not this global.
    monkeypatch.setattr(runner, "number_of_variables", 1000)
    monkeypatch.setattr(runner, "results_dir", str(tmp_path))

    runner.run_all_experiments_multi(num_parallel_workers=1)

    h5_path = tmp_path / "dim1000_runs2" / "Sphere_dim3_runs2_PSO.h5"
    assert h5_path.exists()

    with h5py.File(h5_path, "r") as f:
        assert f.attrs["n_vars"] == 3
        assert f["Sphere"].attrs["n_vars"] == 3


def test_worker_output_depends_only_on_its_seed_not_inherited_rng_state(
    monkeypatch,
):
    """Fork-duplication canary.

    Forked pool workers inherit the parent's RNG state; without per-run
    seeding inside the worker, runs on different workers can be identical
    duplicates. The invariant that prevents it: run_single_instance's output
    is a function of its seed argument ONLY, never of whatever RNG state the
    worker process happens to start with.
    """
    swarm_size = 4
    max_evaluations = 8

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

    def run_with_inherited_state(seed, inherited_state_seed):
        # Simulate the RNG state a worker inherits before it handles a run.
        random.seed(inherited_state_seed)
        np.random.seed(inherited_state_seed)
        return runner.run_single_instance(
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

    # Same seed, different inherited states: identical results.
    first = run_with_inherited_state(seed=123, inherited_state_seed=1)
    second = run_with_inherited_state(seed=123, inherited_state_seed=2)
    np.testing.assert_array_equal(
        first["fitness_history"],
        second["fitness_history"],
    )
    assert first["final_fitness"] == second["final_fitness"]

    # Different seeds, same inherited state: distinct results. (If the
    # per-run seeding inside the worker is ever removed, results become a
    # function of the inherited state and this assertion fires.)
    third = run_with_inherited_state(seed=456, inherited_state_seed=1)
    assert not np.array_equal(
        np.asarray(first["fitness_history"]),
        np.asarray(third["fitness_history"]),
    )


def test_h5_manifest_records_full_provenance(
    monkeypatch,
    tmp_path,
):
    import h5py

    swarm_size = 4
    max_evaluations = 8

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

    class SerialPool:
        def __init__(self, processes=None):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, iterable):
            return [func(item) for item in iterable]

    monkeypatch.setattr(runner, "Pool", SerialPool)
    monkeypatch.setattr(runner, "problems", [Sphere(3)])
    monkeypatch.setattr(runner, "algorithms", {"PSO": algorithm_factory})
    monkeypatch.setattr(runner, "no_of_runs", 2)
    monkeypatch.setattr(runner, "max_evaluations", max_evaluations)
    monkeypatch.setattr(runner, "frequency", swarm_size)
    monkeypatch.setattr(runner, "solutions_size", swarm_size)
    monkeypatch.setattr(runner, "number_of_variables", 1000)
    monkeypatch.setattr(runner, "results_dir", str(tmp_path))

    runner.run_all_experiments_multi(num_parallel_workers=1)

    h5_path = tmp_path / "dim1000_runs2" / "Sphere_dim3_runs2_PSO.h5"
    with h5py.File(h5_path, "r") as f:
        # Campaign configuration.
        assert f.attrs["max_evaluations"] == max_evaluations
        assert f.attrs["solutions_size"] == swarm_size
        assert f.attrs["evaluations_per_snapshot"] == swarm_size
        assert f.attrs["base_seed"] == runner.BASE_SEED
        assert f.attrs["benchmark_base_seed"] == runner.BENCHMARK_BASE_SEED
        # Code identity.
        assert len(f.attrs["git_commit"]) >= 7
        # Problem identity.
        assert f["Sphere"].attrs["problem_class"].endswith(".Sphere")
        # Algorithm parameters, verbatim from the factory source.
        source = f["Sphere"]["PSO"].attrs["factory_source"]
        assert "SingleObjectivePSO" in source
        assert "c1=1.5" in source
