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