import pytest

from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere

from experiment import factories


class CountingSphere(Sphere):
    def __init__(self, number_of_variables: int):
        super().__init__(number_of_variables)
        self.evaluation_count = 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        self.evaluation_count += 1
        return super().evaluate(solution)


ACTIVE_ROLE_BASED_FACTORIES = [
    pytest.param(
        factories.factory_RebelRejectorPSO,
        id="RebelRejectorPSO",
    ),
    pytest.param(
        factories.factory_ContrarianDefeatistPSO,
        id="ContrarianDefeatistPSO",
    ),
    pytest.param(
        factories.factory_EschewerEscapistPSO,
        id="EschewerEscapistPSO",
    ),
    pytest.param(
        factories.factory_HybridFullDisjointPSO,
        id="HybridFullDisjointPSO",
    ),
    pytest.param(
        factories.factory_HybridPartialDisjointPSO,
        id="HybridPartialDisjointPSO",
    ),
    pytest.param(
        factories.factory_HybridAdditivePSO,
        id="HybridAdditivePSO",
    ),
]


@pytest.mark.parametrize(
    "algorithm_factory",
    ACTIVE_ROLE_BASED_FACTORIES,
)
def test_active_role_based_pso_reported_evaluations_match_actual_calls(
    algorithm_factory,
    monkeypatch,
):
    swarm_size = 4
    max_evaluations = 12

    monkeypatch.setattr(
        factories,
        "G_SOLUTIONS_SIZE",
        swarm_size,
    )
    monkeypatch.setattr(
        factories,
        "G_MAX_EVALUATIONS",
        max_evaluations,
    )

    problem = CountingSphere(3)
    algorithm = algorithm_factory(problem)

    algorithm.run()

    assert algorithm.evaluations == problem.evaluation_count
    assert problem.evaluation_count == max_evaluations


@pytest.mark.parametrize(
    "algorithm_factory",
    ACTIVE_ROLE_BASED_FACTORIES,
)
def test_active_role_based_pso_does_not_exceed_evaluation_budget(
    algorithm_factory,
    monkeypatch,
):
    swarm_size = 4
    max_evaluations = 10

    monkeypatch.setattr(
        factories,
        "G_SOLUTIONS_SIZE",
        swarm_size,
    )
    monkeypatch.setattr(
        factories,
        "G_MAX_EVALUATIONS",
        max_evaluations,
    )

    problem = CountingSphere(3)
    algorithm = algorithm_factory(problem)

    algorithm.run()

    assert algorithm.evaluations == problem.evaluation_count
    assert problem.evaluation_count <= max_evaluations