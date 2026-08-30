import pytest
from unittest.mock import patch

from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.pso_ga_hybrids.pgchea import PGCHEA
from operator_wrapper.PSO_GA_wrapper import CrossoverWithPsoAttributes, MutationWithPsoAttributes


def _make_pgchea(max_evaluations: int = 20) -> PGCHEA:
    problem = Sphere(3)
    return PGCHEA(
        problem=problem,
        solutions_size=4,
        c1=1.0,
        c2=1.0,
        w=0.5,
        crossover=SBXCrossover(1.0, 5.0),
        mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
        starting_algorithm="PSO",
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )


def test_pgchea_switch_to_ga_sets_current_algorithm() -> None:
    algorithm = _make_pgchea()
    algorithm.create_initial_solutions()
    algorithm.switch_to_ga()
    assert algorithm.current_algorithm == "GA"


def test_pgchea_switch_to_ga_transfers_pso_solutions_to_ga() -> None:
    algorithm = _make_pgchea()
    algorithm.create_initial_solutions()
    pso_positions = [p.variables[:] for p in algorithm.pso.solutions]

    with patch.object(algorithm.ga, "set_solutions", wraps=algorithm.ga.set_solutions) as mock_set:
        algorithm.switch_to_ga()
        mock_set.assert_called_once()
        transferred = mock_set.call_args[0][0]
        actual_positions = [s.variables[:] for s in transferred]
        assert actual_positions == pso_positions


def test_pgchea_ga_uses_pso_aware_operators() -> None:
    algorithm = _make_pgchea()
    assert isinstance(algorithm.ga.crossover_operator, CrossoverWithPsoAttributes)
    assert isinstance(algorithm.ga.mutation_operator, MutationWithPsoAttributes)
