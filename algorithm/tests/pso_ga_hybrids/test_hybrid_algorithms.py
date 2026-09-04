import pytest
from unittest.mock import patch

from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.pso_ga_hybrids.pgchea import PGCHEA
from algorithm.pso_ga_hybrids.pgphea import PGPHEA
from algorithm.pso_ga_hybrids.pgshea import PGSHEA
from algorithm.tests.counting_problem import CountingProblem
from operator_wrapper.PSO_GA_wrapper import CrossoverWithPsoAttributes, MutationWithPsoAttributes


def _operators(problem):
    return {
        "crossover": SBXCrossover(1.0, 5.0),
        "mutation": PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
    }


def _make_pgchea(problem=None, max_evaluations: int = 20, starting_algorithm: str = "PSO") -> PGCHEA:
    problem = problem or Sphere(3)
    return PGCHEA(
        problem=problem,
        solutions_size=4,
        c1=1.0,
        c2=1.0,
        w=0.5,
        starting_algorithm=starting_algorithm,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        **_operators(problem),
    )


def _initialize(algorithm):
    """Mimic Algorithm.run()'s pre-loop: create, evaluate, init_progress."""
    algorithm.solutions = algorithm.create_initial_solutions()
    algorithm.solutions = algorithm.evaluate(algorithm.solutions)
    algorithm.init_progress()


def test_pgchea_switch_to_ga_sets_current_algorithm() -> None:
    algorithm = _make_pgchea()
    _initialize(algorithm)
    algorithm.switch_to_ga()
    assert algorithm.current_algorithm == "GA"


def test_pgchea_switch_to_ga_transfers_pso_solutions_to_ga() -> None:
    algorithm = _make_pgchea()
    _initialize(algorithm)
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


def test_pgchea_islands_are_seeded_with_evaluated_solutions() -> None:
    algorithm = _make_pgchea()
    _initialize(algorithm)

    # Sphere is strictly positive away from the origin; a 0.0 objective here
    # would be jmetal's never-evaluated placeholder.
    for island in (algorithm.pso.solutions, algorithm.ga.solutions):
        assert island
        for solution in island:
            assert solution.objectives[0] > 0.0
    for particle in algorithm.pso.solutions:
        assert particle.attributes["best_objective"] > 0.0
    assert algorithm.pso.best_global.objectives[0] > 0.0
    assert algorithm.best_global.objectives[0] > 0.0


@pytest.mark.parametrize("starting_algorithm", ["PSO", "GA"])
def test_pgchea_reports_real_optimum_and_exact_evaluation_count(starting_algorithm) -> None:
    problem = CountingProblem(3)
    algorithm = _make_pgchea(problem=problem, starting_algorithm=starting_algorithm)

    algorithm.run()

    # The reported optimum must be a genuinely evaluated value, never the
    # 0.0 placeholder of an unevaluated solution.
    assert algorithm.result().objectives[0] > 0.0
    assert algorithm.evaluations == problem.evaluation_count == 20


def test_pgshea_reports_real_optimum_and_exact_evaluation_count() -> None:
    problem = CountingProblem(3)
    algorithm = PGSHEA(
        problem=problem,
        solutions_size=4,
        c1=1.0,
        c2=1.0,
        w=0.5,
        swap_interval=1,
        starting_algorithm="GA",
        termination_criterion=StoppingByEvaluations(max_evaluations=20),
        **_operators(problem),
    )

    algorithm.run()

    assert algorithm.result().objectives[0] > 0.0
    assert algorithm.evaluations == problem.evaluation_count == 20


def test_pgphea_reports_real_optimum_and_exact_evaluation_count() -> None:
    problem = CountingProblem(3)
    algorithm = PGPHEA(
        problem=problem,
        solutions_size=4,
        c1=1.0,
        c2=1.0,
        w=0.5,
        exchange_interval=1,
        exchange_number=1,
        termination_criterion=StoppingByEvaluations(max_evaluations=20),
        **_operators(problem),
    )

    algorithm.run()

    # Before the fix PGPHEA spent 1.5x the counted budget (pso swarm + a
    # full-size GA offspring per step) and could report placeholder 0.0.
    assert algorithm.result().objectives[0] > 0.0
    assert algorithm.evaluations == problem.evaluation_count == 20
