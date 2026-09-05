# algorithm/tests/sota/test_lshade.py

import pytest
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.sota.lshade import LSHADE


class CountingSphere(Sphere):
    def __init__(self, number_of_variables: int):
        super().__init__(number_of_variables)
        self.evaluation_count = 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        self.evaluation_count += 1
        return super().evaluate(solution)


def test_lshade_reported_evaluations_match_actual_objective_calls():
    problem = CountingSphere(3)

    algorithm = LSHADE(
        problem=problem,
        initial_population_size=8,
        memory_size=4,
        p_best_rate=0.25,
        archive_size_rate=1.0,
        termination_criterion=StoppingByEvaluations(
            max_evaluations=20
        ),
    )

    algorithm.run()

    assert algorithm.evaluations == problem.evaluation_count

def test_lshade_does_not_exceed_evaluation_budget():
    problem = CountingSphere(3)
    max_evaluations = 20

    algorithm = LSHADE(
        problem=problem,
        initial_population_size=8,
        memory_size=4,
        p_best_rate=0.25,
        archive_size_rate=1.0,
        termination_criterion=StoppingByEvaluations(
            max_evaluations=max_evaluations
        ),
    )

    algorithm.run()

    assert problem.evaluation_count == max_evaluations
    assert algorithm.evaluations == max_evaluations


def test_lshade_donor_selection_respects_canonical_exclusions():
    problem = Sphere(3)

    algorithm = LSHADE(
        problem=problem,
        initial_population_size=5,
        memory_size=4,
        p_best_rate=0.25,
        archive_size_rate=1.0,
        termination_criterion=StoppingByEvaluations(max_evaluations=20),
    )

    population = [problem.create_solution() for _ in range(5)]
    algorithm.archive = [problem.create_solution() for _ in range(3)]
    population_ids = {id(p) for p in population}

    for target_index in range(len(population)):
        target = population[target_index]
        for _ in range(200):
            r1, r2 = algorithm._select_donors(population, target_index)
            # r1 comes from the population and is never the target.
            assert id(r1) in population_ids
            assert r1 is not target
            # r2 comes from population + archive, never the target nor r1.
            assert r2 is not target
            assert r2 is not r1


def _make_lshade(pop_size=4):
    return LSHADE(
        problem=Sphere(3),
        initial_population_size=pop_size,
        memory_size=2,
        p_best_rate=0.5,
        archive_size_rate=1.0,
        termination_criterion=StoppingByEvaluations(max_evaluations=100),
    )


def _evaluated_solution(problem, objective):
    s = problem.create_solution()
    s.objectives[0] = objective
    return s


def test_lshade_cr_memory_uses_weighted_lehmer_mean():
    algorithm = _make_lshade()
    algorithm.evaluations = 0
    problem = algorithm.problem

    parents = [_evaluated_solution(problem, 10.0), _evaluated_solution(problem, 5.0)]
    offspring = [
        (_evaluated_solution(problem, 6.0), 0.2, 0.5),  # improvement 4
        (_evaluated_solution(problem, 4.0), 0.4, 0.3),  # improvement 1
    ]

    algorithm.replacement(parents, offspring)

    # weights = [0.8, 0.2]; Lehmer mean = sum(w*x^2) / sum(w*x)
    expected_cr = (0.8 * 0.2 ** 2 + 0.2 * 0.4 ** 2) / (0.8 * 0.2 + 0.2 * 0.4)
    expected_f = (0.8 * 0.5 ** 2 + 0.2 * 0.3 ** 2) / (0.8 * 0.5 + 0.2 * 0.3)
    assert algorithm.memory_cr[0] == pytest.approx(expected_cr)
    assert algorithm.memory_f[0] == pytest.approx(expected_f)


def test_lshade_cr_terminal_value_locks_cr_to_zero():
    algorithm = _make_lshade()
    algorithm.evaluations = 0
    problem = algorithm.problem

    # All successful CRs are zero -> the memory slot becomes terminal.
    parents = [_evaluated_solution(problem, 10.0)]
    offspring = [(_evaluated_solution(problem, 6.0), 0.0, 0.5)]
    algorithm.replacement(parents, offspring)
    assert algorithm.memory_cr[0] is None

    # A terminal slot stays terminal even after nonzero-CR successes.
    algorithm.memory_pos = 0
    parents = [_evaluated_solution(problem, 10.0)]
    offspring = [(_evaluated_solution(problem, 6.0), 0.7, 0.5)]
    algorithm.replacement(parents, offspring)
    assert algorithm.memory_cr[0] is None

    # Reproduction from an all-terminal memory generates CR = 0 only.
    algorithm.memory_cr = [None] * algorithm.memory_size
    algorithm.solutions = [_evaluated_solution(problem, float(i)) for i in range(4)]
    reproduction = algorithm.reproduction(algorithm.solutions, offspring_count=4)
    assert all(cr == 0.0 for _, cr, _ in reproduction)


def test_lshade_archive_limit_tracks_shrinking_population():
    problem = Sphere(3)

    algorithm = LSHADE(
        problem=problem,
        initial_population_size=20,
        memory_size=4,
        p_best_rate=0.25,
        archive_size_rate=1.0,
        termination_criterion=StoppingByEvaluations(max_evaluations=200),
    )

    algorithm.run()

    # Linear population size reduction must have shrunk the population, and
    # the archive limit follows the CURRENT population, not the initial one.
    assert algorithm.population_size < algorithm.initial_population_size
    limit = int(round(algorithm.archive_size_rate * algorithm.population_size))
    assert len(algorithm.archive) <= limit