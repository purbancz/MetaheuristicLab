# algorithm/tests/sota/test_lshade.py

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