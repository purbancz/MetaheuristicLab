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