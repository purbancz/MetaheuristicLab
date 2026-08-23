from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.sota.cma_es import CMAES


class CountingSphere(Sphere):
    def __init__(self, number_of_variables: int):
        super().__init__(number_of_variables)
        self.evaluation_count = 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        self.evaluation_count += 1
        return super().evaluate(solution)


def test_cma_es_reported_evaluations_match_actual_objective_calls():
    problem = CountingSphere(3)

    algorithm = CMAES(
        problem=problem,
        mu=2,
        lambda_=4,
        termination_criterion=StoppingByEvaluations(
            max_evaluations=20
        ),
    )

    algorithm.run()

    assert algorithm.evaluations == problem.evaluation_count

def test_cma_es_initial_population_is_evaluated_once():
    problem = CountingSphere(3)

    algorithm = CMAES(
        problem=problem,
        mu=2,
        lambda_=4,
        termination_criterion=StoppingByEvaluations(
            max_evaluations=4
        ),
    )

    algorithm.run()

    assert problem.evaluation_count == 4
    assert algorithm.evaluations == 4
    assert len(algorithm.solutions) == 4
    assert algorithm.best_solution_so_far is not None