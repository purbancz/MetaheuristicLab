from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere


class CountingProblem(Sphere):
    """Sphere that counts every call to evaluate()."""

    def __init__(self, number_of_variables: int = 3):
        super().__init__(number_of_variables)
        self.evaluation_count = 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        self.evaluation_count += 1
        return super().evaluate(solution)
