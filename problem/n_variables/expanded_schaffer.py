import math

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class ExpandedShaffer(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(ExpandedShaffer, self).__init__()
        self._number_of_variables = number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-100 for _ in range(number_of_variables)]
        self.upper_bound = [100 for _ in range(number_of_variables)]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def g(self, x, y):
        up = math.pow(math.sin(math.sqrt(x ** 2 + y ** 2)), 2) - 0.5
        down = math.pow(1 + 0.001 * (x ** 2 + y ** 2), 2)
        return up / down

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        s = self.number_of_variables() / 2.0
        for i in range(self.number_of_variables()):
            s += self.g(x[i], x[(i + 1) % self.number_of_variables()])
        solution.objectives[0] = s
        return solution

    def name(self) -> str:
        return 'Expanded Shaffer'