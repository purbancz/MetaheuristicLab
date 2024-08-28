import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Shubert(FloatProblem):
    def __init__(self):
        super(Shubert, self).__init__()
        self.lower_bound = [-10, -10]
        self.upper_bound = [10, 10]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x, y = solution.variables
        sum_x = sum(i * math.cos((i + 1) * x + i) for i in range(1, 6))
        sum_y = sum(i * math.cos((i + 1) * y + i) for i in range(1, 6))
        result = sum_x * sum_y
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return 'Shubert'
