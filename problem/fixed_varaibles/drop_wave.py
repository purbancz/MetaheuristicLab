import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class DropWave(FloatProblem):
    def __init__(self):
        super(DropWave, self).__init__()
        self.lower_bound = [-5.12, -5.12]
        self.upper_bound = [5.12, 5.12]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x, y)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return 2

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x, y = solution.variables
        r = math.sqrt(x**2 + y**2)

        numerator = 1.0 + math.cos(12.0 * r)
        denominator = 0.5 * (x**2 + y**2) + 2.0
        result = -(numerator / denominator)

        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Drop-Wave"
