import math

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class SchafferN2(FloatProblem):
    def __init__(self):
        super(SchafferN2, self).__init__()

        self.obj_labels = ['f(x)']
        self.obj_directions = [self.MINIMIZE]

        self.lower_bound = [-100, -100]
        self.upper_bound = [100, 100]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x, y = solution.variables
        solution.objectives[0] = 0.5 + (math.sin(x**2 - y**2)**2 - 0.5) / (1 + 0.001 * (x**2 + y**2))**2
        return solution

    def name(self) -> str:
        return "Schaffer N2"
