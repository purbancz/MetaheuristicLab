import math

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Bird(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(Bird, self).__init__()
        self.lower_bound = [-2*math.pi] * number_of_variables
        self.upper_bound = [2*math.pi] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        penalty = sum(xi ** 2 for xi in x if abs(xi) > 2 * math.pi)
        f = penalty if penalty else (
            math.sin(x[0]) * math.exp((1 - math.cos(x[1])) ** 2) +
            math.cos(x[1]) * math.exp((1 - math.sin(x[0])) ** 2) +
            (x[0] - x[1]) ** 2)
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Bird"