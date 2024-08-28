import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Easom(FloatProblem):
    def __init__(self):
        super(Easom, self).__init__()
        self.lower_bound = [-100, -100]
        self.upper_bound = [100, 100]

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
        result = -math.cos(x) * math.cos(y) * math.exp(-(x - math.pi) ** 2 - (y - math.pi) ** 2)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return 'Easom'
