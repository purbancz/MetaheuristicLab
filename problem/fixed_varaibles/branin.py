import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class BraninRCOC(FloatProblem):
    def __init__(self):
        super(BraninRCOC, self).__init__()
        self.lower_bound = [-5, 0]
        self.upper_bound = [10, 15]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x1, x2 = solution.variables
        a = 1.0
        b = 5.1 / (4 * math.pi ** 2)
        c = 5 / math.pi
        r = 6
        s = 10
        t = 1 / (8 * math.pi)
        result = a * (x2 - b * x1 ** 2 + c * x1 - r) ** 2 + s * (1 - t) * math.cos(x1) + s
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return 'Branin RCOC'
