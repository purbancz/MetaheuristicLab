import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class HolderTable(FloatProblem):
    def __init__(self):
        super(HolderTable, self).__init__()
        self.lower_bound = [-10.0, -10.0]
        self.upper_bound = [10.0, 10.0]

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
        term = math.sin(x) * math.cos(y) * math.exp(
            math.fabs(1.0 - (math.sqrt(x**2 + y**2) / math.pi))
        )
        result = -math.fabs(term)

        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Holder-Table"
