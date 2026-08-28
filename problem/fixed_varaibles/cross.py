import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class CrownedCross(FloatProblem):
    def __init__(self):
        super(CrownedCross, self).__init__()
        self.lower_bound = [-10.0, -10.0]
        self.upper_bound = [10.0, 10.0]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        penalty = sum(math.exp(abs(xi)) for xi in x if abs(xi) > 10)
        f = (penalty if penalty
             else 0.0001 * (abs(math.sin(x[0]) * math.sin(x[1]) *
                                 math.exp(abs(100 - (math.sqrt(x[0]**2 + x[1]**2) / math.pi)))) + 1) ** 0.1)
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Crowned Cross"

class CrossInTray(FloatProblem):
    def __init__(self):
        super(CrossInTray, self).__init__()
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
        x, y = solution.variables[0], solution.variables[1]

        term = math.fabs(
            math.sin(x) * math.sin(y) *
            math.exp(math.fabs(100.0 - (math.sqrt(x**2 + y**2) / math.pi)))
        ) + 1.0

        result = -0.0001 * (term ** 0.1)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Cross-in-Tray"
