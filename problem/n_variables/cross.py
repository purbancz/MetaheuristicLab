import math

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class GeneralizedCrossInTray(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(GeneralizedCrossInTray, self).__init__()
        self.lower_bound = [-10.0] * number_of_variables
        self.upper_bound = [10.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        penalty = sum(xi ** 2 for xi in x if abs(xi) > 10)
        f = (penalty if penalty
             else -0.0001 * (abs(math.sin(x[0]) * math.sin(x[1]) *
                                  math.exp(abs(100 - (math.sqrt(x[0]**2 + x[1]**2) / math.pi)))) + 1) ** 0.1)
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Cross In Tray"


class Cross(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(Cross, self).__init__()
        self.lower_bound = [-10.0] * number_of_variables
        self.upper_bound = [10.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        penalty = sum(xi ** 2 for xi in x if abs(xi) > 10)
        f = (penalty if penalty
             else (abs(math.sin(x[0]) * math.sin(x[1]) *
                       math.exp(abs(100 - (math.sqrt(x[0]**2 + x[1]**2) / math.pi)))) + 1) ** -0.1)
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Cross"


class CrossLeggedTable(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(CrossLeggedTable, self).__init__()
        self.lower_bound = [-10.0] * number_of_variables
        self.upper_bound = [10.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        penalty = sum(xi ** 2 for xi in x if abs(xi) > 10)
        f = (penalty if penalty
             else - (abs(math.sin(x[0]) * math.sin(x[1]) *
                         math.exp(abs(100 - (math.sqrt(x[0]**2 + x[1]**2) / math.pi)))) + 1) ** -0.1)
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Cross Legged Table"