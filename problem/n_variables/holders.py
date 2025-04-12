import math

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class TestTubeHolder(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(TestTubeHolder, self).__init__()
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
        penalty = sum(xi ** 2 for xi in x[:2] if xi < -10 or xi > 10)
        f = (penalty if penalty
             else -4 * abs(math.sin(x[0]) * math.cos(x[1]) *
                           math.exp(abs(math.cos((x[0] ** 2 + x[1] ** 2) / 200.0)))))
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Test Tube Holder"


class GeneralizedHolderTable(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(GeneralizedHolderTable, self).__init__()
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
        f = penalty if penalty else -abs(math.cos(x[0]) * math.cos(x[1]) *
                                          math.exp(abs(1 - (math.sqrt(x[0] ** 2 + x[1] ** 2) / math.pi))))
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Holder Table"




class CarromTable(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(CarromTable, self).__init__()
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
             else - (1.0 / 30.0) * (math.cos(x[0]) * math.cos(x[1]) *
                                    math.exp(abs(1 - (math.sqrt(x[0]**2 + x[1]**2) / math.pi)))) ** 2)
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Carrom Table"


class PenHolder(FloatProblem):
    def __init__(self, number_of_variables=2):
        super(PenHolder, self).__init__()
        self.lower_bound = [-11.0] * number_of_variables
        self.upper_bound = [11.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        penalty = sum(xi ** 2 for xi in x if abs(xi) > 11)
        if penalty:
            f = penalty
        else:
            inner = abs(math.cos(x[0]) * math.cos(x[1]) *
                        math.exp(abs(1 - (math.sqrt(x[0]**2 + x[1]**2) / math.pi))))
            f = -math.exp(- (inner ** -1))
        solution.objectives[0] = f
        return solution

    def name(self) -> str:
        return "Pen Holder"