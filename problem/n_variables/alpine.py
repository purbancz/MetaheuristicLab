import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class AlpineN1(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(AlpineN1, self).__init__()
        self.lower_bound = [-10.0] * number_of_variables
        self.upper_bound = [10.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        result = sum(abs(xi * math.sin(xi) + 0.1 * xi) for xi in x)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Alpine N1"

class AlpineN2(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(AlpineN2, self).__init__()
        self.lower_bound = [-10.0] * number_of_variables
        self.upper_bound = [10.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        result = math.prod(math.sin(xi) * math.sqrt(abs(xi)) for xi in x)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Alpine N2"
