import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class DixonPrice(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(DixonPrice, self).__init__()
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

        result = (x[0] - 1) ** 2

        for i in range(1, len(x)):
            result += (i + 1) * (2 * (x[i] ** 2) - x[i - 1]) ** 2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Dixon-Price"





class GeneralizedDixonPriceRosenbrock(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(GeneralizedDixonPriceRosenbrock, self).__init__()
        self.lower_bound = [-30.0] * number_of_variables
        self.upper_bound = [30.0] * number_of_variables

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
        result = 0.0
        for i in range(len(x) - 1):
            term1 = 100 * (x[i + 1] - x[i] ** 2) ** 8
            term2 = (x[i] - 1) ** 8
            result += term1 + term2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Generalized Dixon-Price-Rosenbrock Function"
