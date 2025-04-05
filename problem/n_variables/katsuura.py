import math
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Katsuura(FloatProblem):
    """
    f(x) = (10 / d^2.2) * ∏_{i=1}^{d} (1 + i * A_i(x))^(10 / d^1.2) - (10 / d^2.2)
    gdzie A_i(x) = ∑_{j=1}^{32} |2^j * x_i - round(2^j * x_i)| / (2^j)
    """

    def __init__(self, number_of_variables: int = 2):
        super(Katsuura, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [5.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        d = self._number_of_variables
        prod = 1.0
        for i in range(d):
            temp = 0.0
            for j in range(1, 33):
                temp += abs((2 ** j) * x[i] - round((2 ** j) * x[i])) / (2 ** j)
            prod *= (1 + (i + 1) * temp) ** (10.0 / (d ** 1.2))
        result = (10.0 / (d ** 2.2)) * prod - (10.0 / (d ** 2.2))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Katsuura"


class ExpandedKatsuura(FloatProblem):
    """
    f(x) = (10 / d^2.2) * ∏_{i=1}^{d} (1 + i * A_i(x))^(10 / d^1.2) - (10 / d^2.2)
    gdzie A_i(x) = ∑_{j=1}^{d} |2^j * x_i - round(2^j * x_i)| / (2^j)
    """

    def __init__(self, number_of_variables: int = 2):
        super(ExpandedKatsuura, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [5.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        d = self._number_of_variables
        prod = 1.0
        for i in range(d):
            temp = 0.0
            for j in range(1, d + 1):
                temp += abs((2 ** j) * x[i] - round((2 ** j) * x[i])) / (2 ** j)
            prod *= (1 + (i + 1) * temp) ** (10.0 / (d ** 1.2))
        result = (10.0 / (d ** 2.2)) * prod - (10.0 / (d ** 2.2))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Expanded Katsuura"
