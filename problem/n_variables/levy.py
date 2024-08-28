import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Levy(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Levy, self).__init__()

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
        w = [1 + (xi - 1) / 4 for xi in x]
        term1 = (math.sin(math.pi * w[0])) ** 2
        term3 = (w[-1] - 1) ** 2 * (1 + (math.sin(2 * math.pi * w[-1])) ** 2)
        sum_terms = sum([(wi - 1) ** 2 * (1 + 10 * (math.sin(math.pi * wi + 1)) ** 2) for wi in w[:-1]])

        solution.objectives[0] = term1 + sum_terms + term3
        return solution

    def name(self) -> str:
        return 'Levy'
