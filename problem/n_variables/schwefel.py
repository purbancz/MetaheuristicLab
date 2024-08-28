import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Schwefel(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Schwefel, self).__init__()
        self.lower_bound = [-500.0] * number_of_variables
        self.upper_bound = [500.0] * number_of_variables

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
        solution.objectives[0] = 418.9829 * len(x) - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)
        return solution

    def name(self) -> str:
        return 'Schwefel'
