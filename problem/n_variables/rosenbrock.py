import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Rosenbrock(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Rosenbrock, self).__init__()
        self.lower_bound = [-5.0] * number_of_variables
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
        solution.objectives[0] = sum(100 * (x[i+1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))
        return solution

    def name(self) -> str:
        return 'Rosenbrock'
