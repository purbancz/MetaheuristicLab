import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Ackley(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Ackley, self).__init__()

        self.lower_bound = [-32.768] * number_of_variables
        self.upper_bound = [32.768] * number_of_variables

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
        a, b, c = 20, 0.2, 2 * math.pi

        sum1 = sum([xi ** 2 for xi in x]) / len(x)
        sum2 = sum([math.cos(c * xi) for xi in x]) / len(x)
        result = -a * math.exp(-b * math.sqrt(sum1)) - math.exp(sum2) + a + math.e

        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return 'Ackley'
