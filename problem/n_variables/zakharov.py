from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Zakharov(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(Zakharov, self).__init__()
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

        sum1 = sum(xi ** 2 for xi in x)
        sum2 = sum(0.5 * i * xi for i, xi in enumerate(x, start=1))

        solution.objectives[0] = sum1 + sum2 ** 2 + sum2 ** 4
        return solution

    def name(self) -> str:
        return "Zakharov"
