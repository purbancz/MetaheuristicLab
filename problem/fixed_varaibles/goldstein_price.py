from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class GoldsteinPrice(FloatProblem):
    def __init__(self):
        super(GoldsteinPrice, self).__init__()
        self.lower_bound = [-2, -2]
        self.upper_bound = [2, 2]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x, y = solution.variables
        part1 = 1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x ** 2 - 14 * y + 6 * x * y + 3 * y ** 2)
        part2 = 30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x ** 2 + 48 * y - 36 * x * y + 27 * y ** 2)
        result = part1 * part2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Goldstein and Price"
