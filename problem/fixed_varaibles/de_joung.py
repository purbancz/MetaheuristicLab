from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class DeJoung(FloatProblem):
    def __init__(self):
        super(DeJoung, self).__init__()
        self.lower_bound = [-5.12, -5.12, -5.12]
        self.upper_bound = [5.12, 5.12, 5.12]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        result = sum(xi ** 2 for xi in solution.variables)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return 'De Joung'
