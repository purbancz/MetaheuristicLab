from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Discus(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(Discus, self).__init__()
        self._number_of_variables = number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-100 for _ in range(number_of_variables)]
        self.upper_bound = [100 for _ in range(number_of_variables)]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        s = 0
        for i in range(1, self.number_of_variables()):
            s += x[i] ** 2
        solution.objectives[0] = s + (10 ** 6) * (x[0] ** 2)
        return solution

    def name(self) -> str:
        return 'Discus'