import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Plateau(FloatProblem):
    """
    A simple unconstrained benchmark function with a plateau-shaped optimum.

    The function is defined as:
      f(x) = max(0, ||x||^2 - c)

    where c is a constant that defines the radius of the plateau.
    For any x with sum(x_i^2) <= c, the function returns 0 (the optimum),
    and for x outside that ball, the function increases quadratically.
    """

    def __init__(self, number_of_variables: int = 10, c: float = 1.0):
        super(Plateau, self).__init__()
        self.c = c

        self.lower_bound = [-5.0 for _ in range(number_of_variables)]
        self.upper_bound = [5.0 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def create_solution(self) -> FloatSolution:
        new_solution = FloatSolution(self.lower_bound, self.upper_bound, self.number_of_objectives(),
                                     self.number_of_constraints())
        new_solution.variables = [
            np.random.uniform(self.lower_bound[i], self.upper_bound[i])
            for i in range(self.number_of_variables())
        ]
        return new_solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        norm_sq = sum(x * x for x in solution.variables)
        solution.objectives[0] = max(0.0, norm_sq - self.c)
        return solution

    def name(self) -> str:
        return "Plateau"
