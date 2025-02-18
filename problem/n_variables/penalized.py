import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class GeneralizedPenalizedN1(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(GeneralizedPenalizedN1, self).__init__()
        self.lower_bound = [-50.0] * number_of_variables
        self.upper_bound = [50.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.a = 10
        self.k = 100
        self.m = 4

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def u(self, x: float) -> float:
        """
        Penalty function:
          u(x,a,k,m)= k*(x-a)^m  if x > a
                      0            if -a <= x <= a
                      k*(-x-a)^m   if x < -a
        """
        if x > self.a:
            return self.k * ((x - self.a) ** self.m)
        elif x < -self.a:
            return self.k * ((-x - self.a) ** self.m)
        else:
            return 0.0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        n = len(x)

        y = [1 + (xi + 1) / 4.0 for xi in x]

        term = 10 * (math.sin(math.pi * y[0])) ** 2
        for i in range(n - 1):
            term += (y[i] - 1) ** 2 * (1 + 10 * (math.sin(math.pi * y[i + 1])) ** 2)
        term += (y[n - 1] - 1) ** 2

        inner = (math.pi / n) * term

        penalty = sum(self.u(xi) for xi in x)

        result = inner + penalty

        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Generalized Penalized Function N1"
