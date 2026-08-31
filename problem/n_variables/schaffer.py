import math
import random

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class SchafferBase(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(SchafferBase, self).__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        pass

    def name(self) -> str:
        pass

class GeneralizedSchafferN7(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(GeneralizedSchafferN7, self).__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-50.0] * number_of_variables
        self.upper_bound = [50.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        d = self.number_of_variables()
        z = x
        s = np.array([z[i]**2 + z[i+1]**2 for i in range(d - 1)])
        inner = np.sum(s + s * (np.sin(50 * (s ** (1/5)))**2)) / (d - 1)
        result = inner**2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Generalized Schaffer N7"


class GeneralizedSchafferN1(FloatProblem):
    """
    Multi-dimensional generalization of Schaffer01:

      f(x) = sum_{i=1}^{n-1} [ 0.5 + ( sin^2(x_i^2 - x_{i+1}^2) - 0.5 ) / (1 + 0.001*(x_i^2 + x_{i+1}^2))^2 ]

    Domain: x_i ∈ [-100, 100] for all i.
    Global optimum: f(x)=0 when all x_i = 0.
    """

    def __init__(self, number_of_variables: int = 10):
        super(GeneralizedSchafferN1, self).__init__()
        self.lower_bound = [-50.0] * number_of_variables
        self.upper_bound = [50.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    # def create_solution(self) -> FloatSolution:
    #     new_solution = FloatSolution(self.lower_bound,
    #                                  self.upper_bound,
    #                                  self.number_of_objectives(),
    #                                  self.number_of_constraints())
    #     new_solution.variables = [random.uniform(lb, ub)
    #                               for lb, ub in zip(self.lower_bound, self.upper_bound)]
    #     return new_solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for i in range(len(solution.variables) - 1):
            x_i = solution.variables[i]
            x_next = solution.variables[i + 1]
            numerator = math.sin(x_i ** 2 - x_next ** 2) ** 2 - 0.5
            denominator = (1 + 0.001 * (x_i ** 2 + x_next ** 2)) ** 2
            total += 0.5 + numerator / denominator
        solution.objectives[0] = total
        return solution

    def name(self) -> str:
        return "Generalized Schaffer N1"


class GeneralizedSchafferN2(FloatProblem):
    """
    Multi-dimensional generalization of Schaffer02:

      f(x) = sum_{i=1}^{n-1} [ 0.5 + ( cos(sin(|x_i^2 - x_{i+1}^2|)) - 0.5 ) / (1+0.001*(x_i^2+x_{i+1}^2))^2 ]

    Domain: x_i ∈ [-100, 100] for all i.
    Note: f(0)=1*(n-1) since each pair yields 0.5+(1-0.5)=1.
    """

    def __init__(self, number_of_variables: int = 10):
        super(GeneralizedSchafferN2, self).__init__()
        self.lower_bound = [-50.0] * number_of_variables
        self.upper_bound = [50.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        n = len(solution.variables)
        for i in range(n - 1):
            x_i = solution.variables[i]
            x_next = solution.variables[i+1]
            numerator = math.cos(math.sin(abs(x_i**2 - x_next**2))) - 0.5
            denominator = (1 + 0.001 * (x_i**2 + x_next**2))**2
            pair_value = 0.5 + numerator/denominator  # f_pair, with f(0,0)=1
            total += pair_value
        # At optimum (all zeros), total == n-1.
        # We shift and reverse:
        transformed = -(total - (n - 1))
        solution.objectives[0] = transformed
        return solution

    def name(self) -> str:
        return "Generalized Schaffer N2"


class GeneralizedSchafferN3(FloatProblem):
    """
    Multi-dimensional generalization of the classical Schaffer F7 pair form
    (note: NOT Schaffer N3 of the N1-N4 numbering, whose optimum is nonzero
    and off the origin):

      f(x) = sum_{i=1}^{n-1} [ (x_i^2+x_{i+1}^2)^0.25 * (1 + sin^2(50*(x_i^2+x_{i+1}^2)^0.1) ) ]

    Domain: x_i ∈ [-50, 50] for all i.
    Global optimum: f(0)=0.
    """

    def __init__(self, number_of_variables: int = 10):
        super(GeneralizedSchafferN3, self).__init__()
        self.lower_bound = [-50.0] * number_of_variables
        self.upper_bound = [50.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        d = self.number_of_variables()
        for i in range(d - 1):
            x_i = solution.variables[i]
            x_next = solution.variables[i+1]
            r_sq = x_i**2 + x_next**2
            # At optimum: if x_i = x_next = 0, then r_sq=0, so term = 0.
            term = (r_sq**0.25) * (1 + math.sin(50*(r_sq**0.1))**2)
            total += term
        solution.objectives[0] = total
        return solution

    def name(self) -> str:
        return "Generalized Schaffer N3"


class GeneralizedSchafferN4(FloatProblem):
    """
    Multi-dimensional generalization of Schaffer04:

      f(x) = sum_{i=1}^{n-1} [ 0.5 + ( cos(sin(|x_i^2-x_{i+1}^2|)) - 0.5 ) / (1+0.001*(x_i^2+x_{i+1}^2)^2)^2 ]

    Domain: x_i ∈ [-100, 100] for all i.
    """

    def __init__(self, number_of_variables: int = 10):
        super(GeneralizedSchafferN4, self).__init__()
        self.lower_bound = [-50.0] * number_of_variables
        self.upper_bound = [50.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    # def create_solution(self) -> FloatSolution:
    #     sol = FloatSolution(self.lower_bound,
    #                         self.upper_bound,
    #                         self.number_of_objectives(),
    #                         self.number_of_constraints())
    #     sol.variables = [random.uniform(lb, ub)
    #                      for lb, ub in zip(self.lower_bound, self.upper_bound)]
    #     return sol

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        n = len(solution.variables)
        for i in range(n - 1):
            x_i = solution.variables[i]
            x_next = solution.variables[i+1]
            numerator = math.cos(math.sin(abs(x_i**2 - x_next**2))) - 0.5
            denominator = (1 + 0.001 * (x_i**2 + x_next**2)**2)**2
            pair_value = 0.5 + numerator/denominator  # f_pair, f(0,0)=1
            total += pair_value
        # At optimum, total = n-1; shift and reverse to get 0.
        transformed = -(total - (n - 1))
        solution.objectives[0] = transformed
        return solution

    def name(self) -> str:
        return "Generalized Schaffer N4"
