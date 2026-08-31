import math

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class LennardJonesMinimumEnergyCluster(FloatProblem):
    """
    Lennard-Jones cluster potential in reduced (r_min-normalized) units.

    Variables encode atom coordinates as (x1,y1,z1, x2,y2,z2, ...); with n
    variables the problem uses n // 3 atoms and IGNORES the n % 3 leftover
    variables (dead dimensions when n is not divisible by 3).

    f(x) = 12.7120622568 + sum_{i<j} [ 1/d_ij^2 - 2/d_ij ],  d_ij = r_ij^6,

    i.e. the pairwise potential r^-12 - 2*r^-6 with pair minimum -1 at
    distance 1. The additive constant is |LJ6 global minimum| (12.712062...),
    chosen so the 6-atom (18-variable) instance has optimum 0; for any other
    atom count it is only a constant shift of the landscape.
    """

    def __init__(self, number_of_variables: int = 10):
        super(LennardJonesMinimumEnergyCluster, self).__init__()
        self._number_of_variables = number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-80 for _ in range(number_of_variables)]
        self.upper_bound = [80 for _ in range(number_of_variables)]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def d(self, i, j, variables):
        sum_k = 0
        for k in range(3):
            sum_k += (variables[3 * i + k] - variables[3 * j + k]) ** 2
        return sum_k ** 3

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        s = 12.7120622568
        num_points = self.number_of_variables() // 3
        epsilon = 1e-6  # anti ZeroDivisionError
        for i in range(num_points - 1):
            sum_j = 0
            for j in range(i + 1, num_points):
                d_tmp = self.d(i, j, x)
                if d_tmp < epsilon:
                    d_tmp = epsilon
                sum_j += (1 / (d_tmp ** 2)) - (2 / d_tmp)
            s += sum_j
        solution.objectives[0] = s
        return solution

    def name(self) -> str:
        return 'Lennard-Jones Minimum Energy Cluster'


# class LennardJones(FloatProblem):
#     def __init__(self, atoms: int, dims: int, eps: float = 1.0, sigma: float = 1.0, *, delay=None):
#         super(LennardJones, self).__init__()
#         self.atoms = atoms
#         self.dims = dims
#         self.eps = eps
#         self.sigma = sigma
#         self.delay = delay
#
#         self._number_of_variables = atoms * dims
#
#         bound = (2 ** (-1 / 6)) * sigma / math.sqrt(np.pi * 3 * atoms)
#         self.lower_bound = [-bound] * self._number_of_variables
#         self.upper_bound = [bound] * self._number_of_variables
#
#         self.obj_directions = [self.MINIMIZE]
#         self.obj_labels = ["f(x)"]
#
#         FloatSolution.lower_bound = self.lower_bound
#         FloatSolution.upper_bound = self.upper_bound
#
#     def number_of_variables(self) -> int:
#         return self._number_of_variables
#
#     def number_of_objectives(self) -> int:
#         return 1
#
#     def number_of_constraints(self) -> int:
#         return 0
#
#     def evaluate(self, solution: FloatSolution) -> FloatSolution:
#         x = np.array(solution.variables)
#         X = x.reshape((self.atoms, self.dims))
#         energy = 0.0
#         for i in range(self.atoms - 1):
#             for j in range(i + 1, self.atoms):
#                 rij = np.linalg.norm(X[i] - X[j])
#                 if rij == 0:
#                     continue
#                 term = self.sigma / rij
#                 energy += term ** 12 - term ** 6
#         solution.objectives[0] = 4 * self.eps * energy
#         return solution
#
#     def name(self) -> str:
#         return f"Lennard-Jones - atoms={self.atoms}, dims={self.dims}"
