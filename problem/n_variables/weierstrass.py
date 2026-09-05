import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class ShiftedRotatedWeierstrass(FloatProblem):
    K_MAX = 20
    A = 0.5
    B = 3.0

    def __init__(self, number_of_variables: int = 2):
        super(ShiftedRotatedWeierstrass, self).__init__()

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-0.5] * number_of_variables
        self.upper_bound = [0.5] * number_of_variables

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.shift = np.random.uniform(-0.5, 0.5, size=self.number_of_variables())
        self.rotation_matrix = self.generate_random_orthogonal_matrix()

        # Precomputed constants (deterministic - consumes no RNG, so seeded
        # instance identity is unaffected): a^k, 2*pi*b^k, and the constant
        # inner sum at z = 0.5.
        k = np.arange(self.K_MAX + 1, dtype=float)
        self._a_pow = self.A ** k
        self._two_pi_b_pow = 2.0 * np.pi * self.B ** k
        self._sum2 = float(np.sum(self._a_pow * np.cos(self._two_pi_b_pow * 0.5)))

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def generate_random_orthogonal_matrix(self):
        random_matrix = np.random.randn(self.number_of_variables(), self.number_of_variables())
        q, _ = np.linalg.qr(random_matrix)
        return q

    def weierstrass(self, x):
        # sum_i sum_k a^k cos(2 pi b^k (x_i + 0.5)) - D * sum_k a^k cos(pi b^k),
        # vectorized over the (dim x k) grid.
        phases = np.outer(np.asarray(x, dtype=float) + 0.5, self._two_pi_b_pow)
        sum1 = float(np.sum(np.cos(phases) @ self._a_pow))
        return sum1 - self.number_of_variables() * self._sum2

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)

        shifted_x = x - self.shift

        rotated_x = np.dot(self.rotation_matrix, shifted_x)

        result = self.weierstrass(rotated_x)

        solution.objectives[0] = result

        return solution

    def name(self) -> str:
        return 'Shifted and Rotated Weierstrass'
