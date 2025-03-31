from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class LenardJohnesMinimumEnergyCluster(FloatProblem):
    def __init__(self, number_of_variables: int = 10):
        super(LenardJohnesMinimumEnergyCluster, self).__init__()
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
            sum_k += (variables[3 * i + k - 2] - variables[3 * j + k - 2]) ** 2
        return sum_k ** 3

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        s = 12.7120622568
        num_points = self.number_of_variables() // 3
        epsilon = 1e-6  # anti ZeroDivisionError
        for i in range(num_points - 2):
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
        return 'Lenard-Johnes Minimum Energy Cluster'