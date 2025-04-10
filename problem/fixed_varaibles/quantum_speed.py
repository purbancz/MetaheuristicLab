import math
import random
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class QSLTimeBoundProblem(FloatProblem):
    """
    QSL Time Bound Problem for a dephasing qubit (generalized from figure 8 of the article).

    In this simplified model we assume:
      - Fixed evolution time: τ = 10.0
      - Transition frequency: ω₀ = 1.0
      - Dephasing factor: F(t) = exp(-T*t)
      - Dephasing rate: γ(t) = T * (1 + 0.1*(s-1))   (assumed constant)

    The QSL time bound ratio is computed as:

         τ_QSL/τ = sqrt( [F(τ) - cos(ω₀τ)]² + sin²(ω₀τ) )
                   ---------------------------------------------
                   (1/τ)∫₀^(τ) sqrt( ω₀² + γ(t)² * F(t) ) dt

    Decision variables:
      x[0] = T, the environmental temperature (range: [0.1, 5.0])
      x[1] = s, the Ohmicity parameter (range: [0.5, 6.0])

    The goal is to minimize this QSL ratio.
    """

    def __init__(self):
        super(QSLTimeBoundProblem, self).__init__()
        # Decision variable bounds:
        self.lower_bound = [0.1, 0.5]
        self.upper_bound = [5.0, 6.0]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["τ_QSL/τ"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def create_solution(self) -> FloatSolution:
        solution = FloatSolution(self.lower_bound,
                                 self.upper_bound,
                                 self.number_of_objectives(),
                                 self.number_of_constraints())
        solution.variables = [random.uniform(lb, ub) for lb, ub in zip(self.lower_bound, self.upper_bound)]
        return solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Unpack decision variables: T (environmental temperature) and s (Ohmicity parameter)
        T, s = solution.variables
        # Fixed parameters:
        tau = 10.0  # evolution time
        omega0 = 1.0  # qubit transition frequency

        # Compute the dephasing factor at time tau:
        F_tau = math.exp(-T * tau)
        # Numerator: based on the expression sqrt((F(τ) - cos(ω₀τ))^2 + sin^2(ω₀τ))
        numerator = math.sqrt((F_tau - math.cos(omega0 * tau)) ** 2 + math.sin(omega0 * tau) ** 2)

        # For the denominator, we integrate from 0 to tau.
        # Our toy model:
        #   F(t) = exp(-T*t) and γ(t) = T*(1+0.1*(s-1))  (assumed constant in time)
        gamma_val = T * (1 + 0.1 * (s - 1))

        # Define the integrand: sqrt( ω₀² + [γ_val]² * exp(-T*t) )
        def integrand(t):
            return math.sqrt(omega0 ** 2 + (gamma_val ** 2) * math.exp(-T * t))

        # Use numpy.trapz for numerical integration
        num_points = 1000
        t_vals = np.linspace(0, tau, num_points)
        y_vals = [integrand(ti) for ti in t_vals]
        integral = np.trapezoid(y_vals, t_vals)
        denom = (1.0 / tau) * integral

        ratio = numerator / denom
        solution.objectives[0] = ratio
        return solution

    def name(self) -> str:
        return "QSL Time Bound Problem"
