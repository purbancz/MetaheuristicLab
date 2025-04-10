import math
import random
import numpy as np
from scipy.integrate import quad
import warnings

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


# ----------------------------
# Helper functions
# ----------------------------

def _coth(x):
    """Compute the hyperbolic cotangent safely."""
    # For small arguments, use series expansion or a safeguard.
    if abs(x) < 1e-8:
        return 1.0 / x if x != 0 else np.inf
    val = 1.0 / np.tanh(x)
    # Cap value to avoid overflow
    if abs(val) > 1e15:
        return np.sign(val) * 1e15
    return val


def _spectral_density(omega, s, omega_c):
    """Compute the Ohmic-like spectral density
       J(omega) = omega^(s) * omega_c^(1-s) * exp(-omega/omega_c).
    """
    if omega <= 1e-15:
        return 0.0
    try:
        log_term = s * np.log(omega) + (1.0 - s) * np.log(omega_c) - omega / omega_c
    except (ValueError, OverflowError):
        return 0.0

    result = np.exp(log_term)
    return result if np.isfinite(result) else 0.0


def _integrand_D(omega, t, s, T_tilde, omega_c, T_THRESHOLD):
    """Integrand for the decay function D(t)."""
    J_val = _spectral_density(omega, s, omega_c)
    # Avoid division by zero and negligible spectral density
    if J_val < 1e-30 or omega < 1e-15:
        return 0.0
    part = (J_val / (omega ** 2)) * (1.0 - np.cos(omega * t))
    # Use coth = 1 for very low temperature (or T_tilde below threshold)
    if T_tilde < T_THRESHOLD:
        coth_val = 1.0
    else:
        arg = omega / (2.0 * T_tilde * omega_c)
        arg = np.clip(arg, -700, 700)
        coth_val = _coth(arg)
    return part * coth_val


def _integrand_gamma(omega, t, s, T_tilde, omega_c, T_THRESHOLD):
    """Integrand for the dephasing rate gamma(t)."""
    J_val = _spectral_density(omega, s, omega_c)
    if J_val < 1e-30:
        return 0.0
    if omega < 1e-15:
        part = J_val * t
    else:
        part = (J_val / omega) * np.sin(omega * t)
    if T_tilde < T_THRESHOLD:
        coth_val = 1.0
    else:
        arg = omega / (2.0 * T_tilde * omega_c)
        arg = np.clip(arg, -700, 700)
        coth_val = _coth(arg)
    return part * coth_val


# ----------------------------
# QSL Problem as a jMetalPy Problem
# ----------------------------

class QuantumSpeedLimit2D(FloatProblem):
    """
    Quantum Speed Limit (QSL) Time Bound problem for a dephasing qubit,
    based on Eq. (26) and Figure 8 from Cai et al., APL Quantum 2, 026102 (2025).

    Decision variables:
       x[0] = s          (Ohmicity parameter, e.g. in [0.1, 8.0])
       x[1] = T_tilde    (Scaled temperature T/omega_c, e.g. in [0.0, 2.0])

    Fixed physical parameters:
       omega_c   : cutoff frequency (set to 1.0 for scaling)
       omega_0   : transition frequency (set equal to omega_c)
       tau       : fixed evolution time (e.g., 10.0 / omega_c)

    The dephasing factor F(t) is computed as:
         F(t) = exp(-D(t)),
    where
         D(t) = ∫₀^(∞) _integrand_D(omega, t, s, T_tilde, omega_c, T_threshold) domega.

    The dephasing rate gamma(t) is computed as:
         gamma(t) = ∫₀^(∞) _integrand_gamma(omega, t, s, T_tilde, omega_c, T_threshold) domega.

    The QSL ratio is defined as:
         QSL_Ratio = N / ( (1/tau) ∫₀^(tau) v(t) dt ),
    with
         N = sqrt((F(tau) - cos(omega_0*tau))^2 + sin^2(omega_0*tau)),
         v(t) = sqrt(omega_0^2 + [gamma(t)]^2 * F(t)).

    The goal is to minimize QSL_Ratio.
    """

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def __init__(self):
        super(QuantumSpeedLimit2D, self).__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['QSL Ratio']

        # Decision variable bounds:
        # s (Ohmicity parameter): e.g. range [0.1, 8.0]
        # T_tilde (dimensionless temperature): e.g. range [0.0, 2.0]
        self.lower_bound = [0.1, 0.0]
        self.upper_bound = [8.0, 2.0]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        # Fixed physical parameters:
        self.omega_c = 1.0  # cutoff frequency
        self.omega_0 = self.omega_c  # set transition frequency equal to cutoff
        self.tau = 10.0 / self.omega_c  # fixed evolution time

        # Integration parameters:
        self.INTEGRATION_LIMIT = 50 * self.omega_c  # upper bound for omega integration
        self.T_THRESHOLD = 1e-9  # threshold for T_tilde considered to be 0

        # For memoization to speed up repeated evaluations:
        self._memo_D = {}
        self._memo_gamma = {}

    def create_solution(self) -> FloatSolution:
        solution = FloatSolution(self.lower_bound, self.upper_bound,
                                 self.number_of_objectives(), self.number_of_constraints())
        solution.variables = [random.uniform(lb, ub) for lb, ub in zip(self.lower_bound, self.upper_bound)]
        return solution

    def _compute_D(self, t, s, T_tilde):
        # Use a key for memoization
        key = (round(t, 8), round(s, 8), round(T_tilde, 8))
        if key in self._memo_D:
            return self._memo_D[key]
        try:
            print(f"Integrating D(t) for t={t:.4f}, s={s:.4f}, T_tilde={T_tilde:.4f}...")
            result, error = quad(_integrand_D, 1e-15, self.INTEGRATION_LIMIT,
                                 args=(t, s, T_tilde, self.omega_c, self.T_THRESHOLD),
                                 limit=50, epsabs=1e-6, epsrel=1e-6, full_output=0)
            print(f"Done: D(t) = {result:.4e}, error = {error:.2e}")
        except Exception as e:
            warnings.warn(f"Error in computing D(t): {e}")
            result = np.inf
        self._memo_D[key] = result
        return result

    def _compute_F(self, t, s, T_tilde):
        D_val = self._compute_D(t, s, T_tilde)
        try:
            F_val = np.exp(-D_val)
        except OverflowError:
            F_val = 0.0
        return F_val

    def _compute_gamma(self, t, s, T_tilde):
        key = (round(t, 8), round(s, 8), round(T_tilde, 8))
        if key in self._memo_gamma:
            return self._memo_gamma[key]
        try:
            print(f"Integrating D(t) for t={t:.4f}, s={s:.4f}, T_tilde={T_tilde:.4f}...")
            result, error = quad(_integrand_D, 1e-15, self.INTEGRATION_LIMIT,
                                 args=(t, s, T_tilde, self.omega_c, self.T_THRESHOLD),
                                 limit=50, epsabs=1e-6, epsrel=1e-6, full_output=0)
            print(f"Done: D(t) = {result:.4e}, error = {error:.2e}")
        except Exception as e:
            warnings.warn(f"Error in computing gamma(t): {e}")
            result = 0.0
        self._memo_gamma[key] = result
        return result

    def _denom_integrand(self, t, s, T_tilde):
        # Compute gamma(t) and F(t)
        gamma_t = self._compute_gamma(t, s, T_tilde)
        F_t = self._compute_F(t, s, T_tilde)
        # Use the formula from the paper without extra squaring on F
        inner = self.omega_0 ** 2 + (gamma_t ** 2) * F_t
        return np.sqrt(inner)

    def _compute_denominator(self, s, T_tilde):
        t_vals = np.linspace(0, self.tau, 1000)
        y_vals = [self._denom_integrand(t, s, T_tilde) for t in t_vals]
        integral = np.trapezoid(y_vals, t_vals)
        return integral / self.tau

    def _compute_numerator(self, s, T_tilde):
        F_tau = self._compute_F(self.tau, s, T_tilde)
        num = math.sqrt((F_tau - math.cos(self.omega_0 * self.tau)) ** 2 +
                        (math.sin(self.omega_0 * self.tau)) ** 2)
        return num

    def calculate_qsl_ratio(self, s, T_tilde):
        num = self._compute_numerator(s, T_tilde)
        denom = self._compute_denominator(s, T_tilde)
        if denom < 1e-12:
            return np.inf
        return num / denom

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Clear memo dictionaries for each new evaluation (can be optimized)
        self._memo_D.clear()
        self._memo_gamma.clear()

        s = solution.variables[0]
        T_tilde = solution.variables[1]
        print(f"Evaluating solution: s = {s:.4f}, T_tilde = {T_tilde:.4f}")
        ratio = self.calculate_qsl_ratio(s, T_tilde)
        print(f"Computed QSL Ratio = {ratio:.6f}")
        solution.objectives[0] = ratio
        return solution

    def name(self) -> str:
        return "QuantumSpeedLimit2D"


# ----------------------------
# Optional: For generalizing to higher dimensions (pairs)
# ----------------------------

class GeneralizedQuantumSpeedLimit(FloatProblem):
    """
    N-dimensional generalization of the QSL benchmark.

    This class generalizes QuantumSpeedLimit2D by considering that the decision
    vector is a concatenation of pairs, each pair corresponding to (s, T_tilde)
    for one "channel". The objective is the sum of the QSL ratios for each pair.

    The number of decision variables must be even.
    """

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def number_of_variables(self) -> int:
        return self._n_variables

    def __init__(self, number_of_variables: int = 4):
        # First, store the number of decision variables in a separate attribute.
        if number_of_variables % 2 != 0:
            raise ValueError("Number of variables must be even for GeneralizedQuantumSpeedLimit")
        self._n_variables = number_of_variables

        # Call the parent constructor.
        super(GeneralizedQuantumSpeedLimit, self).__init__()

        # Now set up bounds for each pair.
        half = self._n_variables // 2
        self.lower_bound = [0.1, 0.0] * half
        self.upper_bound = [8.0, 2.0] * half

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["Total QSL Ratio"]

        # Set the static lower/upper bounds for FloatSolution.
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        # Re-use our QuantumSpeedLimit2D instance for single-pair evaluation:
        self.single_pair_problem = QuantumSpeedLimit2D()

    def create_solution(self) -> FloatSolution:
        solution = FloatSolution(self.lower_bound, self.upper_bound,
                                 self.number_of_objectives(), self.number_of_constraints())
        solution.variables = [random.uniform(lb, ub) for lb, ub in zip(self.lower_bound, self.upper_bound)]
        return solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        # Loop over pairs
        half = self.number_of_variables() // 2
        for i in range(half):
            s = solution.variables[2 * i]
            T_tilde = solution.variables[2 * i + 1]
            # Use the single-pair calculation from our QuantumSpeedLimit2D class
            pair_ratio = self.single_pair_problem.calculate_qsl_ratio(s, T_tilde)
            total += pair_ratio
        solution.objectives[0] = total
        return solution

    def name(self) -> str:
        return f"GeneralizedQuantumSpeedLimit(N={self.number_of_variables})"

