import numpy as np
import math
from typing import List
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion
from algorithm.single_objective_PSO import SingleObjectivePSO


class DCSPSO(SingleObjectivePSO):
    """
    Dynamic Cognitive-Social PSO (DCS-PSO) as in Kassoul et al. (2021).
    10.1109/ICARA51699.2021.9376550

    My intention is to compare it with my Global and Personal Adaptive :)

    - No inertia term.
    - Dynamic cognitive coefficient c1(t) with logarithmic "leaping".
    - Dynamic social coefficient c2(t) = (chi*pi)^alpha.
    - Velocity limits derived from problem bounds.
    - Stagnation escape via logarithmic jump.
    """
    def __init__(
        self,
        problem: FloatProblem,
        swarm_size: int,
        termination_criterion: TerminationCriterion,
        a0: float = math.e,
        d0: float = 0.9995,
        k: float = 0.99999,
        chi: float = 0.7298,
        epsilon: float = 1e-8
    ):
        # c1, c2, w passed to base but not used
        super().__init__(problem, swarm_size, c1=0.0, c2=0.0, w=0.0, termination_criterion=termination_criterion)
        self.a = a0
        self.d = d0
        self.k = k
        self.chi = chi
        self.epsilon = epsilon
        # Precompute velocity limits
        lb = np.array(self.problem.lower_bound, dtype=float)
        ub = np.array(self.problem.upper_bound, dtype=float)
        span = np.ceil(ub - lb)
        self.v_max = 0.9 * span
        self.v_min = -0.1 * span

    def update_coefficient(self):
        # update damping and cognitive parameter
        self.d *= self.k
        self.a *= self.d

    def compute_alpha(self) -> float:
        # alpha = 5 if variable range <= 10, else 1
        rng = np.abs(np.array(self.problem.upper_bound) - np.array(self.problem.lower_bound))
        return 5.0 if np.all(rng <= 10.0) else 1.0

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        alpha = self.compute_alpha()
        gbest = np.array(self.best_global.variables, dtype=float)
        for particle in swarm:
            curr = np.array(particle.variables, dtype=float)
            pbest = np.array(particle.attributes['best_position'], dtype=float)
            # dynamic coefficients
            dist = np.linalg.norm(pbest - curr)
            c1_t = (self.a ** alpha) * math.log(dist + self.epsilon)
            phi = self.chi * math.pi
            c2_t = phi ** alpha
            # random weights
            r1, r2 = np.random.rand(), np.random.rand()
            # proportional update (no inertia)
            h = r1 * c1_t * (pbest - curr) + r2 * c2_t * (gbest - curr)
            # clip velocity
            v = np.clip(h, self.v_min, self.v_max)
            particle.attributes['velocity'] = v.tolist()

    def step(self) -> None:
        # 1. adapt dynamic parameters
        self.update_coefficient()
        # 2. update velocities
        self.update_velocity(self.solutions)
        # 3. move particles
        self.update_position(self.solutions)
        # 4. evaluate
        self.solutions = self.evaluate(self.solutions)
        # 5. update bests
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)

    def get_name(self) -> str:
        return "DCSPSO"
