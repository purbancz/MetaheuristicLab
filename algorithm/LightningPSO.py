import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion
from scipy.stats import rankdata

from algorithm.single_objective_PSO import SingleObjectivePSO


class LightningPSO(SingleObjectivePSO):
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 c3: float,
                 w: float,
                 active_ratio: float,
                 grad_sample: float,
                 dim_sample: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.active_ratio = active_ratio  # e.g. 0.3 for 30% worst performers
        self.grad_sample = grad_sample    # fraction for gradient sampling (e.g. 0.1)
        self.dim_sample = dim_sample      # fraction of dimensions to update (e.g. 0.5)
        self.c3 = c3
        self.grad_estimate = np.zeros(problem.number_of_variables())

    def _numerical_gradient(self, particle: FloatSolution, epsilon: float = 1e-6) -> np.ndarray:
        grad = np.zeros(self.problem.number_of_variables())
        for i in range(self.problem.number_of_variables()):
            x = np.array(particle.variables, dtype=float)
            x_plus = x.copy()
            x_plus[i] += epsilon
            sol_plus = self.problem.create_solution()
            sol_plus.variables = x_plus.tolist()
            self.problem.evaluate(sol_plus)
            x_minus = x.copy()
            x_minus[i] -= epsilon
            sol_minus = self.problem.create_solution()
            sol_minus.variables = x_minus.tolist()
            self.problem.evaluate(sol_minus)
            grad[i] = (sol_plus.objectives[0] - sol_minus.objectives[0]) / (2 * epsilon)
        return -grad  # Negative gradient for minimization

    def step(self):
        # 1. Select active particles: choose a fraction (e.g. 30%) with worst fitness.
        fitnesses = np.array([s.objectives[0] for s in self.solutions])
        num_active = max(1, int(self.swarm_size * self.active_ratio))
        # For minimization, worst performers are those with highest objective values.
        active_idx = np.argsort(fitnesses)[-num_active:]
        # 2. Approximate swarm gradient using a sample of particles.
        num_grad_samples = max(1, int(self.swarm_size * self.grad_sample))
        grad_indices = np.random.choice(self.swarm_size, num_grad_samples, replace=False)
        gradients = [self._numerical_gradient(self.solutions[i]) for i in grad_indices]
        self.grad_estimate = np.mean(gradients, axis=0)
        # 3. Update only the active particles.
        for idx in active_idx:
            particle = self.solutions[idx]
            current_velocity = np.array(particle.attributes['velocity'])
            current_position = np.array(particle.variables)
            # 4. Random dimension mask: update only in a subset of dimensions.
            dim_mask = np.random.rand(self.problem.number_of_variables()) < self.dim_sample
            new_velocity = (self.w * current_velocity +
                            self.c1 * np.random.rand() * (np.array(self.best_global.variables) - current_position) -
                            self.c2 * np.random.rand() * (current_position) +
                            self.c3 * self.grad_estimate)
            updated_velocity = np.where(dim_mask, new_velocity, current_velocity)
            particle.attributes['velocity'] = updated_velocity.tolist()
            new_position = current_position + updated_velocity
            new_position = np.clip(new_position, self.problem.lower_bound, self.problem.upper_bound)
            particle.variables = new_position.tolist()
        # Re-evaluate only the updated (active) particles and update the global best.
        active_particles = [self.solutions[i] for i in active_idx]
        self.evaluate(active_particles)
        self.update_global_best(self.solutions)