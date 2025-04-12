import random
from typing import List

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.single_objective_PSO import SingleObjectivePSO


class GradientEnhancedPSO(SingleObjectivePSO):
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 c3: float,  # New gradient coefficient
                 w: float,
                 termination_criterion: TerminationCriterion,
                 use_analytical_grad: bool = True):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.c3 = c3
        self.use_analytical_grad = use_analytical_grad

    def compute_gradient(self, particle: FloatSolution) -> np.ndarray:
        """Compute gradient using either analytical or numerical method"""
        if self.use_analytical_grad and hasattr(self.problem, 'gradient'):
            return self.problem.gradient(particle.variables)
        else:
            return self.numerical_gradient(particle)

    def numerical_gradient(self, particle: FloatSolution, epsilon: float = 1e-6) -> np.ndarray:
        """Finite difference approximation"""
        grad = np.zeros(self.problem.number_of_variables())  # Predefined gradient array

        for i in range(self.problem.number_of_variables()):
            # Perturbing the solution for x_plus
            x_plus = np.array(particle.variables)
            x_plus[i] += epsilon

            sol_plus = self.problem.create_solution()
            sol_plus.variables = x_plus.tolist()
            self.problem.evaluate(sol_plus)

            # Perturbing the solution for x_minus
            x_minus = np.array(particle.variables)
            x_minus[i] -= epsilon

            sol_minus = self.problem.create_solution()
            sol_minus.variables = x_minus.tolist()
            self.problem.evaluate(sol_minus)

            # Compute finite central difference for the gradient
            grad[i] = (sol_plus.objectives[0] - sol_minus.objectives[0]) / (2 * epsilon)

        return -grad  # Return negative gradient for minimization

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            # Standard PSO components
            r1, r2, r3 = [random.random() for _ in range(3)]
            velocity = np.array(particle.attributes['velocity'])
            pbest = np.array(particle.attributes['best_position'])
            gbest = np.array(self.best_global.variables)

            # Gradient component
            gradient = self.compute_gradient(particle)
            gradient_norm = np.linalg.norm(gradient)

            if gradient_norm > 0:
                gradient_component = gradient / gradient_norm  # Normalize
            else:
                gradient_component = 0

            # Hybrid velocity update
            new_velocity = (self.w * velocity +
                            self.c1 * r1 * (pbest - particle.variables) +
                            self.c2 * r2 * (gbest - particle.variables) +
                            self.c3 * r3 * gradient_component)

            particle.attributes['velocity'] = new_velocity.tolist()