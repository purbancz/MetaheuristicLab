import random
from collections import deque
from typing import List, TypeVar

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.WAPSO import WorstAwarePSO
from algorithm.single_objective_PSO import SingleObjectivePSO

S = TypeVar('S')


class RoleMixin:
    @staticmethod
    def mark_particles(swarm: List[S], fraction: float, role: str) -> None:
        """Mark a fraction of particles with the given role attribute."""
        count = max(1, int(len(swarm) * fraction))
        indices = random.sample(range(len(swarm)), count)
        for i, particle in enumerate(swarm):
            particle.attributes[role] = (i in indices)

    @staticmethod
    def compute_component(particle: S, target: np.ndarray,
                          current: np.ndarray, normal_coefficient: float,
                          role_coefficient: float, role_flag: str) -> np.ndarray:
        """
        Computes a directional component based on whether the particle has the given role.
        If the role is active, the direction is inverted (i.e. repulsion) and scaled by role_coefficient.
        Otherwise, the normal attraction is computed scaled by normal_coefficient.
        """
        r = random.random()
        if particle.attributes.get(role_flag, False):
            # inverted direction
            return r * role_coefficient * (current - target)
        else:
            return r * normal_coefficient * (target - current)


class RebelPSO(SingleObjectivePSO, RoleMixin):
    """PSO with rebel particles opposing global best"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac2: float,
                 w: float,
                 rebel_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac2 = ac2
        self.rebel_fraction = rebel_fraction

    def create_initial_solutions(self) -> List[FloatSolution]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.rebel_fraction, 'is_rebel')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        for particle in swarm:
            r1 = random.random()
            cognitive_vec = self.c1 * r1 * (
                    np.array(particle.attributes['best_position']) - np.array(particle.variables))
            social_vec = self.compute_component(particle, g_best, np.array(particle.variables),
                                                self.c2, self.ac2, 'is_rebel')
            velocity = (self.w * np.array(particle.attributes['velocity'])
                        + cognitive_vec
                        + social_vec)
            particle.attributes['velocity'] = velocity.tolist()


class RejectorPSO(SingleObjectivePSO, RoleMixin):
    """PSO with rejector particles opposing personal best"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 w: float,
                 escapist_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.escapist_fraction = escapist_fraction

    def create_initial_solutions(self) -> List[FloatSolution]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.escapist_fraction, 'is_rejector')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        for particle in swarm:
            r2 = random.random()
            cognitive_vec = self.compute_component(particle, np.array(particle.attributes['best_position']),
                                                   np.array(particle.variables),
                                                   self.c1, self.ac1, 'is_rejector')
            social_vec = self.c2 * r2 * (g_best - np.array(particle.variables))
            velocity = (self.w * np.array(particle.attributes['velocity'])
                        + cognitive_vec
                        + social_vec)
            particle.attributes['velocity'] = velocity.tolist()


class RebelRejectorPSO(SingleObjectivePSO, RoleMixin):
    """PSO with both rebel and rejector particles"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 ac2: float,
                 w: float,
                 rebel_fraction: float,
                 escapist_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.ac2 = ac2
        self.rebel_fraction = rebel_fraction
        self.escapist_fraction = escapist_fraction

    def _mark_special_particles(self, swarm: List[S]) -> None:
        self.mark_particles(swarm, self.rebel_fraction, 'is_rebel')

        # disjoint sets, so to use my method I need to recalculate the fraction
        total = len(swarm)
        desired_escapists = max(1, int(total * self.escapist_fraction))
        num_rebels = sum(1 for p in swarm if p.attributes.get('is_rebel', False))
        non_rebel_count = total - num_rebels
        effective_fraction = desired_escapists / non_rebel_count if non_rebel_count > 0 else 1.0
        effective_fraction = min(1.0, effective_fraction)

        non_rebels = [p for p in swarm if not p.attributes.get('is_rebel', False)]
        self.mark_particles(non_rebels, effective_fraction, 'is_escapist')

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._mark_special_particles(solutions)
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        for particle in swarm:
            current = np.array(particle.variables)
            p_best = np.array(particle.attributes['best_position'])
            social_vec = self.compute_component(particle, g_best, current, self.c2, self.ac2, 'is_rebel')
            cognitive_vec = self.compute_component(particle, p_best, current, self.c1, self.ac1, 'is_escapist')
            velocity = (self.w * np.array(particle.attributes['velocity'])
                        + social_vec
                        + cognitive_vec)
            particle.attributes['velocity'] = velocity.tolist()


class RRAPSO(RebelRejectorPSO):
    """PSO with rebel and escapist particles and adaptive parameters"""

    def __init__(self,
                 problem: FloatProblem,
                 termination_criterion: TerminationCriterion,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 ac2: float,
                 base_inertia: float,
                 min_inertia: float,
                 max_inertia: float,
                 rebel_fraction: float,
                 escapist_fraction: float,
                 window_size: int = 10,
                 perturbation_probability: float = 0.1,
                 perturbation_scale: float = 0.1,
                 max_rebel_fraction: float = 0.8,
                 max_escapist_fraction: float = 0.8,
                 diversity_threshold: float = 0.1,
                 improvement_threshold: float = 0.01):
        # Initialize base using rebel and escapist fractions.
        super().__init__(problem, swarm_size, c1, c2, ac1, ac2, base_inertia, rebel_fraction, escapist_fraction,
                         termination_criterion)
        # Adaptive parameters
        self.base_inertia = base_inertia
        self.min_inertia = min_inertia
        self.max_inertia = max_inertia
        self.w = base_inertia
        self.max_rebel_fraction = max_rebel_fraction
        self.max_escapist_fraction = max_escapist_fraction
        self.original_rebel_fraction = rebel_fraction
        self.original_escapist_fraction = escapist_fraction
        self.diversity_threshold = diversity_threshold
        self.improvement_threshold = improvement_threshold
        self.perturbation_probability = perturbation_probability
        self.perturbation_scale = perturbation_scale
        self.window_size = window_size
        self.convergence_window = deque(maxlen=self.window_size)

    def _mark_special_particles(self, swarm: List[S]) -> None:
        self.mark_particles(swarm, self.rebel_fraction, 'is_rebel')
        self.mark_particles(swarm, self.escapist_fraction, 'is_escapist')

    def update_special_particles(self, swarm: List[S]) -> None:
        """
        Adjust the rebel and escapist properties for the swarm based on
        self.rebel_fraction and self.escapist_fraction.
        """
        total_particles = len(swarm)

        # Determine desired counts (ensuring at least one particle per type)
        desired_num_rebels = max(1, int(total_particles * self.rebel_fraction))
        desired_num_escapists = max(1, int(total_particles * self.escapist_fraction))

        # Get current particles with these properties
        current_rebels = [p for p in swarm if p.attributes.get('is_rebel', False)]
        current_escapists = [p for p in swarm if p.attributes.get('is_escapist', False)]

        # --- Adjust Rebel Particles ---
        if len(current_rebels) < desired_num_rebels:
            # Increase: Only assign to those that are not yet rebels.
            non_rebels = [p for p in swarm if not p.attributes.get('is_rebel', False)]
            num_to_assign = desired_num_rebels - len(current_rebels)
            if non_rebels and num_to_assign > 0:
                selected = random.sample(non_rebels, min(num_to_assign, len(non_rebels)))
                for particle in selected:
                    particle.attributes['is_rebel'] = True
        elif len(current_rebels) > desired_num_rebels:
            # Decrease: Remove rebel property randomly from those that currently are rebels.
            num_to_remove = len(current_rebels) - desired_num_rebels
            if current_rebels and num_to_remove > 0:
                selected = random.sample(current_rebels, num_to_remove)
                for particle in selected:
                    particle.attributes['is_rebel'] = False

        # --- Adjust Escapist Particles ---
        if len(current_escapists) < desired_num_escapists:
            # Increase: Only assign to those that are not yet escapists.
            non_escapists = [p for p in swarm if not p.attributes.get('is_escapist', False)]
            num_to_assign = desired_num_escapists - len(current_escapists)
            if non_escapists and num_to_assign > 0:
                selected = random.sample(non_escapists, min(num_to_assign, len(non_escapists)))
                for particle in selected:
                    particle.attributes['is_escapist'] = True
        elif len(current_escapists) > desired_num_escapists:
            # Decrease: Remove escapist property randomly.
            num_to_remove = len(current_escapists) - desired_num_escapists
            if current_escapists and num_to_remove > 0:
                selected = random.sample(current_escapists, num_to_remove)
                for particle in selected:
                    particle.attributes['is_escapist'] = False

    def update_velocity(self, swarm: List[S]) -> None:
        diversity = self.calculate_swarm_diversity(swarm)
        self.adapt_parameters(diversity, swarm)
        super().update_velocity(swarm)

    def adapt_parameters(self, diversity: float, swarm: List[FloatSolution]) -> None:
        if diversity < self.diversity_threshold:
            self.w = min(self.max_inertia, self.w * 1.01)
        else:
            self.w = max(self.min_inertia, self.w * 0.99)

        # Role adaptation: update ratios based on improvement rate
        improvement_rate = self.calculate_improvement_rate()
        if improvement_rate < self.improvement_threshold:
            self.rebel_fraction = min(self.max_rebel_fraction, self.rebel_fraction + 1 / super().swarm_size)
            self.escapist_fraction = min(self.max_escapist_fraction, self.escapist_fraction + 1 / super().swarm_size)
        else:
            self.rebel_fraction = max(self.original_rebel_fraction, self.rebel_fraction - 1 / super().swarm_size)
            self.escapist_fraction = max(self.original_escapist_fraction,
                                         self.escapist_fraction - 1 / super().swarm_size)

        self.update_special_particles(swarm)

    @staticmethod
    def calculate_swarm_diversity(swarm) -> float:
        """Measure population spread using mean pairwise distance"""
        positions = np.array([p.variables for p in swarm])
        centroid = np.mean(positions, axis=0)
        return np.mean(np.linalg.norm(positions - centroid, axis=1))

    def calculate_improvement_rate(self) -> float:
        """Calculate relative fitness improvement over the last window_size iterations."""
        self.convergence_window.append(self.best_global.objectives[0])

        if len(self.convergence_window) < 2:
            return 0.0

        initial = self.convergence_window[0]
        latest = self.convergence_window[-1]

        epsilon = 1e-8
        if abs(initial) < epsilon:
            return 0.0

        improvement_rate = (initial - latest) / abs(initial)
        return improvement_rate

    def perturbation(self, swarm: List[S]) -> None:
        """Chaotic perturbation for diversity maintenance with parameterized probability and scale."""
        best = self.best_global.variables
        for particle in swarm:
            if random.random() < self.perturbation_probability * (1 - self.w):
                noise = self.perturbation_scale * (self.max_inertia - self.w) * (np.random.rand() - 0.5)
                particle.variables = [
                    np.clip(x + noise * (x - best[i]),
                            self.problem.lower_bound[i],
                            self.problem.upper_bound[i])
                    for i, x in enumerate(particle.variables)
                ]

    def get_name(self) -> str:
        return "RRAPSO"


#####################################
# Worst aware roles
# Negative
#####################################

class ContrarianPSO(WorstAwarePSO, RoleMixin):
    """
    Contrarian PSO:
    Particles marked as contrarian (flag 'is_contrarian') use global worst instead of global best.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac2: float,
                 w: float,
                 contrarian_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac2 = ac2
        self.contrarian_fraction = contrarian_fraction

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.contrarian_fraction, 'is_contrarian')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        # For each particle, if marked as contrarian, use global worst; otherwise use global best.
        for particle in swarm:
            current = np.array(particle.variables)
            p_best = np.array(particle.attributes['best_position'])
            if particle.attributes.get('is_contrarian', False):
                social_vec = self.ac2 * random.random() * (g_worst - current)
            else:
                social_vec = self.c2 * random.random() * (g_best - current)
            cognitive_vec = self.c1 * random.random() * (p_best - current)
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)


class DefeatistPSO(WorstAwarePSO, RoleMixin):
    """
    Defeatist PSO:
    Particles marked as defeatist (flag 'is_defeatist') use their personal worst instead of personal best.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 w: float,
                 defeatist_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.defeatist_fraction = defeatist_fraction

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.defeatist_fraction, 'is_defeatist')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        # For each particle, if marked as defeatist, use personal worst; otherwise use personal best.
        for particle in swarm:
            current = np.array(particle.variables)
            if particle.attributes.get('is_defeatist', False):
                p_worst = np.array(particle.attributes['worst_position'])  # personal worst
                cognitive_vec = self.ac1 * random.random() * (p_worst - current)
            else:
                p_best = np.array(particle.attributes['best_position'])
                cognitive_vec = self.c1 * random.random() * (p_best - current)
            social_vec = self.c2 * random.random() * (g_best - current)
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)


class ContrarianDefeatistPSO(WorstAwarePSO, RoleMixin):
    """
    Contrarian-Defeatist PSO:
    Particles may be marked as contrarian (using global worst) and/or defeatist (using personal worst).
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 ac2: float,
                 w: float,
                 termination_criterion: TerminationCriterion,
                 contrarian_fraction: float,
                 defeatist_fraction: float):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.ac2 = ac2
        self.contrarian_fraction = contrarian_fraction
        self.defeatist_fraction = defeatist_fraction

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.contrarian_fraction, 'is_contrarian')
        self.mark_particles(solutions, self.defeatist_fraction, 'is_defeatist')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        for particle in swarm:
            current = np.array(particle.variables)
            # Social component: choose global worst if contrarian; else global best.
            if particle.attributes.get('is_contrarian', False):
                social_dir = np.array(self.global_worst.variables) - current
                social_vec = self.ac2 * random.random() * social_dir
            else:
                social_dir = np.array(self.best_global.variables) - current
                social_vec = self.c2 * random.random() * social_dir
            # Cognitive component: choose personal worst if defeatist; else personal best.
            if particle.attributes.get('is_defeatist', False):
                cognitive_dir = np.array(particle.attributes['worst_position']) - current
                cognitive_vec = self.ac1 * random.random() * cognitive_dir
            else:
                cognitive_dir = np.array(particle.attributes['best_position']) - current
                cognitive_vec = self.c1 * random.random() * cognitive_dir
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)


#####################################
# Worst aware roles
# Positive
#####################################


class EschewerPSO(WorstAwarePSO, RoleMixin):
    """
    Eschewer PSO:
    Particles marked as eschewer (flag 'is_eschewer') avoid global worst instead of going towards global best.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac2: float,
                 w: float,
                 eschewer_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac2 = ac2
        self.eschewer_fraction = eschewer_fraction

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.eschewer_fraction, 'is_eschewer')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        for particle in swarm:
            current = np.array(particle.variables)
            p_best = np.array(particle.attributes['best_position'])
            if particle.attributes.get('is_eschewer', False):
                social_vec = self.ac2 * random.random() * (current - g_worst)
            else:
                social_vec = self.c2 * random.random() * (g_best - current)
            cognitive_vec = self.c1 * random.random() * (p_best - current)
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)


class EscapistPSO(WorstAwarePSO, RoleMixin):
    """
    Escapist PSO:
    Particles marked as escapist (flag 'is_escapist') avoid their personal worst instead of going towards
    personal best.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 w: float,
                 escapist_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.escapist_fraction = escapist_fraction

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.escapist_fraction, 'is_escapist')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        for particle in swarm:
            current = np.array(particle.variables)
            if particle.attributes.get('is_escapist', False):
                p_worst = np.array(particle.attributes['worst_position'])  # personal worst
                cognitive_vec = self.ac1 * random.random() * (current - p_worst)
            else:
                p_best = np.array(particle.attributes['best_position'])
                cognitive_vec = self.c1 * random.random() * (p_best - current)
            social_vec = self.c2 * random.random() * (g_best - current)
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)


class EschewerEscapistPSO(WorstAwarePSO, RoleMixin):

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 ac2: float,
                 w: float,
                 eschewer_fraction: float,
                 escapist_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.ac2 = ac2
        self.eschewer_fraction = eschewer_fraction
        self.escapist_fraction = escapist_fraction

    """
    Eschewer - Escapist PSO:
    Particles may be marked as eschewer (avoiding global worst) and/or escapist (avoiding personal worst).
    """

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.eschewer_fraction, 'is_eschewer')
        self.mark_particles(solutions, self.escapist_fraction, 'is_escapist')
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        for particle in swarm:
            current = np.array(particle.variables)
            # Social component: choose global worst if eschewer; else global best.
            if particle.attributes.get('is_eschewer', False):
                social_dir = current - np.array(self.global_worst.variables)
                social_vec = self.ac2 * random.random() * social_dir
            else:
                social_dir = np.array(self.best_global.variables) - current
                social_vec = self.c2 * random.random() * social_dir
            # Cognitive component: choose personal worst if escapist; else personal best.
            if particle.attributes.get('is_escapist', False):
                cognitive_dir = current - np.array(particle.attributes['worst_position'])
                cognitive_vec = self.ac1 * random.random() * cognitive_dir
            else:
                cognitive_dir = np.array(particle.attributes['best_position']) - current
                cognitive_vec = self.c1 * random.random() * cognitive_dir
            velocity = self.w * np.array(particle.attributes['velocity']) + social_vec + cognitive_vec
            particle.attributes['velocity'] = velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)