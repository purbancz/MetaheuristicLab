import random
from collections import deque
from typing import List, TypeVar, Dict

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


class AdaptiveRoleMixin:
    def _init_adaptive(self,
                       swarm_size: int,
                       base_inertia: float,
                       min_inertia: float,
                       max_inertia: float,
                       role_fractions: Dict[str, float],
                       max_role_fractions: Dict[str, float],
                       diversity_threshold: float,
                       improvement_threshold: float,
                       window_size: int):
        # inertia params
        self.base_inertia = base_inertia
        self.min_inertia = min_inertia
        self.max_inertia = max_inertia
        self.w = base_inertia

        # role fraction params
        self.original_fractions = role_fractions.copy()
        self.role_fractions = role_fractions
        self.max_role_fractions = max_role_fractions

        # scheduling thresholds
        self.diversity_threshold = diversity_threshold
        self.improvement_threshold = improvement_threshold
        self.convergence_window = deque(maxlen=window_size)
        self.window_size = window_size

    def calculate_swarm_diversity(self, swarm: List) -> float:
        pos = np.array([p.variables for p in swarm])
        cen = pos.mean(axis=0)
        return np.linalg.norm(pos - cen, axis=1).mean()

    def calculate_improvement_rate(self) -> float:
        val = self.best_global.objectives[0]
        self.convergence_window.append(val)
        if len(self.convergence_window) < 2:
            return 0.0
        initial, latest = self.convergence_window[0], self.convergence_window[-1]
        return (initial - latest) / abs(initial) if abs(initial) > 1e-8 else 0.0

    def adapt_parameters(self, swarm: List) -> None:
        # adjust inertia
        div = self.calculate_swarm_diversity(swarm)
        if div < self.diversity_threshold:
            self.w = min(self.max_inertia, self.w * 1.01)
        else:
            self.w = max(self.min_inertia, self.w * 0.99)

        # adjust each role fraction
        rate = self.calculate_improvement_rate()
        for flag, orig in self.original_fractions.items():
            maxf = self.max_role_fractions[flag]
            if rate < self.improvement_threshold:
                self.role_fractions[flag] = min(maxf,
                                                self.role_fractions[flag] + 1 / self.swarm_size)
            else:
                self.role_fractions[flag] = max(orig,
                                                self.role_fractions[flag] - 1 / self.swarm_size)

        # re-mark the swarm according to new fractions
        self._update_special_particles(swarm)

    def _update_special_particles(self, swarm: List) -> None:
        # ensure each role exactly matches its fraction
        total = len(swarm)
        for flag, frac in self.role_fractions.items():
            desired = max(1, int(total * frac))
            current = [p for p in swarm if p.attributes.get(flag, False)]
            if len(current) < desired:
                candidates = [p for p in swarm if not p.attributes.get(flag, False)]
                for p in random.sample(candidates, desired - len(current)):
                    p.attributes[flag] = True
            elif len(current) > desired:
                for p in random.sample(current, len(current) - desired):
                    p.attributes[flag] = False


class AdaptiveRolePSO(SingleObjectivePSO, AdaptiveRoleMixin, RoleMixin):
    def __init__(self,
                 problem, swarm_size,
                 c1, c2, w,
                 termination_criterion,
                 base_inertia, min_inertia, max_inertia,
                 role_fractions: Dict[str,float],
                 max_role_fractions: Dict[str,float],
                 diversity_threshold, improvement_threshold,
                 window_size):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self._init_adaptive(swarm_size,
                            base_inertia, min_inertia, max_inertia,
                            role_fractions, max_role_fractions,
                            diversity_threshold, improvement_threshold,
                            window_size)

    def create_initial_solutions(self):
        solutions = super().create_initial_solutions()
        # initial marking from original fractions
        for flag, frac in self.original_fractions.items():
            self.mark(solutions, frac, flag)
        return solutions

    def update_velocity(self, swarm: List):
        # adapt before updating
        self.adapt_parameters(swarm)
        # now call the standard velocity update of the parent
        super().update_velocity(swarm)


#######################################################


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

    def get_name(self) -> str:
        return "RebelPSO"


class RejectorPSO(SingleObjectivePSO, RoleMixin):
    """PSO with rejector particles opposing personal best"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 ac1: float,
                 w: float,
                 rejector_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.rejector_fraction = rejector_fraction

    def create_initial_solutions(self) -> List[FloatSolution]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.rejector_fraction, 'is_rejector')
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

    def get_name(self) -> str:
        return "RejectorPSO"


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
                 rejector_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.ac1 = ac1
        self.ac2 = ac2
        self.rebel_fraction = rebel_fraction
        self.rejector_fraction = rejector_fraction

    def _mark_special_particles(self, swarm: List[S]) -> None:
        self.mark_particles(swarm, self.rebel_fraction, 'is_rebel')

        # disjoint sets, so to use my method I need to recalculate the fraction
        total = len(swarm)
        desired_rejectors = max(1, int(total * self.rejector_fraction))
        num_rebels = sum(1 for p in swarm if p.attributes.get('is_rebel', False))
        non_rebel_count = total - num_rebels
        effective_fraction = desired_rejectors / non_rebel_count if non_rebel_count > 0 else 1.0
        effective_fraction = min(1.0, effective_fraction)

        non_rebels = [p for p in swarm if not p.attributes.get('is_rebel', False)]
        self.mark_particles(non_rebels, effective_fraction, 'is_rejector')

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
            cognitive_vec = self.compute_component(particle, p_best, current, self.c1, self.ac1, 'is_rejector')
            velocity = (self.w * np.array(particle.attributes['velocity'])
                        + social_vec
                        + cognitive_vec)
            particle.attributes['velocity'] = velocity.tolist()

    def get_name(self) -> str:
        return "RebelRejectorPSO"


class RRAPSO(RebelRejectorPSO, AdaptiveRoleMixin):
    def __init__(self,
                 problem: FloatProblem,
                 termination_criterion: TerminationCriterion,
                 swarm_size: int,
                 c1: float, c2: float,
                 ac1: float, ac2: float,
                 base_inertia: float,
                 min_inertia: float,
                 max_inertia: float,
                 rebel_fraction: float,
                 rejector_fraction: float,
                 window_size: int = 10,
                 max_rebel_fraction: float = 0.8,
                 max_rejector_fraction: float = 0.8,
                 diversity_threshold: float = 0.1,
                 improvement_threshold: float = 0.01):

        # 1) Initialize Rebel+Rejector behavior
        RebelRejectorPSO.__init__(
            self,
            problem=problem,
            swarm_size=swarm_size,
            c1=c1,
            c2=c2,
            ac1=ac1,
            ac2=ac2,
            w=base_inertia,
            rebel_fraction=rebel_fraction,
            rejector_fraction=rejector_fraction,
            termination_criterion=termination_criterion
        )

        # 2) Initialize only the adaptive‐scheduler state
        AdaptiveRoleMixin._init_adaptive(
            self,
            swarm_size=swarm_size,
            base_inertia=base_inertia,
            min_inertia=min_inertia,
            max_inertia=max_inertia,
            role_fractions={
                'is_rebel':    rebel_fraction,
                'is_rejector': rejector_fraction,
            },
            max_role_fractions={
                'is_rebel':    max_rebel_fraction,
                'is_rejector': max_rejector_fraction,
            },
            diversity_threshold=diversity_threshold,
            improvement_threshold=improvement_threshold,
            window_size=window_size
        )

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

    def get_name(self) -> str:
        return "ContrarianPSO"


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

    def get_name(self) -> str:
        return "DefeatistPSO"


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

    def get_name(self) -> str:
        return "ContrarianDefeatistPSO"


class CDAPSO(ContrarianDefeatistPSO, AdaptiveRoleMixin):
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
                 contrarian_fraction: float,
                 defeatist_fraction: float,
                 window_size: int = 10,
                 max_contrarian_fraction: float = 0.8,
                 max_defeatist_fraction: float = 0.8,
                 diversity_threshold: float = 0.1,
                 improvement_threshold: float = 0.01):

        # 1) Initialize the Contrarian+Defeatist behavior
        ContrarianDefeatistPSO.__init__(
            self,
            problem=problem,
            swarm_size=swarm_size,
            c1=c1,
            c2=c2,
            ac1=ac1,
            ac2=ac2,
            w=base_inertia,
            contrarian_fraction=contrarian_fraction,
            defeatist_fraction=defeatist_fraction,
            termination_criterion=termination_criterion
        )

        # 2) Initialize only the adaptive‑scheduler state
        AdaptiveRoleMixin._init_adaptive(
            self,
            swarm_size=swarm_size,
            base_inertia=base_inertia,
            min_inertia=min_inertia,
            max_inertia=max_inertia,
            role_fractions={
                'is_contrarian': contrarian_fraction,
                'is_defeatist': defeatist_fraction,
            },
            max_role_fractions={
                'is_contrarian': max_contrarian_fraction,
                'is_defeatist': max_defeatist_fraction,
            },
            diversity_threshold=diversity_threshold,
            improvement_threshold=improvement_threshold,
            window_size=window_size
        )

    def get_name(self) -> str:
        return "CDAPSO"



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

    def get_name(self) -> str:
        return "EschewerPSO"


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

    def get_name(self) -> str:
        return "EscapistPSO"


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

    def get_name(self) -> str:
        return "EschewerEscapistPSO"


class EEAPSO(EschewerEscapistPSO, AdaptiveRoleMixin):
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
                 eschewer_fraction: float,
                 escapist_fraction: float,
                 window_size: int = 10,
                 max_eschewer_fraction: float = 0.8,
                 max_escapist_fraction: float = 0.8,
                 diversity_threshold: float = 0.1,
                 improvement_threshold: float = 0.01):

        # 1) Initialize the Eschewer+Escapist behavior
        EschewerEscapistPSO.__init__(
            self,
            problem=problem,
            swarm_size=swarm_size,
            c1=c1,
            c2=c2,
            ac1=ac1,
            ac2=ac2,
            w=base_inertia,
            eschewer_fraction=eschewer_fraction,
            escapist_fraction=escapist_fraction,
            termination_criterion=termination_criterion
        )

        # 2) Initialize only the adaptive‑scheduler state
        AdaptiveRoleMixin._init_adaptive(
            self,
            swarm_size=swarm_size,
            base_inertia=base_inertia,
            min_inertia=min_inertia,
            max_inertia=max_inertia,
            role_fractions={
                'is_eschewer':  eschewer_fraction,
                'is_escapist': escapist_fraction,
            },
            max_role_fractions={
                'is_eschewer':  max_eschewer_fraction,
                'is_escapist': max_escapist_fraction,
            },
            diversity_threshold=diversity_threshold,
            improvement_threshold=improvement_threshold,
            window_size=window_size
        )

    def get_name(self) -> str:
        return "EEAPSO"
