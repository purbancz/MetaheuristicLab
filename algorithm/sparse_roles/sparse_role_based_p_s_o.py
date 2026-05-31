"""Sparse single-role PSO variants.

Sparse role PSO applies role-based movement modifications only to selected
coordinates of the velocity update, analogously to gene-level mutation in
evolutionary algorithms.
"""

from __future__ import annotations

import random
from typing import List, TypeVar

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.WAPSO import WorstAwarePSO
from algorithm.particles_with_roles import RoleMixin
from algorithm.single_objective_PSO import PerturbationPSO, SingleObjectivePSO
from algorithm.sparse_roles.coordinate_mask import CoordinateMaskMixin

S = TypeVar("S")


class SparseRoleMixin(CoordinateMaskMixin):
    """Shared coordinate-mask parameter handling for sparse role PSOs."""

    def _init_single_coordinate_params(
            self,
            coordinate_mode: str,
            coordinate_fraction: float,
            coordinate_scale: float,
            coordinate_count: int,
    ) -> None:
        self.coordinate_mode = coordinate_mode
        self.coordinate_fraction = coordinate_fraction
        self.coordinate_scale = coordinate_scale
        self.coordinate_count_value = coordinate_count

    def _single_mask(self, dim: int) -> np.ndarray:
        return self.coordinate_mask(
            dim=dim,
            mode=self.coordinate_mode,
            fraction=self.coordinate_fraction,
            scale=self.coordinate_scale,
            count=self.coordinate_count_value,
        )

    def _init_component_coordinate_params(
            self,
            social_coordinate_mode: str,
            cognitive_coordinate_mode: str,
            social_coordinate_fraction: float,
            cognitive_coordinate_fraction: float,
            social_coordinate_scale: float,
            cognitive_coordinate_scale: float,
            social_coordinate_count: int,
            cognitive_coordinate_count: int,
    ) -> None:
        self.social_coordinate_mode = social_coordinate_mode
        self.cognitive_coordinate_mode = cognitive_coordinate_mode
        self.social_coordinate_fraction = social_coordinate_fraction
        self.cognitive_coordinate_fraction = cognitive_coordinate_fraction
        self.social_coordinate_scale = social_coordinate_scale
        self.cognitive_coordinate_scale = cognitive_coordinate_scale
        self.social_coordinate_count = social_coordinate_count
        self.cognitive_coordinate_count = cognitive_coordinate_count

    def _social_mask(self, dim: int) -> np.ndarray:
        return self.coordinate_mask(
            dim=dim,
            mode=self.social_coordinate_mode,
            fraction=self.social_coordinate_fraction,
            scale=self.social_coordinate_scale,
            count=self.social_coordinate_count,
        )

    def _cognitive_mask(self, dim: int) -> np.ndarray:
        return self.coordinate_mask(
            dim=dim,
            mode=self.cognitive_coordinate_mode,
            fraction=self.cognitive_coordinate_fraction,
            scale=self.cognitive_coordinate_scale,
            count=self.cognitive_coordinate_count,
        )


class SparseWandererPSO(SingleObjectivePSO, RoleMixin, CoordinateMaskMixin):
    """Coordinate-wise Wanderer PSO.

    Wanderer particles receive additive random noise only on selected
    coordinates. Non-wanderer particles follow standard PSO.
    """

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            termination_criterion: TerminationCriterion,
            w: float,
            c1: float,
            c2: float,
            noise_strength: float,
            wanderer_fraction: float,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.noise_strength = noise_strength
        self.wanderer_fraction = max(0.0, min(1.0, wanderer_fraction))
        self.coordinate_mode = coordinate_mode
        self.coordinate_fraction = coordinate_fraction
        self.coordinate_scale = coordinate_scale
        self.coordinate_count_value = coordinate_count

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.wanderer_fraction, "is_wanderer")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            current_vel = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            cognitive_vec = self.c1 * r1 * (p_best - current)
            social_vec = self.c2 * r2 * (g_best - current)

            base_velocity = self.w * current_vel + cognitive_vec + social_vec

            if particle.attributes.get("is_wanderer", False):
                mask = self.coordinate_mask(
                    dim=dim,
                    mode=self.coordinate_mode,
                    fraction=self.coordinate_fraction,
                    scale=self.coordinate_scale,
                    count=self.coordinate_count_value,
                )
                noise = self.noise_strength * np.random.uniform(-1.0, 1.0, dim)
                sparse_noise = np.where(mask, noise, 0.0)
                final_velocity = base_velocity + sparse_noise
            else:
                final_velocity = base_velocity

            particle.attributes["velocity"] = final_velocity.tolist()

    def get_name(self) -> str:
        return "SparseWandererPSO"


class SparseDefeatistPSO(WorstAwarePSO, RoleMixin, CoordinateMaskMixin):
    """Coordinate-wise Defeatist PSO.

    Defeatist particles use personal-worst attraction only on selected
    coordinates; unmasked coordinates retain standard personal-best attraction.
    """

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            defeatist_c: float,
            w: float,
            defeatist_fraction: float,
            termination_criterion: TerminationCriterion,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.defeatist_c = defeatist_c
        self.defeatist_fraction = max(0.0, min(1.0, defeatist_fraction))
        self.coordinate_mode = coordinate_mode
        self.coordinate_fraction = coordinate_fraction
        self.coordinate_scale = coordinate_scale
        self.coordinate_count_value = coordinate_count

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.defeatist_fraction, "is_defeatist")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            current_vel = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])
            p_worst = np.array(particle.attributes["worst_position"])

            r1 = random.random()
            r2 = random.random()
            r3 = random.random()

            normal_cognitive = self.c1 * r1 * (p_best - current)
            defeatist_cognitive = self.defeatist_c * r2 * (p_worst - current)

            if particle.attributes.get("is_defeatist", False):
                mask = self.coordinate_mask(
                    dim=dim,
                    mode=self.coordinate_mode,
                    fraction=self.coordinate_fraction,
                    scale=self.coordinate_scale,
                    count=self.coordinate_count_value,
                )
                cognitive_vec = self.mix_by_mask(normal_cognitive, defeatist_cognitive, mask)
            else:
                cognitive_vec = normal_cognitive

            social_vec = self.c2 * r3 * (g_best - current)
            new_velocity = self.w * current_vel + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "SparseDefeatistPSO"


class SparseContrarianDefeatistPSO(WorstAwarePSO, RoleMixin, CoordinateMaskMixin):
    """Coordinate-wise Contrarian-Defeatist PSO.

    Contrarian behavior is applied sparsely to the social component and
    defeatist behavior is applied sparsely to the cognitive component.
    """

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            defeatist_c: float,
            contrarian_c: float,
            w: float,
            termination_criterion: TerminationCriterion,
            contrarian_fraction: float,
            defeatist_fraction: float,
            social_coordinate_mode: str = "sqrt",
            cognitive_coordinate_mode: str = "sqrt",
            social_coordinate_fraction: float = 0.1,
            cognitive_coordinate_fraction: float = 0.1,
            social_coordinate_scale: float = 1.0,
            cognitive_coordinate_scale: float = 1.0,
            social_coordinate_count: int = 10,
            cognitive_coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.defeatist_c = defeatist_c
        self.contrarian_c = contrarian_c
        self.contrarian_fraction = max(0.0, min(1.0, contrarian_fraction))
        self.defeatist_fraction = max(0.0, min(1.0, defeatist_fraction))

        self.social_coordinate_mode = social_coordinate_mode
        self.cognitive_coordinate_mode = cognitive_coordinate_mode
        self.social_coordinate_fraction = social_coordinate_fraction
        self.cognitive_coordinate_fraction = cognitive_coordinate_fraction
        self.social_coordinate_scale = social_coordinate_scale
        self.cognitive_coordinate_scale = cognitive_coordinate_scale
        self.social_coordinate_count = social_coordinate_count
        self.cognitive_coordinate_count = cognitive_coordinate_count

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.contrarian_fraction, "is_contrarian")
        self.mark_particles(solutions, self.defeatist_fraction, "is_defeatist")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            current_vel = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])
            p_worst = np.array(particle.attributes["worst_position"])

            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            r4 = random.random()

            normal_social = self.c2 * r1 * (g_best - current)
            contrarian_social = self.contrarian_c * r2 * (g_worst - current)

            if particle.attributes.get("is_contrarian", False):
                social_mask = self.coordinate_mask(
                    dim=dim,
                    mode=self.social_coordinate_mode,
                    fraction=self.social_coordinate_fraction,
                    scale=self.social_coordinate_scale,
                    count=self.social_coordinate_count,
                )
                social_vec = self.mix_by_mask(normal_social, contrarian_social, social_mask)
            else:
                social_vec = normal_social

            normal_cognitive = self.c1 * r3 * (p_best - current)
            defeatist_cognitive = self.defeatist_c * r4 * (p_worst - current)

            if particle.attributes.get("is_defeatist", False):
                cognitive_mask = self.coordinate_mask(
                    dim=dim,
                    mode=self.cognitive_coordinate_mode,
                    fraction=self.cognitive_coordinate_fraction,
                    scale=self.cognitive_coordinate_scale,
                    count=self.cognitive_coordinate_count,
                )
                cognitive_vec = self.mix_by_mask(normal_cognitive, defeatist_cognitive, cognitive_mask)
            else:
                cognitive_vec = normal_cognitive

            new_velocity = self.w * current_vel + social_vec + cognitive_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "SparseContrarianDefeatistPSO"


class SparseRebelPSO(SingleObjectivePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Rebel PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            rebel_c: float,
            w: float,
            rebel_fraction: float,
            termination_criterion: TerminationCriterion,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.rebel_c = rebel_c
        self.rebel_fraction = max(0.0, min(1.0, rebel_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.rebel_fraction, "is_rebel")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            cognitive_vec = self.c1 * r1 * (p_best - current)
            normal_social = self.c2 * r2 * (g_best - current)
            rebel_social = self.rebel_c * r2 * (current - g_best)

            if particle.attributes.get("is_rebel", False):
                social_vec = self.mix_by_mask(normal_social, rebel_social, self._single_mask(dim))
            else:
                social_vec = normal_social

            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseRebelPSO"


class SparseRejectorPSO(SingleObjectivePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Rejector PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            rejector_c: float,
            w: float,
            rejector_fraction: float,
            termination_criterion: TerminationCriterion,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.rejector_c = rejector_c
        self.rejector_fraction = max(0.0, min(1.0, rejector_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.rejector_fraction, "is_rejector")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            normal_cognitive = self.c1 * r1 * (p_best - current)
            rejector_cognitive = self.rejector_c * r1 * (current - p_best)

            if particle.attributes.get("is_rejector", False):
                cognitive_vec = self.mix_by_mask(normal_cognitive, rejector_cognitive, self._single_mask(dim))
            else:
                cognitive_vec = normal_cognitive

            social_vec = self.c2 * r2 * (g_best - current)
            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseRejectorPSO"


class SparseRebelRejectorPSO(SingleObjectivePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Rebel-Rejector PSO with disjoint role marking."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            rejector_c: float,
            rebel_c: float,
            w: float,
            rebel_fraction: float,
            rejector_fraction: float,
            termination_criterion: TerminationCriterion,
            social_coordinate_mode: str = "sqrt",
            cognitive_coordinate_mode: str = "sqrt",
            social_coordinate_fraction: float = 0.1,
            cognitive_coordinate_fraction: float = 0.1,
            social_coordinate_scale: float = 1.0,
            cognitive_coordinate_scale: float = 1.0,
            social_coordinate_count: int = 10,
            cognitive_coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.rejector_c = rejector_c
        self.rebel_c = rebel_c
        self.rebel_fraction = max(0.0, min(1.0, rebel_fraction))
        self.rejector_fraction = max(0.0, min(1.0, rejector_fraction))
        self._init_component_coordinate_params(
            social_coordinate_mode,
            cognitive_coordinate_mode,
            social_coordinate_fraction,
            cognitive_coordinate_fraction,
            social_coordinate_scale,
            cognitive_coordinate_scale,
            social_coordinate_count,
            cognitive_coordinate_count,
        )

    def _mark_special_particles(self, swarm: List[S]) -> None:
        self.mark_particles(swarm, self.rebel_fraction, "is_rebel")

        total = len(swarm)
        desired_rejectors = max(1, int(total * self.rejector_fraction))
        num_rebels = sum(1 for p in swarm if p.attributes.get("is_rebel", False))
        non_rebel_count = total - num_rebels
        effective_fraction = desired_rejectors / non_rebel_count if non_rebel_count > 0 else 1.0
        effective_fraction = min(1.0, effective_fraction)

        non_rebels = [p for p in swarm if not p.attributes.get("is_rebel", False)]
        self.mark_particles(non_rebels, effective_fraction, "is_rejector")

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._mark_special_particles(solutions)
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            normal_cognitive = self.c1 * r1 * (p_best - current)
            rejector_cognitive = self.rejector_c * r1 * (current - p_best)
            if particle.attributes.get("is_rejector", False):
                cognitive_vec = self.mix_by_mask(normal_cognitive, rejector_cognitive, self._cognitive_mask(dim))
            else:
                cognitive_vec = normal_cognitive

            normal_social = self.c2 * r2 * (g_best - current)
            rebel_social = self.rebel_c * r2 * (current - g_best)
            if particle.attributes.get("is_rebel", False):
                social_vec = self.mix_by_mask(normal_social, rebel_social, self._social_mask(dim))
            else:
                social_vec = normal_social

            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseRebelRejectorPSO"


class SparseContrarianPSO(WorstAwarePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Contrarian PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            contrarian_c: float,
            w: float,
            contrarian_fraction: float,
            termination_criterion: TerminationCriterion,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.contrarian_c = contrarian_c
        self.contrarian_fraction = max(0.0, min(1.0, contrarian_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.contrarian_fraction, "is_contrarian")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            normal_social = self.c2 * r1 * (g_best - current)
            contrarian_social = self.contrarian_c * r1 * (g_worst - current)
            if particle.attributes.get("is_contrarian", False):
                social_vec = self.mix_by_mask(normal_social, contrarian_social, self._single_mask(dim))
            else:
                social_vec = normal_social

            cognitive_vec = self.c1 * r2 * (p_best - current)
            new_velocity = self.w * velocity + social_vec + cognitive_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "SparseContrarianPSO"


class SparseEschewerPSO(WorstAwarePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Eschewer PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            eschewer_c: float,
            w: float,
            eschewer_fraction: float,
            termination_criterion: TerminationCriterion,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.eschewer_c = eschewer_c
        self.eschewer_fraction = max(0.0, min(1.0, eschewer_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.eschewer_fraction, "is_eschewer")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            normal_social = self.c2 * r1 * (g_best - current)
            eschewer_social = self.eschewer_c * r1 * (current - g_worst)
            if particle.attributes.get("is_eschewer", False):
                social_vec = self.mix_by_mask(normal_social, eschewer_social, self._single_mask(dim))
            else:
                social_vec = normal_social

            cognitive_vec = self.c1 * r2 * (p_best - current)
            new_velocity = self.w * velocity + social_vec + cognitive_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "SparseEschewerPSO"


class SparseEscapistPSO(WorstAwarePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Escapist PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            escapist_c: float,
            w: float,
            escapist_fraction: float,
            termination_criterion: TerminationCriterion,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.escapist_c = escapist_c
        self.escapist_fraction = max(0.0, min(1.0, escapist_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.escapist_fraction, "is_escapist")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])
            p_worst = np.array(particle.attributes["worst_position"])

            r1 = random.random()
            r2 = random.random()

            normal_cognitive = self.c1 * r1 * (p_best - current)
            escapist_cognitive = self.escapist_c * r1 * (current - p_worst)
            if particle.attributes.get("is_escapist", False):
                cognitive_vec = self.mix_by_mask(normal_cognitive, escapist_cognitive, self._single_mask(dim))
            else:
                cognitive_vec = normal_cognitive

            social_vec = self.c2 * r2 * (g_best - current)
            new_velocity = self.w * velocity + social_vec + cognitive_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "SparseEscapistPSO"


class SparseEschewerEscapistPSO(WorstAwarePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Eschewer-Escapist PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            c1: float,
            c2: float,
            escapist_c: float,
            eschewer_c: float,
            w: float,
            eschewer_fraction: float,
            escapist_fraction: float,
            termination_criterion: TerminationCriterion,
            social_coordinate_mode: str = "sqrt",
            cognitive_coordinate_mode: str = "sqrt",
            social_coordinate_fraction: float = 0.1,
            cognitive_coordinate_fraction: float = 0.1,
            social_coordinate_scale: float = 1.0,
            cognitive_coordinate_scale: float = 1.0,
            social_coordinate_count: int = 10,
            cognitive_coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.escapist_c = escapist_c
        self.eschewer_c = eschewer_c
        self.eschewer_fraction = max(0.0, min(1.0, eschewer_fraction))
        self.escapist_fraction = max(0.0, min(1.0, escapist_fraction))
        self._init_component_coordinate_params(
            social_coordinate_mode,
            cognitive_coordinate_mode,
            social_coordinate_fraction,
            cognitive_coordinate_fraction,
            social_coordinate_scale,
            cognitive_coordinate_scale,
            social_coordinate_count,
            cognitive_coordinate_count,
        )

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.eschewer_fraction, "is_eschewer")
        self.mark_particles(solutions, self.escapist_fraction, "is_escapist")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])
            p_worst = np.array(particle.attributes["worst_position"])

            r1 = random.random()
            r2 = random.random()

            normal_social = self.c2 * r1 * (g_best - current)
            eschewer_social = self.eschewer_c * r1 * (current - g_worst)
            if particle.attributes.get("is_eschewer", False):
                social_vec = self.mix_by_mask(normal_social, eschewer_social, self._social_mask(dim))
            else:
                social_vec = normal_social

            normal_cognitive = self.c1 * r2 * (p_best - current)
            escapist_cognitive = self.escapist_c * r2 * (current - p_worst)
            if particle.attributes.get("is_escapist", False):
                cognitive_vec = self.mix_by_mask(normal_cognitive, escapist_cognitive, self._cognitive_mask(dim))
            else:
                cognitive_vec = normal_cognitive

            new_velocity = self.w * velocity + social_vec + cognitive_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "SparseEschewerEscapistPSO"


class SparseAnarchicPSO(SingleObjectivePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Anarchic PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            termination_criterion: TerminationCriterion,
            w: float,
            c1: float,
            c2: float,
            random_strength: float,
            anarchic_fraction: float,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.random_strength = random_strength
        self.anarchic_fraction = max(0.0, min(1.0, anarchic_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.anarchic_fraction, "is_anarchic")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            cognitive_vec = self.c1 * r1 * (p_best - current)
            normal_social = self.c2 * r2 * (g_best - current)
            random_social = self.random_strength * np.random.uniform(-1.0, 1.0, dim)

            if particle.attributes.get("is_anarchic", False):
                social_vec = self.mix_by_mask(normal_social, random_social, self._single_mask(dim))
            else:
                social_vec = normal_social

            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseAnarchicPSO"


class SparseAmnesiacPSO(SingleObjectivePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Amnesiac PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            termination_criterion: TerminationCriterion,
            w: float,
            c1: float,
            c2: float,
            random_strength: float,
            amnesiac_fraction: float,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.random_strength = random_strength
        self.amnesiac_fraction = max(0.0, min(1.0, amnesiac_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.amnesiac_fraction, "is_amnesiac")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            normal_cognitive = self.c1 * r1 * (p_best - current)
            random_cognitive = self.random_strength * np.random.uniform(-1.0, 1.0, dim)
            if particle.attributes.get("is_amnesiac", False):
                cognitive_vec = self.mix_by_mask(normal_cognitive, random_cognitive, self._single_mask(dim))
            else:
                cognitive_vec = normal_cognitive

            social_vec = self.c2 * r2 * (g_best - current)
            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseAmnesiacPSO"


class SparseAnarchicAmnesiacPSO(SingleObjectivePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Anarchic-Amnesiac PSO with disjoint role marking."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            termination_criterion: TerminationCriterion,
            w: float,
            c1: float,
            c2: float,
            random_strength_social: float,
            random_strength_cognitive: float,
            anarchic_fraction: float,
            amnesiac_fraction: float,
            social_coordinate_mode: str = "sqrt",
            cognitive_coordinate_mode: str = "sqrt",
            social_coordinate_fraction: float = 0.1,
            cognitive_coordinate_fraction: float = 0.1,
            social_coordinate_scale: float = 1.0,
            cognitive_coordinate_scale: float = 1.0,
            social_coordinate_count: int = 10,
            cognitive_coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.random_strength_social = random_strength_social
        self.random_strength_cognitive = random_strength_cognitive
        self.anarchic_fraction = max(0.0, min(1.0, anarchic_fraction))
        self.amnesiac_fraction = max(0.0, min(1.0, amnesiac_fraction))
        self._init_component_coordinate_params(
            social_coordinate_mode,
            cognitive_coordinate_mode,
            social_coordinate_fraction,
            cognitive_coordinate_fraction,
            social_coordinate_scale,
            cognitive_coordinate_scale,
            social_coordinate_count,
            cognitive_coordinate_count,
        )

    def _mark_special_particles(self, swarm: List[S]) -> None:
        self.mark_particles(swarm, self.anarchic_fraction, "is_anarchic")

        total = len(swarm)
        desired_amnesiacs = max(1, int(total * self.amnesiac_fraction))
        num_anarchics = sum(1 for p in swarm if p.attributes.get("is_anarchic", False))
        non_anarchic_count = total - num_anarchics
        effective_fraction = desired_amnesiacs / non_anarchic_count if non_anarchic_count > 0 else 1.0
        effective_fraction = min(1.0, effective_fraction)

        non_anarchics = [p for p in swarm if not p.attributes.get("is_anarchic", False)]
        self.mark_particles(non_anarchics, effective_fraction, "is_amnesiac")

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self._mark_special_particles(solutions)
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            normal_cognitive = self.c1 * r1 * (p_best - current)
            random_cognitive = self.random_strength_cognitive * np.random.uniform(-1.0, 1.0, dim)
            if particle.attributes.get("is_amnesiac", False):
                cognitive_vec = self.mix_by_mask(normal_cognitive, random_cognitive, self._cognitive_mask(dim))
            else:
                cognitive_vec = normal_cognitive

            normal_social = self.c2 * r2 * (g_best - current)
            random_social = self.random_strength_social * np.random.uniform(-1.0, 1.0, dim)
            if particle.attributes.get("is_anarchic", False):
                social_vec = self.mix_by_mask(normal_social, random_social, self._social_mask(dim))
            else:
                social_vec = normal_social

            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseAnarchicAmnesiacPSO"


class SparseErraticPSO(SingleObjectivePSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Erratic PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            termination_criterion: TerminationCriterion,
            w: float,
            c1: float,
            c2: float,
            random_strength: float,
            erratic_fraction: float,
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion, constraint_handling_mode)
        self.random_strength = random_strength
        self.erratic_fraction = max(0.0, min(1.0, erratic_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.erratic_fraction, "is_erratic")
        return solutions

    def update_velocity(self, swarm: List[S]) -> None:
        g_best = np.array(self.best_global.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])
            p_best = np.array(particle.attributes["best_position"])

            r1 = random.random()
            r2 = random.random()

            standard_velocity = (
                self.w * velocity
                + self.c1 * r1 * (p_best - current)
                + self.c2 * r2 * (g_best - current)
            )
            erratic_velocity = self.w * velocity + self.random_strength * np.random.uniform(-1.0, 1.0, dim)

            if particle.attributes.get("is_erratic", False):
                new_velocity = self.mix_by_mask(standard_velocity, erratic_velocity, self._single_mask(dim))
            else:
                new_velocity = standard_velocity

            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseErraticPSO"


class SparseDrifterPSO(PerturbationPSO, RoleMixin, SparseRoleMixin):
    """Coordinate-wise Drifter PSO."""

    def __init__(
            self,
            problem: FloatProblem,
            swarm_size: int,
            termination_criterion: TerminationCriterion,
            w: float,
            c1: float,
            c2: float,
            drifter_fraction: float,
            perturbation_scale: float,
            perturbation_method: str = "gaussian",
            coordinate_mode: str = "sqrt",
            coordinate_fraction: float = 0.1,
            coordinate_scale: float = 1.0,
            coordinate_count: int = 10,
            constraint_handling_mode: str = "clip",
    ):
        super().__init__(
            problem,
            swarm_size,
            c1,
            c2,
            w,
            termination_criterion,
            constraint_handling_mode,
            perturbation_method,
            perturbation_scale,
        )
        self.drifter_fraction = max(0.0, min(1.0, drifter_fraction))
        self._init_single_coordinate_params(coordinate_mode, coordinate_fraction, coordinate_scale, coordinate_count)

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.drifter_fraction, "is_drifter")
        return solutions

    def perturbation(self, swarm: List[S]) -> None:
        lower_bound = np.array(self.problem.lower_bound)
        upper_bound = np.array(self.problem.upper_bound)
        dim = self.problem.number_of_variables()

        for particle in [p for p in swarm if p.attributes.get("is_drifter", False)]:
            pos = np.array(particle.variables)
            velocity = np.array(particle.attributes["velocity"])

            if self.perturbation_method.lower() == "gaussian":
                noise = np.random.normal(loc=0.0, scale=self.perturbation_scale, size=pos.shape)
            elif self.perturbation_method.lower() == "cauchy":
                noise = np.random.standard_cauchy(size=pos.shape) * self.perturbation_scale
            else:
                raise ValueError("Unknown perturbation method: choose 'gaussian' or 'cauchy'")

            sparse_noise = np.where(self._single_mask(dim), noise, 0.0)
            new_pos = pos + sparse_noise
            new_pos, new_velocity = self.handle_constraints(new_pos, velocity, lower_bound, upper_bound)

            particle.variables = new_pos.tolist()
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseDrifterPSO"
