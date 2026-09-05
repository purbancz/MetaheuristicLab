"""Sparse hybrid role PSO variants.

Sparse hybrid role PSO applies special cognitive/social role components only
to selected coordinates, while unmasked coordinates keep standard PSO behavior.
"""

from __future__ import annotations

import random
from typing import List, TypeVar

import numpy as np

from algorithm.role_based.role_hybrids import HybridAdditivePSO, HybridFullDisjointPSO, HybridPartialDisjointPSO
from algorithm.sparse_roles.coordinate_mask_utilities import SparseCoordinateMixin

S = TypeVar("S")


class SparseHybridPartialDisjointPSO(HybridPartialDisjointPSO, SparseCoordinateMixin):
    """Coordinate-wise version of HybridPartialDisjointPSO."""

    def __init__(
            self,
            *args,
            cognitive_coordinate_mode: str = "sqrt",
            social_coordinate_mode: str = "sqrt",
            cognitive_coordinate_fraction: float = 0.1,
            social_coordinate_fraction: float = 0.1,
            cognitive_coordinate_scale: float = 1.0,
            social_coordinate_scale: float = 1.0,
            cognitive_coordinate_count: int = 10,
            social_coordinate_count: int = 10,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
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

    def update_velocity(self, swarm: List[S]) -> None:
        if self.best_global is None or self.global_worst is None or not swarm:
            return

        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            attrs = particle.attributes
            required_attrs = ["velocity", "best_position", "worst_position", "cognitive_role", "social_role"]
            if not all(attr in attrs for attr in required_attrs):
                continue

            current = np.array(particle.variables)
            velocity = np.array(attrs["velocity"])
            p_best = np.array(attrs["best_position"])
            p_worst = np.array(attrs["worst_position"])

            cognitive_role = attrs["cognitive_role"]
            social_role = attrs["social_role"]

            r1 = random.random()
            r2 = random.random()

            standard_cognitive = self.c1 * r1 * (p_best - current)
            if cognitive_role == "rejector":
                special_cognitive = self.rejector_c * r1 * (current - p_best)
            elif cognitive_role == "defeatist":
                special_cognitive = self.defeatist_c * r1 * (p_worst - current)
            elif cognitive_role == "escapist":
                special_cognitive = self.escapist_c * r1 * (current - p_worst)
            else:
                special_cognitive = standard_cognitive

            if cognitive_role == "standard":
                cognitive_vec = standard_cognitive
            else:
                cognitive_vec = self.mix_by_mask(standard_cognitive, special_cognitive, self._cognitive_mask(dim))

            standard_social = self.c2 * r2 * (g_best - current)
            if social_role == "rebel":
                special_social = self.rebel_c * r2 * (current - g_best)
            elif social_role == "contrarian":
                special_social = self.contrarian_c * r2 * (g_worst - current)
            elif social_role == "eschewer":
                special_social = self.eschewer_c * r2 * (current - g_worst)
            else:
                special_social = standard_social

            if social_role == "standard":
                social_vec = standard_social
            else:
                social_vec = self.mix_by_mask(standard_social, special_social, self._social_mask(dim))

            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseHybridPartialDisjointPSO"


class SparseHybridFullDisjointPSO(HybridFullDisjointPSO, SparseCoordinateMixin):
    """Coordinate-wise version of HybridFullDisjointPSO."""

    def __init__(
            self,
            *args,
            cognitive_coordinate_mode: str = "sqrt",
            social_coordinate_mode: str = "sqrt",
            cognitive_coordinate_fraction: float = 0.1,
            social_coordinate_fraction: float = 0.1,
            cognitive_coordinate_scale: float = 1.0,
            social_coordinate_scale: float = 1.0,
            cognitive_coordinate_count: int = 10,
            social_coordinate_count: int = 10,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
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

    def update_velocity(self, swarm: List[S]) -> None:
        if self.best_global is None or self.global_worst is None or not swarm:
            return

        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            attrs = particle.attributes
            required_attrs = ["velocity", "best_position", "worst_position", "assigned_role"]
            if not all(attr in attrs for attr in required_attrs):
                continue

            current = np.array(particle.variables)
            velocity = np.array(attrs["velocity"])
            p_best = np.array(attrs["best_position"])
            p_worst = np.array(attrs["worst_position"])
            assigned_role = attrs["assigned_role"]

            r1 = random.random()
            r2 = random.random()

            standard_cognitive = self.coefficients["std_cognitive"] * r1 * (p_best - current)
            standard_social = self.coefficients["std_social"] * r2 * (g_best - current)

            cognitive_vec = standard_cognitive
            social_vec = standard_social

            if assigned_role in self.special_cognitive_roles:
                if assigned_role == "rejector":
                    special_cognitive = self.coefficients["rejector"] * r1 * (current - p_best)
                elif assigned_role == "defeatist":
                    special_cognitive = self.coefficients["defeatist"] * r1 * (p_worst - current)
                elif assigned_role == "escapist":
                    special_cognitive = self.coefficients["escapist"] * r1 * (current - p_worst)
                else:
                    special_cognitive = standard_cognitive

                cognitive_vec = self.mix_by_mask(standard_cognitive, special_cognitive, self._cognitive_mask(dim))

            elif assigned_role in self.special_social_roles:
                if assigned_role == "rebel":
                    special_social = self.coefficients["rebel"] * r2 * (current - g_best)
                elif assigned_role == "contrarian":
                    special_social = self.coefficients["contrarian"] * r2 * (g_worst - current)
                elif assigned_role == "eschewer":
                    special_social = self.coefficients["eschewer"] * r2 * (current - g_worst)
                else:
                    special_social = standard_social

                social_vec = self.mix_by_mask(standard_social, special_social, self._social_mask(dim))

            new_velocity = self.w * velocity + cognitive_vec + social_vec
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseHybridFullDisjointPSO"


class SparseHybridAdditivePSO(HybridAdditivePSO, SparseCoordinateMixin):
    """Coordinate-wise version of HybridAdditivePSO.

    Special role contributions apply only on their sampled coordinate masks.
    The standard component applies to ALL coordinates when the particle's
    is_std_* flag is on (dense additive semantics); when the flag is off, the
    standard component still applies on every coordinate NOT covered by any
    active special mask, so unmasked coordinates keep standard PSO behavior
    (module contract) instead of drifting on inertia alone.
    """

    def __init__(
            self,
            *args,
            cognitive_coordinate_mode: str = "sqrt",
            social_coordinate_mode: str = "sqrt",
            cognitive_coordinate_fraction: float = 0.1,
            social_coordinate_fraction: float = 0.1,
            cognitive_coordinate_scale: float = 1.0,
            social_coordinate_scale: float = 1.0,
            cognitive_coordinate_count: int = 10,
            social_coordinate_count: int = 10,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)
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

    def update_velocity(self, swarm: List[S]) -> None:
        if self.best_global is None or self.global_worst is None or not swarm:
            return

        g_best = np.array(self.best_global.variables)
        g_worst = np.array(self.global_worst.variables)
        dim = self.problem.number_of_variables()

        for particle in swarm:
            attrs = particle.attributes
            required_core_attrs = ["velocity", "best_position", "worst_position"]
            if not all(attr in attrs for attr in required_core_attrs):
                continue

            for flag_name in self.coefficients.keys():
                if flag_name not in attrs:
                    attrs[flag_name] = False

            current = np.array(particle.variables)
            velocity = np.array(attrs["velocity"])
            p_best = np.array(attrs["best_position"])
            p_worst = np.array(attrs["worst_position"])

            cognitive_component = np.zeros_like(current)
            social_component = np.zeros_like(current)
            rand_factors = {flag: random.random() for flag in self.coefficients}

            cognitive_special_union = np.zeros(dim, dtype=bool)
            any_special_cognitive_active = False

            if attrs.get("is_rejector", False):
                vec = self.coefficients["is_rejector"] * rand_factors["is_rejector"] * (current - p_best)
                mask = self._cognitive_mask(dim)
                cognitive_component += np.where(mask, vec, 0.0)
                cognitive_special_union |= mask
                any_special_cognitive_active = True

            if attrs.get("is_defeatist", False):
                vec = self.coefficients["is_defeatist"] * rand_factors["is_defeatist"] * (p_worst - current)
                mask = self._cognitive_mask(dim)
                cognitive_component += np.where(mask, vec, 0.0)
                cognitive_special_union |= mask
                any_special_cognitive_active = True

            if attrs.get("is_escapist", False):
                vec = self.coefficients["is_escapist"] * rand_factors["is_escapist"] * (current - p_worst)
                mask = self._cognitive_mask(dim)
                cognitive_component += np.where(mask, vec, 0.0)
                cognitive_special_union |= mask
                any_special_cognitive_active = True

            standard_cognitive = (
                self.coefficients["is_std_cognitive"]
                * rand_factors["is_std_cognitive"]
                * (p_best - current)
            )
            if attrs.get("is_std_cognitive", False) or not any_special_cognitive_active:
                cognitive_component += standard_cognitive
            else:
                # Unmasked coordinates keep standard PSO behavior.
                cognitive_component += np.where(cognitive_special_union, 0.0, standard_cognitive)

            social_special_union = np.zeros(dim, dtype=bool)
            any_special_social_active = False

            if attrs.get("is_rebel", False):
                vec = self.coefficients["is_rebel"] * rand_factors["is_rebel"] * (current - g_best)
                mask = self._social_mask(dim)
                social_component += np.where(mask, vec, 0.0)
                social_special_union |= mask
                any_special_social_active = True

            if attrs.get("is_contrarian", False):
                vec = self.coefficients["is_contrarian"] * rand_factors["is_contrarian"] * (g_worst - current)
                mask = self._social_mask(dim)
                social_component += np.where(mask, vec, 0.0)
                social_special_union |= mask
                any_special_social_active = True

            if attrs.get("is_eschewer", False):
                vec = self.coefficients["is_eschewer"] * rand_factors["is_eschewer"] * (current - g_worst)
                mask = self._social_mask(dim)
                social_component += np.where(mask, vec, 0.0)
                social_special_union |= mask
                any_special_social_active = True

            standard_social = (
                self.coefficients["is_std_social"]
                * rand_factors["is_std_social"]
                * (g_best - current)
            )
            if attrs.get("is_std_social", False) or not any_special_social_active:
                social_component += standard_social
            else:
                # Unmasked coordinates keep standard PSO behavior.
                social_component += np.where(social_special_union, 0.0, standard_social)

            new_velocity = self.w * velocity + cognitive_component + social_component
            particle.attributes["velocity"] = new_velocity.tolist()

    def get_name(self) -> str:
        return "SparseHybridAdditivePSO"
