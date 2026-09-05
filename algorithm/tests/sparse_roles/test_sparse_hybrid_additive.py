"""SparseHybridAdditivePSO velocity semantics (module contract):

special role contributions apply only on their coordinate masks, and
unmasked coordinates keep standard PSO behavior - a special-role particle
with the std flag off must NOT drift on inertia alone off-mask.
"""

import numpy as np
import pytest
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

import algorithm.sparse_roles.sparse_hybrid as sparse_hybrid
from algorithm.sparse_roles.sparse_hybrid import SparseHybridAdditivePSO

DIM = 4
MASK = np.array([True, True, False, False])


def _make():
    return SparseHybridAdditivePSO(
        problem=Sphere(DIM),
        swarm_size=4,
        termination_criterion=StoppingByEvaluations(max_evaluations=8),
        w=0.5,
        c1=1.0,
        c2=2.0,
        rejector_c=3.0,
        rejector_prob=0.5,
        std_cognitive_prob=0.5,
        std_social_prob=0.5,
    )


def _particle(algorithm, flags):
    p = algorithm.problem.create_solution()
    p.variables = [1.0, 1.0, 1.0, 1.0]
    p.objectives[0] = 4.0
    p.attributes["velocity"] = [0.0] * DIM
    p.attributes["best_position"] = [0.0] * DIM
    p.attributes["worst_position"] = [2.0] * DIM
    for flag in algorithm.coefficients:
        p.attributes[flag] = flag in flags
    return p


def _run_update(algorithm, particle, monkeypatch):
    monkeypatch.setattr(sparse_hybrid.random, "random", lambda: 1.0)
    monkeypatch.setattr(algorithm, "_cognitive_mask", lambda dim: MASK.copy())
    monkeypatch.setattr(algorithm, "_social_mask", lambda dim: MASK.copy())

    best = algorithm.problem.create_solution()
    best.variables = [0.0] * DIM
    best.objectives[0] = 0.0
    worst = algorithm.problem.create_solution()
    worst.variables = [2.0] * DIM
    worst.objectives[0] = 16.0
    algorithm.best_global = best
    algorithm.global_worst = worst

    algorithm.update_velocity([particle])
    return np.array(particle.attributes["velocity"])


def test_special_only_particle_keeps_standard_behavior_off_mask(monkeypatch):
    algorithm = _make()
    particle = _particle(algorithm, flags={"is_rejector"})

    velocity = _run_update(algorithm, particle, monkeypatch)

    # current=1, p_best=0, g_best=0. On masked coords: rejector 3*(1-0)=3,
    # no standard cognitive; on unmasked coords: standard cognitive
    # 1*(0-1)=-1. Standard social flag off and no special social is active,
    # so the fallback applies the full standard social 2*(0-1)=-2 everywhere.
    expected = np.where(MASK, 3.0, -1.0) + (-2.0)
    np.testing.assert_allclose(velocity, expected)


def test_std_flag_on_keeps_dense_additive_semantics(monkeypatch):
    algorithm = _make()
    particle = _particle(algorithm, flags={"is_rejector", "is_std_cognitive", "is_std_social"})

    velocity = _run_update(algorithm, particle, monkeypatch)

    # Standard cognitive applies on ALL coordinates (-1), rejector adds +3 on
    # masked coordinates, standard social applies everywhere (-2).
    expected = np.where(MASK, 3.0 - 1.0, -1.0) + (-2.0)
    np.testing.assert_allclose(velocity, expected)


def test_no_roles_falls_back_to_full_standard(monkeypatch):
    algorithm = _make()
    particle = _particle(algorithm, flags=set())

    velocity = _run_update(algorithm, particle, monkeypatch)

    np.testing.assert_allclose(velocity, np.full(DIM, -1.0 - 2.0))


def test_no_coordinate_is_ever_attractor_free(monkeypatch):
    # The defect this file guards against: with the std flag off, off-mask
    # coordinates previously received ZERO cognitive and social input.
    algorithm = _make()
    particle = _particle(algorithm, flags={"is_rejector", "is_rebel"})

    velocity = _run_update(algorithm, particle, monkeypatch)

    # Off-mask coords: standard cognitive (-1) + standard social (-2) = -3;
    # masked coords: rejector +3 and rebel 1*(1-0)... rebel_c defaults.
    assert not np.any(velocity == 0.0)
    np.testing.assert_allclose(velocity[~MASK], -3.0)
