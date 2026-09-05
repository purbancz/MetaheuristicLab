"""CAPSO/IAPSO adaptation must react to actual pbest improvements.

The old condition compared objectives against a personal best that had
already absorbed them, so the "improved" branch was unreachable and the
coefficients drifted monotonically to their extremes.
"""

import pytest
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.role_based.adaptive_pso import CoAdaptativePSO, IndividualAdaptivePSO


def _make(cls, max_evaluations=8):
    return cls(
        problem=Sphere(3),
        swarm_size=4,
        c1=1.0,
        c2=1.0,
        max_c1=3.0,
        max_c2=3.0,
        w=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )


def _particle(algorithm, objective, best_objective):
    p = algorithm.problem.create_solution()
    p.objectives[0] = objective
    p.attributes['best_objective'] = best_objective
    p.attributes['best_position'] = list(p.variables)
    return p


def test_update_particle_best_records_improvement_flag():
    algorithm = _make(CoAdaptativePSO)
    improved = _particle(algorithm, objective=1.0, best_objective=5.0)
    stagnant = _particle(algorithm, objective=7.0, best_objective=5.0)

    algorithm.update_particle_best([improved, stagnant])

    assert improved.attributes['improved_last_iteration'] is True
    assert improved.attributes['best_objective'] == 1.0
    assert stagnant.attributes['improved_last_iteration'] is False
    assert stagnant.attributes['best_objective'] == 5.0


def test_capso_improvement_branch_is_reachable():
    algorithm = _make(CoAdaptativePSO)
    improved = _particle(algorithm, 1.0, 5.0)
    algorithm.update_particle_best([improved])
    algorithm.solutions = [improved]

    algorithm.update_coefficient()

    # The (previously dead) improvement branch: c1 up, c2 down/at floor.
    assert algorithm.c1 == pytest.approx(1.1)
    assert algorithm.c2 == pytest.approx(1.0)  # clamped at min_c2

    # And the stagnation branch still works.
    improved.attributes['improved_last_iteration'] = False
    algorithm.update_coefficient()
    assert algorithm.c2 == pytest.approx(1.1)


def test_iapso_adapts_per_particle():
    algorithm = _make(IndividualAdaptivePSO)
    improved = _particle(algorithm, 1.0, 5.0)
    stagnant = _particle(algorithm, 7.0, 5.0)
    for p in (improved, stagnant):
        p.attributes['c1'] = 1.0
        p.attributes['c2'] = 1.0

    algorithm.update_particle_best([improved, stagnant])
    algorithm.solutions = [improved, stagnant]
    algorithm.update_coefficient()

    assert improved.attributes['c1'] == pytest.approx(1.1)
    assert improved.attributes['c2'] == pytest.approx(1.0)
    assert stagnant.attributes['c1'] == pytest.approx(1.0)
    assert stagnant.attributes['c2'] == pytest.approx(1.1)


@pytest.mark.parametrize("cls", [CoAdaptativePSO, IndividualAdaptivePSO])
def test_full_run_keeps_coefficients_within_bounds(cls):
    algorithm = _make(cls, max_evaluations=40)
    algorithm.run()

    if cls is CoAdaptativePSO:
        assert algorithm.min_c1 <= algorithm.c1 <= algorithm.max_c1
        assert algorithm.min_c2 <= algorithm.c2 <= algorithm.max_c2
    else:
        for p in algorithm.solutions:
            assert algorithm.min_c1 <= p.attributes['c1'] <= algorithm.max_c1
            assert algorithm.min_c2 <= p.attributes['c2'] <= algorithm.max_c2
