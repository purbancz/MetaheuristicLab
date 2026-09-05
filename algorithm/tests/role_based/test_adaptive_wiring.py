"""The adaptive variants must actually adapt.

RRAPSO/CDAPSO/EEAPSO/DAPSO initialize AdaptiveRoleMixin state; these tests
pin that adapt_parameters really runs during optimization: inertia moves
away from base_inertia and role fractions grow toward their maxima when the
thresholds demand it.
"""

import pytest
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.role_based.roles import RRAPSO, CDAPSO, EEAPSO, DAPSO, DrifterPSO


ADAPTIVE_KWARGS = dict(
    swarm_size=6,
    c1=1.5,
    c2=1.5,
    base_inertia=0.5,
    min_inertia=0.1,
    max_inertia=0.9,
    window_size=5,
    # diversity is normalized (uniform swarm ~0.29), so 0.99 forces the
    # "low diversity" branch every iteration -> inertia climbs.
    diversity_threshold=0.99,
    # an unreachable improvement rate forces role fractions to grow.
    improvement_threshold=10.0,
)


def _run(algorithm):
    algorithm.run()
    return algorithm


@pytest.mark.parametrize(
    "cls,extra",
    [
        (RRAPSO, dict(ac1=1.0, ac2=1.0, rebel_fraction=0.2, rejector_fraction=0.2,
                      max_rebel_fraction=0.8, max_rejector_fraction=0.8)),
        (CDAPSO, dict(ac1=1.0, ac2=1.0, contrarian_fraction=0.2, defeatist_fraction=0.2,
                      max_contrarian_fraction=0.8, max_defeatist_fraction=0.8)),
        (EEAPSO, dict(ac1=1.0, ac2=1.0, eschewer_fraction=0.2, escapist_fraction=0.2,
                      max_eschewer_fraction=0.8, max_escapist_fraction=0.8)),
        (DAPSO, dict(perturbation_scale=0.05, drifter_fraction=0.2,
                     max_drifter_fraction=0.8)),
    ],
)
def test_adaptive_variants_actually_adapt(cls, extra):
    algorithm = cls(
        problem=Sphere(5),
        termination_criterion=StoppingByEvaluations(max_evaluations=120),
        **ADAPTIVE_KWARGS,
        **extra,
    )
    _run(algorithm)

    # Inertia adapted away from its starting value (toward max_inertia).
    assert algorithm.w > ADAPTIVE_KWARGS["base_inertia"]

    # Every role fraction grew from its original value (improvement rate is
    # always below the unreachable threshold).
    for flag, original in algorithm.original_fractions.items():
        assert algorithm.role_fractions[flag] > original, flag

    # The swarm marking follows the adapted fractions.
    for flag, frac in algorithm.role_fractions.items():
        marked = sum(1 for p in algorithm.solutions if p.attributes.get(flag, False))
        assert marked == max(1, int(len(algorithm.solutions) * frac)), flag


def test_adaptation_can_be_frozen_by_thresholds():
    # With an always-satisfied improvement threshold and an always-high
    # diversity branch, parameters must move the OTHER way (down to originals
    # / min inertia), proving the branches are both live.
    algorithm = RRAPSO(
        problem=Sphere(5),
        termination_criterion=StoppingByEvaluations(max_evaluations=120),
        swarm_size=6,
        c1=1.5,
        c2=1.5,
        ac1=1.0,
        ac2=1.0,
        base_inertia=0.5,
        min_inertia=0.1,
        max_inertia=0.9,
        rebel_fraction=0.2,
        rejector_fraction=0.2,
        max_rebel_fraction=0.8,
        max_rejector_fraction=0.8,
        window_size=5,
        diversity_threshold=0.0,       # never "low diversity" -> inertia shrinks
        improvement_threshold=-10.0,   # always "improving" -> fractions stay at originals
    )
    algorithm.run()

    assert algorithm.w < 0.5
    for flag, original in algorithm.original_fractions.items():
        assert algorithm.role_fractions[flag] == pytest.approx(original)


def test_drifter_pso_forwards_perturbation_method():
    algorithm = DrifterPSO(
        problem=Sphere(3),
        swarm_size=4,
        termination_criterion=StoppingByEvaluations(max_evaluations=8),
        w=0.5,
        c1=1.5,
        c2=1.5,
        drifter_fraction=0.5,
        perturbation_scale=0.05,
        perturbation_method="cauchy",
    )
    assert algorithm.perturbation_method == "cauchy"
    assert algorithm.perturbation_scale == 0.05
