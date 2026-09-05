"""Every factory must construct, and every algorithm with grouped role
fractions must carry normalized values.

Regression guard: a factory once transcribed raw (un-normalized) tuned
fractions summing to 2.13 and crashed in the full-disjoint fraction guard;
the partial-disjoint classes would instead silently cap over-1.0 groups
during assignment, making the running configuration differ from the
factory's nominal constants.
"""

import inspect

import pytest
from jmetal.problem import Sphere

import experiment.factories as factories

ALL_FACTORIES = sorted(
    name for name, fn in vars(factories).items()
    if name.startswith("factory_") and inspect.isfunction(fn)
)

TOLERANCE = 1e-9


@pytest.mark.parametrize("factory_name", ALL_FACTORIES)
def test_factory_constructs_with_normalized_fraction_groups(factory_name):
    algorithm = getattr(factories, factory_name)(Sphere(3))
    assert algorithm is not None

    # Full-disjoint family: ALL special fractions share one budget.
    if hasattr(algorithm, "special_role_fractions"):
        total = sum(algorithm.special_role_fractions.values())
        assert total <= 1.0 + TOLERANCE, (
            f"{factory_name}: special role fractions sum to {total:.4f} > 1.0"
        )
        if hasattr(algorithm, "standard_fraction"):
            assert -TOLERANCE <= algorithm.standard_fraction <= 1.0 + TOLERANCE

    # Partial-disjoint family: cognitive and social groups have separate
    # budgets; over-1.0 groups would be silently capped during assignment.
    for attr, label in (
        ("cognitive_role_fractions_input", "cognitive"),
        ("social_role_fractions_input", "social"),
    ):
        if hasattr(algorithm, attr):
            total = sum(getattr(algorithm, attr).values())
            assert total <= 1.0 + TOLERANCE, (
                f"{factory_name}: {label} role fractions sum to {total:.4f} > 1.0 "
                f"(would be silently capped at runtime)"
            )
