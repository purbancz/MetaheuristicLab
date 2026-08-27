import numpy as np

from experiment.problem_identity import (
    create_seeded_problem,
)

from problem.n_variables.CEC import (
    RotatedBentCigar,
    ShiftedRotatedHappyCat,
)


def test_same_problem_seed_creates_same_instance():
    first = create_seeded_problem(
        ShiftedRotatedHappyCat,
        5,
        12345,
    )

    second = create_seeded_problem(
        ShiftedRotatedHappyCat,
        5,
        12345,
    )

    assert first.instance_seed == second.instance_seed
    assert first.instance_id == second.instance_id

    np.testing.assert_array_equal(
        first.shift,
        second.shift,
    )

    np.testing.assert_array_equal(
        first.rotation_matrix,
        second.rotation_matrix,
    )


def test_different_base_seed_creates_different_instance():
    first = create_seeded_problem(
        ShiftedRotatedHappyCat,
        5,
        12345,
    )

    second = create_seeded_problem(
        ShiftedRotatedHappyCat,
        5,
        54321,
    )

    assert first.instance_seed != second.instance_seed
    assert first.instance_id != second.instance_id


def test_different_dimensions_create_different_instances():
    first = create_seeded_problem(
        RotatedBentCigar,
        5,
        12345,
    )

    second = create_seeded_problem(
        RotatedBentCigar,
        10,
        12345,
    )

    assert first.instance_seed != second.instance_seed
    assert first.instance_id != second.instance_id


def test_problem_creation_does_not_consume_global_numpy_rng():
    np.random.seed(123)

    expected = np.random.random(5)

    np.random.seed(123)

    create_seeded_problem(
        ShiftedRotatedHappyCat,
        5,
        999,
    )

    actual = np.random.random(5)

    np.testing.assert_array_equal(
        actual,
        expected,
    )