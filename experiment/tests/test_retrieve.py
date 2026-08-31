import numpy as np

from experiment.aggregation import combine_data


def test_match_problem_rebuilds_other_dimensions(monkeypatch):
    # Imported inside the test: importing experiment.retrieve runs
    # setup_experiment() at module import (a few seconds).
    import experiment.retrieve as retrieve
    from jmetal.problem import Sphere

    from experiment.problem_identity import create_seeded_problem
    from problem.n_variables.CEC import RotatedBentCigar

    monkeypatch.setattr(retrieve, "problems", [Sphere(10)])

    exact = retrieve.match_problem("Sphere", 10)
    assert exact.number_of_variables() == 10

    rebuilt = retrieve.match_problem("Sphere", 5)
    assert rebuilt is not None
    assert rebuilt.number_of_variables() == 5

    assert retrieve.match_problem("NoSuchProblem", 5) is None

    seeded = create_seeded_problem(RotatedBentCigar, 10, 42)
    monkeypatch.setattr(retrieve, "problems", [seeded])

    rebuilt_seeded = retrieve.match_problem(seeded.name(), 6)
    reference = create_seeded_problem(RotatedBentCigar, 6, 42)
    assert rebuilt_seeded.number_of_variables() == 6
    assert rebuilt_seeded.instance_id == reference.instance_id


def make_result(
    problem,
    n_vars,
    values,
    algorithm="PSO",
):
    data = np.asarray(
        values,
        dtype=float,
    ).reshape(-1, 1)

    return {
        "problem": problem,
        "n_vars": n_vars,
        "results": {
            algorithm: {
                "data": data,
                "avg_fitness": float(np.mean(data)),
                "std_dev": float(np.std(data)),
                "avg_time": 1.0,
            }
        },
    }


def test_same_problem_and_dimension_are_combined():
    first = make_result(
        "Rastrigin",
        100,
        [1.0, 2.0],
    )

    second = make_result(
        "Rastrigin",
        100,
        [3.0, 4.0],
    )

    combined = combine_data(
        [first, second]
    )

    assert len(combined) == 1
    assert ("Rastrigin", 100) in combined

    instance = combined[
        ("Rastrigin", 100)
    ]

    assert instance["problem"] == "Rastrigin"
    assert instance["n_vars"] == 100

    np.testing.assert_array_equal(
        instance["results"]["PSO"]["data"],
        np.array([
            [1.0],
            [2.0],
            [3.0],
            [4.0],
        ]),
    )


def test_same_problem_different_dimensions_are_separate():
    dim_100 = make_result(
        "Rastrigin",
        100,
        [1.0, 2.0],
    )

    dim_500 = make_result(
        "Rastrigin",
        500,
        [3.0, 4.0],
    )

    dim_1000 = make_result(
        "Rastrigin",
        1000,
        [5.0, 6.0],
    )

    combined = combine_data([
        dim_100,
        dim_500,
        dim_1000,
    ])

    assert len(combined) == 3

    assert ("Rastrigin", 100) in combined
    assert ("Rastrigin", 500) in combined
    assert ("Rastrigin", 1000) in combined

    np.testing.assert_array_equal(
        combined[
            ("Rastrigin", 100)
        ]["results"]["PSO"]["data"],
        np.array([
            [1.0],
            [2.0],
        ]),
    )

    np.testing.assert_array_equal(
        combined[
            ("Rastrigin", 500)
        ]["results"]["PSO"]["data"],
        np.array([
            [3.0],
            [4.0],
        ]),
    )

    np.testing.assert_array_equal(
        combined[
            ("Rastrigin", 1000)
        ]["results"]["PSO"]["data"],
        np.array([
            [5.0],
            [6.0],
        ]),
    )


def test_different_problems_same_dimension_are_separate():
    rastrigin = make_result(
        "Rastrigin",
        100,
        [1.0],
    )

    sphere = make_result(
        "Sphere",
        100,
        [2.0],
    )

    combined = combine_data([
        rastrigin,
        sphere,
    ])

    assert len(combined) == 2

    assert ("Rastrigin", 100) in combined
    assert ("Sphere", 100) in combined


def test_problem_metadata_is_preserved():
    result = make_result(
        "Rastrigin",
        500,
        [1.0, 2.0],
    )

    combined = combine_data([result])

    instance = combined[
        ("Rastrigin", 500)
    ]

    assert instance["problem"] == "Rastrigin"
    assert instance["n_vars"] == 500

def test_run_count_is_stored_per_problem_dimension():
    dim_100 = make_result(
        "Rastrigin",
        100,
        [1.0, 2.0],
    )

    dim_500 = make_result(
        "Rastrigin",
        500,
        [3.0, 4.0, 5.0],
    )

    combined = combine_data([
        dim_100,
        dim_500,
    ])

    assert combined[
        ("Rastrigin", 100)
    ]["runs"] == 2

    assert combined[
        ("Rastrigin", 500)
    ]["runs"] == 3

    assert combined[
        ("Rastrigin", 100)
    ]["results"]["PSO"]["runs"] == 2

    assert combined[
        ("Rastrigin", 500)
    ]["results"]["PSO"]["runs"] == 3