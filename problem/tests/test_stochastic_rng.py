import numpy as np

from jmetal.core.solution import FloatSolution

from problem.n_variables.stochastic import Stochastic


def test_stochastic_does_not_consume_global_numpy_rng():
    problem = Stochastic(
        number_of_variables=3,
        seed=999,
    )

    solution = FloatSolution(problem.lower_bound, problem.upper_bound, 1)
    solution.variables = [1.0, 2.0, 3.0]

    np.random.seed(12345)
    expected = np.random.random(5)

    np.random.seed(12345)

    problem.evaluate(solution)

    actual = np.random.random(5)

    np.testing.assert_array_equal(
        actual,
        expected,
    )


def test_stochastic_same_seed_produces_same_noise():
    first = Stochastic(3, seed=123)
    second = Stochastic(3, seed=123)

    solution_1 = FloatSolution(
        first.lower_bound,
        first.upper_bound,
        1,
    )
    solution_1.variables = [1.0, 2.0, 3.0]

    solution_2 = FloatSolution(
        second.lower_bound,
        second.upper_bound,
        1,
    )
    solution_2.variables = [1.0, 2.0, 3.0]

    first.evaluate(solution_1)
    second.evaluate(solution_2)

    assert (
        solution_1.objectives[0]
        == solution_2.objectives[0]
    )