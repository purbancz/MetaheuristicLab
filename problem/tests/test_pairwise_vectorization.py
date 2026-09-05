"""Certification of the vectorized pairwise-family benchmark problems.

Each reference below is the original pure-Python loop, kept verbatim. The
vectorized evaluates must agree to tight tolerance across dimensions, and
the seeded ShiftedRotatedSchafferF7 instance identity must be untouched.
"""

import math

import numpy as np
import pytest
from jmetal.core.solution import FloatSolution

from experiment.problem_identity import create_seeded_problem
from problem.n_variables.CEC import ShiftedRotatedSchafferF7
from problem.n_variables.eggholder import EggHolder
from problem.n_variables.expanded_schaffer import ExpandedShaffer
from problem.n_variables.schaffer import (
    GeneralizedSchafferN1,
    GeneralizedSchafferN2,
    GeneralizedSchafferN3,
    GeneralizedSchafferN4,
)
from problem.n_variables.schmidt_vetters import GeneralizedSchmidtVetters
from problem.n_variables.shubert import ShubertN3, ShubertN4
from problem.n_variables.sine_envelope import SineEnvelope
from problem.n_variables.strechedv import StretchedV


def _pairs(x):
    return zip(x[:-1], x[1:])


def ref_schaffer_n1(problem, x):
    total = 0.0
    for a, b in _pairs(x):
        total += 0.5 + (math.sin(a**2 - b**2) ** 2 - 0.5) / (1 + 0.001 * (a**2 + b**2)) ** 2
    return total


def ref_schaffer_n2(problem, x):
    n = len(x)
    total = 0.0
    for a, b in _pairs(x):
        total += 0.5 + (math.cos(math.sin(abs(a**2 - b**2))) - 0.5) / (1 + 0.001 * (a**2 + b**2)) ** 2
    return -(total - (n - 1))


def ref_schaffer_n3(problem, x):
    total = 0.0
    for a, b in _pairs(x):
        r_sq = a**2 + b**2
        total += (r_sq**0.25) * (1 + math.sin(50 * (r_sq**0.1)) ** 2)
    return total


def ref_schaffer_n4(problem, x):
    n = len(x)
    total = 0.0
    for a, b in _pairs(x):
        total += 0.5 + (math.cos(math.sin(abs(a**2 - b**2))) - 0.5) / (1 + 0.001 * (a**2 + b**2) ** 2) ** 2
    return -(total - (n - 1))


def ref_expanded_schaffer(problem, x):
    n = len(x)
    s = n / 2.0
    for i in range(n):
        a, b = x[i], x[(i + 1) % n]
        s += (math.sin(math.sqrt(a**2 + b**2)) ** 2 - 0.5) / (1 + 0.001 * (a**2 + b**2)) ** 2
    return s


def ref_shubert_n3(problem, x):
    return sum(sum(j * math.sin((j + 1) * xd) + j for j in range(1, 6)) for xd in x)


def ref_shubert_n4(problem, x):
    return sum(sum(j * math.cos((j + 1) * xd) + j for j in range(1, 6)) for xd in x)


def ref_schmidt_vetters(problem, x):
    total = 0.0
    for a, b in _pairs(x):
        num = math.sin(a**2 - b**2) ** 2 + math.cos(a**2 + b**2) ** 2 - 1
        total += num / (1 + 0.001 * (a**2 + b**2)) ** 2
    return total + (len(x) - 1) * 0.75


def ref_eggholder(problem, x):
    total = 0.0
    for a, b in _pairs(x):
        total += -a * math.sin(math.sqrt(abs(a - b - 47)))
        total += -(b + 47) * math.sin(math.sqrt(abs(0.5 * a + b + 47)))
    return total


def ref_sine_envelope(problem, x):
    total = 0.0
    for a, b in _pairs(x):
        r2 = a**2 + b**2
        total += (math.sin(math.sqrt(r2) - 0.5) ** 2) / ((0.001 * r2 + 1) ** 2) + 0.5
    return -total


def ref_stretched_v(problem, x):
    total = 0.0
    for a, b in _pairs(x):
        t = a**2 + b**2
        total += (t ** (1 / 4)) * (math.sin(50 * (t**0.1)) + 1) ** 2
    return total


def ref_schaffer_f7(problem, x):
    z = np.dot(problem.rotation_matrix, np.array(x) - problem.shift)
    z = 10 * z
    d = problem.number_of_variables()
    s = np.array([z[i] ** 2 + z[i + 1] ** 2 for i in range(d - 1)])
    inner = np.sum(s + s * (np.sin(50 * (s ** (1 / 5))) ** 2)) / (d - 1)
    return inner**2


CASES = [
    (GeneralizedSchafferN1, ref_schaffer_n1),
    (GeneralizedSchafferN2, ref_schaffer_n2),
    (GeneralizedSchafferN3, ref_schaffer_n3),
    (GeneralizedSchafferN4, ref_schaffer_n4),
    (ExpandedShaffer, ref_expanded_schaffer),
    (ShubertN3, ref_shubert_n3),
    (ShubertN4, ref_shubert_n4),
    (GeneralizedSchmidtVetters, ref_schmidt_vetters),
    (EggHolder, ref_eggholder),
    (SineEnvelope, ref_sine_envelope),
    (StretchedV, ref_stretched_v),
]


def _evaluate(problem, variables):
    s = FloatSolution(problem.lower_bound, problem.upper_bound, 1, 0)
    s.objectives = [0.0]
    s.variables = list(variables)
    return problem.evaluate(s).objectives[0]


@pytest.mark.parametrize("cls,reference", CASES, ids=lambda c: getattr(c, "__name__", ""))
@pytest.mark.parametrize("dim", [2, 10, 1000])
def test_vectorized_matches_loop_reference(cls, reference, dim):
    problem = cls(number_of_variables=dim)
    rng = np.random.default_rng(dim)
    lower = np.asarray(problem.lower_bound)
    upper = np.asarray(problem.upper_bound)
    for _ in range(3):
        x = list(rng.uniform(lower, upper))
        expected = reference(problem, x)
        assert _evaluate(problem, x) == pytest.approx(expected, rel=1e-12, abs=1e-9)


@pytest.mark.parametrize("dim", [10, 1000])
def test_schaffer_f7_vectorized_matches_loop_reference(dim):
    problem = create_seeded_problem(ShiftedRotatedSchafferF7, dim, 42)
    rng = np.random.default_rng(dim)
    for _ in range(3):
        x = list(rng.uniform(-100.0, 100.0, size=dim))
        expected = ref_schaffer_f7(problem, x)
        assert _evaluate(problem, x) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_schaffer_f7_seeded_instance_identity_is_preserved():
    # Fingerprints captured on the pre-vectorization implementation.
    expected = {
        10: "2098c8dedc550d2a8128f010ecd1c341ce6a135c09d3903dfab1c99d3c203935",
        100: "aea3c9e3a40345f3868146539d47ac985c71ae74f9f573f6853c5d4eb7f9f189",
    }
    for dim, fingerprint in expected.items():
        problem = create_seeded_problem(ShiftedRotatedSchafferF7, dim, 42)
        assert problem.instance_id == fingerprint
