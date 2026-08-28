import hashlib

import numpy as np


def derive_problem_seed(base_seed: int, problem_class, number_of_variables: int,) -> int:
    """
    Derive a stable problem-instance seed from the base seed,
    problem class and dimensionality.

    The result does not depend on Python's randomized hash().
    """
    identity = f"{base_seed}:" f"{problem_class.__module__}." f"{problem_class.__qualname__}:" f"{number_of_variables}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()

    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def problem_instance_fingerprint(problem) -> str | None:
    """
    Create a fingerprint of the actual randomized benchmark instance.

    Random static benchmark components currently used in the repository
    are shift vectors and rotation matrices.
    """
    hasher = hashlib.sha256()
    hasher.update(f"{problem.__class__.__module__}." f"{problem.__class__.__qualname__}".encode("utf-8"))
    hasher.update(str(problem.number_of_variables()).encode("utf-8"))

    found_random_component = False

    for attribute_name in ("shift", "rotation_matrix"):
        if not hasattr(problem, attribute_name):
            continue

        found_random_component = True

        array = np.ascontiguousarray(np.asarray(getattr(problem, attribute_name), dtype=np.float64))
        hasher.update(attribute_name.encode("utf-8"))
        hasher.update(str(array.shape).encode("utf-8"))
        hasher.update(array.tobytes())

    if not found_random_component:
        return None

    return hasher.hexdigest()


def create_seeded_problem(problem_class, number_of_variables: int, base_seed: int):
    """
    Create a deterministic randomized benchmark instance without
    changing the global NumPy RNG state seen by the algorithms.
    """
    instance_seed = derive_problem_seed(base_seed, problem_class, number_of_variables)
    global_rng_state = np.random.get_state()

    try:
        np.random.seed(instance_seed)
        problem = problem_class(number_of_variables)
    finally:
        np.random.set_state(global_rng_state)

    problem.instance_seed = instance_seed
    problem.instance_id = (problem_instance_fingerprint(problem))

    return problem