import numpy as np
from scipy.spatial.distance import pdist
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class LennardJonesMinimumEnergyCluster(FloatProblem):
    """
    Lennard-Jones cluster potential in reduced (r_min-normalized) units:

        f(x) = sum_{i<j} [ r_ij^-12 - 2 * r_ij^-6 ]

    i.e. the pairwise well depth is 1 and the pair minimum is -1 at distance
    1. This is the convention of the CEC-2011 real-world suite (Das &
    Suganthan, problem 2) and of the Cambridge Cluster Database (CCD), so
    objective values are DIRECTLY comparable to published cluster energies
    (no offset is applied).

    Variables encode atom coordinates as (x1,y1,z1, x2,y2,z2, ...), so only
    D = 3N is meaningful; with n variables the problem uses N = n // 3 atoms
    and IGNORES the n % 3 leftover variables (dead dimensions). Prefer
    dimensions divisible by 3.

    Bounds: each coordinate is confined to [-L, L] with L = 2 * N^(1/3).
    Rationale: optimal LJ_N clusters have roughly unit density (radius
    ~0.6 * N^(1/3), cf. Wales & Doye 1997 / CCD structures), so this box
    contains the global minimum with ample margin while keeping random
    initializations within interaction range; cluster-optimization practice
    confines atoms to similarly N-scaled containers (CEC-2011 uses tight
    atom-indexed boxes). Overly wide boxes make the landscape numerically
    flat: at the previously used [-80, 80] a random 333-atom configuration
    had ZERO interacting pairs.

    Reference global minima (CCD, unit = pair well depth; Wales, Doye et al.,
    https://www-wales.ch.cam.ac.uk/~jon/structures/LJ/tables.150.html):
        N=2: -1.0        N=3: -3.0        N=4: -6.0
        N=5: -9.103852   N=6: -12.712062  N=7: -16.505384
        N=13: -44.326801 N=38: -173.928427
        N=100: -557.039820  N=150: -893.310258
    """

    def __init__(self, number_of_variables: int = 10):
        super(LennardJonesMinimumEnergyCluster, self).__init__()
        self._number_of_variables = number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        # Only D = 3N is meaningful (see class docstring); the box scale is
        # derived from the atom count, not from the raw variable count.
        n_atoms = max(1, number_of_variables // 3)
        half_width = 2.0 * n_atoms ** (1.0 / 3.0)
        self.lower_bound = [-half_width for _ in range(number_of_variables)]
        self.upper_bound = [half_width for _ in range(number_of_variables)]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        num_points = self.number_of_variables() // 3
        coords = np.asarray(
            solution.variables[: 3 * num_points], dtype=float
        ).reshape(num_points, 3)

        # Pairwise squared distances over the upper triangle (i < j);
        # d6 = r^6, as in the pairwise potential r^-12 - 2*r^-6.
        # (Plain multiplications instead of ** and a single reciprocal: the
        # libm pow/extra divisions dominate the runtime otherwise.)
        r2 = pdist(coords, 'sqeuclidean')
        d6 = r2 * r2 * r2
        # NUMERICAL SAFEGUARD, not part of the standard LJ definition: caps
        # the repulsion of (near-)coincident atoms (r^6 < 1e-6, i.e.
        # r < ~0.1) to avoid division by zero / inf propagation. Irrelevant
        # anywhere near optima.
        np.maximum(d6, 1e-6, out=d6)

        np.reciprocal(d6, out=d6)  # d6 is now r^-6
        energy = float(d6 @ d6) - 2.0 * float(np.sum(d6))
        solution.objectives[0] = energy
        return solution

    def name(self) -> str:
        return 'Lennard-Jones Minimum Energy Cluster'
