from skopt.space import Real, Categorical, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import PolynomialMutation, SBXCrossover

from PGSHEA import PGSHEA

# Define the space of parameters to search
space = [
    Real(2, 4, name='c1'),
    Real(0.2, 0.4, name='c2'),
    Real(0.14, 0.22, name='w'),
    # Real(1e-5, 1.0, "log-uniform", name='mutation_rate'),
    # Real(0.5, 1.0, name='crossover_rate'),
    # Integer(10, 100, name='swap_limit'),
    # Categorical(['PSO', 'GA'], name='starting_algorithm')
]


@use_named_args(space)
def objective(c1, c2, w):
    problem = Rastrigin(100)
    mutation = PolynomialMutation(1.0 / problem.number_of_variables(), 20.0)
    crossover = SBXCrossover(1, 5.0)

    algorithm = PGSHEA(
        problem=problem,
        solutions_size=100,
        c1=c1,
        c2=c2,
        w=w,
        mutation=mutation,
        crossover=crossover,
        swap_limit=66,
        starting_algorithm='PSO',
        termination_criterion=StoppingByEvaluations(max_evaluations=25000)
    )

    algorithm.run()
    result = algorithm.result()

    # Minimize the objective function
    return result.objectives[0]


# Perform optimization
res_gp = gp_minimize(objective, space, n_calls=30, random_state=0)

print("Best parameters:", res_gp.x)
print("Best objective:", res_gp.fun)
