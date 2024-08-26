from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import gp_minimize
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator import PolynomialMutation, SBXCrossover

from algorithm.PGPHEA import PGPHEA
from algorithm.PGSHEA import PGSHEA

# Define the space of parameters to search
space = [
    Real(0.1, 4, name='c1'),
    Real(0.1, 4, name='c2'),
    Real(0.1, 4, name='w'),
    Real(1e-3, 7, "log-uniform", name='mutation_factor'),
    Real(0.5, 1.0, name='crossover_rate'),
    Integer(1, 100, name='exchange_interval'),
    Integer(1, 100, name='exchange_number'),
    # Categorical(['PSO', 'GA'], name='starting_algorithm')
]


@use_named_args(space)
def objective(c1, c2, w, mutation_factor, crossover_rate, exchange_interval, exchange_number):
    problem = Rastrigin(100)
    mutation = PolynomialMutation(mutation_factor / problem.number_of_variables(), 20.0)
    crossover = SBXCrossover(crossover_rate, 5.0)

    algorithm = PGPHEA(
        problem=problem,
        solutions_size=100,
        c1=c1,
        c2=c2,
        w=w,
        mutation=mutation,
        crossover=crossover,
        exchange_interval=exchange_interval,
        exchange_number=exchange_number,
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
