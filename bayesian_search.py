from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from SingleObjectivePSO import SingleObjectivePSO

# Define the space of parameters to search
space = [Real(0.01, 2.5, name='c1'),
         Real(0.01, 2.5, name='c2'),
         Real(0.01, 0.99, name='w')]


@use_named_args(space)
def objective(c1, c2, w):
    problem = Rastrigin(100)
    algorithm = SingleObjectivePSO(
        problem=problem,
        swarm_size=100,
        c1=c1,
        c2=c2,
        w=w,
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
    )

    algorithm.run()
    result = algorithm.result()

    # Here we assume that the objective to minimize is the first (and only) objective
    return result.objectives[0]


# Perform optimization
res_gp = gp_minimize(objective, space, n_calls=30, random_state=0)

print("Best parameters: {}".format(res_gp.x))
print("Best objective: {}".format(res_gp.fun))
