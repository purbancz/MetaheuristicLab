import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import deepcopy

from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO
from algorithm.single_objective_PSO import SingleObjectivePSO
from observer.swarm_animation_observer import SwarmAnimationObserver
from problem.n_variables.plateau import Plateau

OUTPUT_DIR = "frames"


def get_heatmap_data(func, resolution=100):
    x_min, x_max = func.lower_bound[0], func.upper_bound[0]
    y_min, y_max = func.lower_bound[1], func.upper_bound[1]
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    d = func.number_of_variables()
    fixed = []
    for i in range(2, d):
        midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
        fixed.append(midpoint)

    for i in range(resolution):
        for j in range(resolution):
            variables = [X[i, j], Y[i, j]] + fixed
            sol = deepcopy(func.create_solution())
            sol.variables = variables
            func.evaluate(sol)
            Z[i, j] = sol.objectives[0]

    return X, Y, Z


def animate_swarm(func, observer):
    X, Y, Z = get_heatmap_data(func, resolution=150)

    fig, ax = plt.subplots(figsize=(12, 10))
    heatmap = ax.contourf(X, Y, Z, levels=50, cmap='viridis')
    plt.colorbar(heatmap, ax=ax)
    scatter = ax.scatter([], [], c='orange', s=30)
    ax.set_title(f"Swarm Animation of {algorithm.get_name()} on {func.name()}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.tight_layout()
    func_name = func.name().replace(' ', '_').replace('-', '_')

    should_save_frames = False

    def init():
        animation_dir = os.path.join(OUTPUT_DIR, f"{algorithm.get_name()}_{func_name}")
        os.makedirs(animation_dir, exist_ok=True)
        scatter.set_offsets(np.empty((0, 2)))
        return scatter,

    def update(frame_idx):
        positions = observer.frames[frame_idx]
        scatter.set_offsets(positions)
        ax.set_title(f"{algorithm.get_name()} on {func.name()}: Iteration {frame_idx * observer.capture_interval:03d}")
        if should_save_frames:
            if frame_idx == 0 or frame_idx % 10 == 0 or frame_idx == len(observer.frames) - 1:
                filename = os.path.join(OUTPUT_DIR, f"{algorithm.get_name()}_{func_name}", f"frame_{frame_idx:04d}.png")
                plt.savefig(filename, dpi=300)
                print(f"Saved {filename}")
        return scatter,

    anim = animation.FuncAnimation(
        fig,
        update,
        frames=len(observer.frames),
        init_func=init,
        interval=200,  # czas pomiędzy klatkami w ms
        blit=True
    )
    should_save_frames = True
    anim.save(f"{algorithm.get_name()}_{func_name}_swarm_animation.mp4", writer='ffmpeg', fps=10)
    # should_save_frames = False
    # anim.save(f"{algorithm.get_name()}_{func_name}_swarm_animation.gif", writer='imagemagick', fps=10)
    return anim


if __name__ == "__main__":
    problem = Rastrigin(number_of_variables=2)

    max_evaluations = 4501

    algorithm = SingleObjectivePSO(
        problem=problem,
        swarm_size=30,
        c1=2,
        c2=2,
        w=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
    )

    observer = SwarmAnimationObserver(capture_interval=1)
    algorithm.observable.register(observer)

    algorithm.run()
    animate_swarm(problem, observer)
