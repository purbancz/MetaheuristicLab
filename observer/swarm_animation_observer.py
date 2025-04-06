import numpy as np
from jmetal.core.observer import Observer

class SwarmAnimationObserver(Observer):
    def __init__(self, capture_interval=1):
        self.frames = []
        self.capture_interval = capture_interval
        self.counter = 0

    def update(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.capture_interval == 0:
            swarm = kwargs.get("SWARM", [])
            positions = [particle.variables[:2] for particle in swarm]
            self.frames.append(np.array(positions))
