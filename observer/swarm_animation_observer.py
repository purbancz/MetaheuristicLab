import numpy as np
from jmetal.core.observer import Observer
from jmetal.core.solution import FloatSolution # Import base class if needed

class SwarmAnimationObserver(Observer):
    """
    Observes a swarm's position and velocity, ensuring data robustness
    for animation, capturing only the first two dimensions.
    """
    def __init__(self, capture_interval=1, num_dimensions_to_capture=2):
        self.frames = []           # List to store position arrays for each frame
        self.velocities = []       # List to store velocity arrays for each frame
        self.capture_interval = capture_interval
        self.counter = 0
        self.num_dimensions = num_dimensions_to_capture
        self._expected_vars = None # To store expected variable count from the problem if available

    def update(self, *args, **kwargs):
        self.counter += 1
        if self.counter % self.capture_interval != 0:
            return # Skip update if not on the capture interval

        swarm = kwargs.get("SWARM", [])
        problem = kwargs.get("PROBLEM", None)

        # Try to get expected dimensions from the problem once
        if problem and self._expected_vars is None:
             try:
                 self._expected_vars = problem.number_of_variables()
             except AttributeError:
                 # print("Observer Warning: Could not get number_of_variables from problem.")
                 self._expected_vars = -1 # Indicate it's unknown

        current_positions = []
        current_velocities = []

        if not isinstance(swarm, list) or not swarm:
            # Append empty arrays to maintain frame count consistency
            self.frames.append(np.empty((0, self.num_dimensions)))
            self.velocities.append(np.empty((0, self.num_dimensions)))
            return # Skip processing empty swarm

        particle_count = len(swarm)
        pos_fallback_value = [0.0] * self.num_dimensions
        vel_fallback_value = [0.0] * self.num_dimensions

        for i, particle in enumerate(swarm):
            # --- Process Position ---
            pos = pos_fallback_value
            if hasattr(particle, 'variables') and isinstance(particle.variables, (list, np.ndarray)):
                if len(particle.variables) >= self.num_dimensions:
                    try:
                        pos = np.array(particle.variables[:self.num_dimensions], dtype=float).tolist()
                        if not np.all(np.isfinite(pos)):
                             # print(f"Observer Warning: Particle {i} has non-finite position {pos}. Using fallback.")
                             pos = pos_fallback_value
                    except (ValueError, TypeError):
                         # print(f"Observer Warning: Particle {i} has problematic position {particle.variables}. Using fallback.")
                         pos = pos_fallback_value

            current_positions.append(pos)

            # --- Process Velocity ---
            vel = vel_fallback_value
            if hasattr(particle, 'attributes') and isinstance(particle.attributes, dict) and 'velocity' in particle.attributes:
                raw_vel = particle.attributes['velocity']
                if isinstance(raw_vel, (list, np.ndarray)):
                     if len(raw_vel) >= self.num_dimensions:
                         try:
                             vel = np.array(raw_vel[:self.num_dimensions], dtype=float).tolist()
                             if not np.all(np.isfinite(vel)):
                                  # print(f"Observer Warning: Particle {i} has non-finite velocity {vel}. Using fallback.")
                                  vel = vel_fallback_value
                         except (ValueError, TypeError):
                              # print(f"Observer Warning: Particle {i} has problematic velocity {raw_vel}. Using fallback.")
                              vel = vel_fallback_value

            current_velocities.append(vel)

        # Convert lists of lists/arrays to 2D numpy arrays
        try:
            final_positions = np.array(current_positions, dtype=float)
            final_velocities = np.array(current_velocities, dtype=float)
        except ValueError as e:
             print(f"Observer Error: Could not convert captured data to NumPy array: {e}. Storing empty.")
             final_positions = np.empty((0, self.num_dimensions))
             final_velocities = np.empty((0, self.num_dimensions))


        # Final sanity check on shapes
        if final_positions.shape != (particle_count, self.num_dimensions):
             print(f"Observer Error: Final positions shape mismatch! Expected {(particle_count, self.num_dimensions)}, Got {final_positions.shape}. Storing empty.")
             final_positions = np.empty((0, self.num_dimensions))
             # Ensure velocities are also empty if positions failed, for consistency
             final_velocities = np.empty((0, self.num_dimensions))

        elif final_velocities.shape != (particle_count, self.num_dimensions):
             print(f"Observer Error: Final velocities shape mismatch! Expected {(particle_count, self.num_dimensions)}, Got {final_velocities.shape}. Storing empty.")
             final_velocities = np.empty((0, self.num_dimensions))
              # Ensure positions are also empty if velocities failed, for consistency
             final_positions = np.empty((0, self.num_dimensions))


        self.frames.append(final_positions)
        self.velocities.append(final_velocities)