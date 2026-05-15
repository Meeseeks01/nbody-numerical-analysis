
import numpy as np
from src.physics.body import Body

def bodies_to_state_vector(bodies):
    """
    Convert a list of Body objects to a state vector for the dynamical system.
    The state vector will contain the positions and velocities of all bodies in a flattened format.

    Parameters:
    - bodies: A list of Body objects.

    Returns:
    - state_vector: A numpy array containing the positions and velocities of all bodies.
    """
    state_vector = []
    for body in bodies:
        state_vector.extend(body.position)
        state_vector.extend(body.velocity)
    return np.array(state_vector)

def state_vector_to_bodies(state_vector, masses):
    """
    Convert a state vector back to a list of Body objects.
    The state vector is expected to contain the positions and velocities of all bodies in a flattened format.

    Parameters:
    - state_vector: A numpy array containing the positions and velocities of all bodies.
    - masses: A list of masses corresponding to each body.

    Returns:
    - bodies: A list of Body objects reconstructed from the state vector.
    """
    bodies = []
    num_bodies = len(masses)
    for i in range(num_bodies):
        position = state_vector[4*i:4*i+2]  # Extract position (2D)
        velocity = state_vector[4*i+2:4*i+4]  # Extract velocity (2D)
        body = Body(mass=masses[i], position=position, velocity=velocity)
        bodies.append(body)
    return bodies



def compute_lyapunov_exponent(bodies, integrator, dt, num_steps, epsilon=1e-8, seed=None):
    """
    Estimate the maximal Lyapunov exponent of an N-body system
    using the Benettin algorithm.

    This method evolves two nearby trajectories:
    the original system and a perturbed copy.
    The separation between the trajectories is measured
    after each integration step and periodically renormalised
    to maintain a small perturbation magnitude.

    Parameters
    ----------
    bodies : list[Body]
    Initial list of Body objects representing the system.

    integrator : callable
    Numerical integrator function used to evolve the system.
    The integrator must follow the signature:
    integrator(bodies, dt)

    dt : float
    Simulation time step.

    num_steps : int
    Number of integration steps used for the estimation.

    epsilon : float, optional
    Initial perturbation magnitude used for trajectory separation.
    Default is 1e-8.

    Returns
    -------
    float
    Estimated maximal Lyapunov exponent per unit time.

    Raises
    ------
    ValueError
    If no valid perturbation growth steps are obtained.
    """
  

    # Initialize variables
    masses = [body.mass for body in bodies]
    state = bodies_to_state_vector(bodies)

    lyapunov_sum = 0.0 # To accumulate the logarithm of the growth factors
    valid_steps = 0 # To count the number of valid steps where the perturbation does not collapse to zero

    # Generate a random number generator for reproducibility
    rng = np.random.default_rng(seed)
    perturbation = rng.random(state.shape)
    perturbation = epsilon * perturbation / np.linalg.norm(perturbation)

    # Evolve the system and the perturbed copy for a number of steps
    for _ in range(num_steps):
        normal_bodies = state_vector_to_bodies(state, masses)
        perturbed_bodies = state_vector_to_bodies(state + perturbation, masses)

        integrator(normal_bodies, dt)
        integrator(perturbed_bodies, dt)

        next_state = bodies_to_state_vector(normal_bodies)
        next_perturbed_state = bodies_to_state_vector(perturbed_bodies)

        next_perturbation = next_perturbed_state - next_state
        norm = np.linalg.norm(next_perturbation)
        # If the perturbation has collapsed to zero, we cannot compute the growth factor for this step
        if norm == 0:
            continue

        lyapunov_sum += np.log(norm / epsilon)

        state = next_state
        perturbation = epsilon * next_perturbation / norm

        valid_steps += 1

    if valid_steps == 0:
            # If no valid steps were taken, we cannot compute the Lyapunov exponent
            raise ValueError("No valid steps taken during Lyapunov exponent computation. " \
                             "Perturbation may have collapsed to zero.")
    
    # Compute the average Lyapunov exponent
    lyapunov_exponent = lyapunov_sum / (valid_steps * dt)
    
    return lyapunov_exponent




   