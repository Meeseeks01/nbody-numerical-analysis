import numpy as np
from src.physics.body import Body


def bodies_to_state_vector(bodies):
    """
    Convert a list of Body objects to a flat state vector.
    """
    state_vector = []

    for body in bodies:
        state_vector.extend(np.asarray(body.position, dtype=float))
        state_vector.extend(np.asarray(body.velocity, dtype=float))

    return np.array(state_vector, dtype=float)


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
        position = state_vector[4 * i : 4 * i + 2].copy()  # Extract position (2D)
        velocity = state_vector[4 * i + 2 : 4 * i + 4].copy()  # Extract velocity (2D)
        body = Body(mass=masses[i], position=position, velocity=velocity)
        bodies.append(body)
    return bodies


def compute_lyapunov_exponent(
    bodies,
    integrator,
    dt,
    num_steps=200000,
    epsilon=1e-8,
    seed=None,
    renormalization_steps=50,
):
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

    lyapunov_sum = 0.0  # To accumulate the logarithm of the growth factors
    valid_steps = 0  # To count the number of valid steps where the perturbation does not collapse to zero

    rng = np.random.default_rng(seed)
    perturbation = rng.random(state.shape)
    perturbation = epsilon * perturbation / np.linalg.norm(perturbation)

    # Evolve the system and the perturbed copy for a number of steps
    for step in range(1, num_steps + 1):

        normal_bodies = state_vector_to_bodies(state, masses)
        perturbed_bodies = state_vector_to_bodies(state + perturbation, masses)

        integrator(normal_bodies, dt)
        integrator(perturbed_bodies, dt)

        next_state = bodies_to_state_vector(normal_bodies)
        next_perturbed_state = bodies_to_state_vector(perturbed_bodies)

        next_perturbation = next_perturbed_state - next_state

        state = next_state
        perturbation = next_perturbation

        if step % renormalization_steps == 0:
            norm = np.linalg.norm(perturbation)

            if norm == 0:
                continue

            lyapunov_sum += np.log(norm / epsilon)

            perturbation = epsilon * perturbation / norm

            valid_steps += 1

    if valid_steps == 0:
        # If no valid steps were taken, we cannot compute the Lyapunov exponent
        raise ValueError(
            "No valid steps taken during Lyapunov exponent computation. "
            "Perturbation may have collapsed to zero."
        )

    # Compute the average Lyapunov exponent
    lyapunov_exponent = lyapunov_sum / (valid_steps * renormalization_steps * dt)

    return lyapunov_exponent
