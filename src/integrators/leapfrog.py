from src.physics.system import compute_accelerations


def leapfrog_integrator(bodies, dt):
    """
    Perform a single Leapfrog (kick-drift-kick) integration step.

    Leapfrog is a second-order symplectic integrator commonly used for
    Hamiltonian systems. Although less accurate per step than RK4, it
    typically exhibits superior long-term conservation of energy and
    phase-space structure.

    Parameters
    ----------
    bodies : list[Body]
        Current system state. Positions and velocities are updated in place.
    dt : float
        Time step size.

    Returns
    -------
    None
        The input bodies are mutated in place.
    """
    # Compute accelerations based on current positions
    accelerations = compute_accelerations(bodies)

    # Kick: advance velocities by half a time step
    for body, acceleration in zip(bodies, accelerations):
        body.velocity += 0.5 * acceleration * dt  # Update velocity at half step

    # Drift: advance positions using half-step velocities
    for body in bodies:
        body.position += (
            body.velocity * dt
        )  # Update position, using velocity at half step

    # Recompute accelerations based on new positions
    new_accelerations = compute_accelerations(bodies)

    # Final kick: complete the velocity update
    for body, new_acceleration in zip(bodies, new_accelerations):
        body.velocity += 0.5 * new_acceleration * dt  # Update velocity at full step
