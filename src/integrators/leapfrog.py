from src.physics.system import compute_accelerations


def leapfrog_integrator(bodies, dt):
    """
    Perform a single Leapfrog integration step.

    Parameters
    ----------
    bodies : list[Body]
        Current system state.
    dt : float
        Time step.

    Returns
    -------
    None
        Updates body positions and velocities in place.
    """
    # Compute accelerations based on current positions
    accelerations = compute_accelerations(bodies)

    # Update velocities at half step
    for body, acceleration in zip(bodies, accelerations):
        body.velocity += 0.5 * acceleration * dt  # Update velocity at half step

    # Update positions using updated velocities
    for body in bodies:
        body.position += (
            body.velocity * dt
        )  # Update position, using velocity at half step

    # Recompute accelerations based on new positions
    new_accelerations = compute_accelerations(bodies)

    # Update velocities at full step using new accelerations
    for body, new_acceleration in zip(bodies, new_accelerations):
        body.velocity += 0.5 * new_acceleration * dt  # Update velocity at full step
