from src.physics.system import compute_accelerations


def euler_integrator(bodies, dt):
    """
    Perform a single Euler integration step.
    Implements the standard Euler method :
    r(t + dt) = r(t) + v(t) * dt
    v(t + dt) = v(t) + a(t) * dt
    where r is position, v is velocity, a is acceleration, and dt is the time step.
    First-order method, O(dt) accuracy, not symplectic, can lead to energy drift over time.

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

    # Update velocities and positions using Euler's method

    for body, acceleration in zip(bodies, accelerations):
        body.position += (
            body.velocity * dt
        )  # Update position -> r(t + dt) = r(t) + v(t) * dt
        body.velocity += (
            acceleration * dt
        )  # Update velocity -> v(t + dt) = v(t) + a(t) * dt
