from src.physics.system import compute_accelerations
from src.physics.body import Body


def rk4_integrator(bodies, dt):
    """
    Perform a single Runge-Kutta 4th order (RK4) integration step.
    RK4 is a higher-order method that provides better accuracy than Euler's method.
    It evaluates intermediate slopes with local truncation error O(dt^5) and global error O(dt^4).

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
    # Stage 1 - k1
    k1_accelerations = compute_accelerations(bodies)

    # Stage 2 - k2 bodies use k1 velocity and acceleration
    k2_bodies = []
    for body, a1 in zip(bodies, k1_accelerations):
        new_pos = body.position + 0.5 * body.velocity * dt
        new_vel = body.velocity + 0.5 * a1 * dt
        k2_bodies.append(Body(body.mass, new_pos, new_vel))
    k2_accelerations = compute_accelerations(k2_bodies)

    # Stage 3 - k3 bodies use k2 velocity and acceleration
    k3_bodies = []
    for body, k2_body, a2 in zip(bodies, k2_bodies, k2_accelerations):
        new_pos = body.position + 0.5 * k2_body.velocity * dt
        new_vel = body.velocity + 0.5 * a2 * dt
        k3_bodies.append(Body(body.mass, new_pos, new_vel))
    k3_accelerations = compute_accelerations(k3_bodies)

    # Stage 4 - k4 bodies use k3 velocity and acceleration
    k4_bodies = []
    for body, k3_body, a3 in zip(bodies, k3_bodies, k3_accelerations):
        new_pos = body.position + k3_body.velocity * dt
        new_vel = body.velocity + a3 * dt
        k4_bodies.append(Body(body.mass, new_pos, new_vel))
    k4_accelerations = compute_accelerations(k4_bodies)

    # Update positions and velocities using the RK4 formula
    for i, body in enumerate(bodies):
        a1 = k1_accelerations[i]
        a2 = k2_accelerations[i]
        a3 = k3_accelerations[i]
        a4 = k4_accelerations[i]

        body.position += (
            body.velocity
            + 2 * (k2_bodies[i].velocity + k3_bodies[i].velocity)
            + k4_bodies[i].velocity
        ) * (dt / 6)
        body.velocity += (a1 + 2 * (a2 + a3) + a4) * (dt / 6)
