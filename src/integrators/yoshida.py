from src.physics.system import compute_accelerations


def yoshida_integrator(bodies, dt):
    """
    Perform a single 4th-order symplectic Yoshida integration step.

    Uses a 7-stage DKDKDKD (drift-kick) decomposition that preserves the
    symplectic structure of Hamiltonian systems, ensuring long-term energy
    conservation without numerical dissipation.

    Parameters
    ----------
    bodies : list[Body]
        Current system state. Positions and velocities are updated in place.
    dt : float
        Time step size in simulation units.

    Note
    ----
    Mutates `bodies` in place; no new list is returned.

    Warns
    -----
    No stability check is performed; caller is responsible for choosing
    a sufficiently small `dt` relative to the shortest orbital period.

    References
    ----------
    Yoshida, H. (1990). Construction of higher order symplectic integrators.
    Physics Letters A, 150(5-7), 262-268.
    """

    # Coefficients for the Yoshida integrator (4th order symplectic integrator)
    # W0 and W1 are derived from the requirement of symplecticity and time-reversibility
    w1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
    w0 = -(2.0 ** (1.0 / 3.0)) * w1

    # c1, c2, c3, c4 are the position update coefficients
    c1 = w1 / 2  
    c2 = (w0 + w1) / 2  
    c3 = c2  
    c4 = c1  

    # d1, d2, d3 are the velocity update coefficients
    d1 = w1  
    d2 = w0  
    d3 = w1  

    # --- Stage 1: Drift ---
    for body in bodies:
        body.position += c1 * body.velocity * dt

    # --- Stage 2: Kick ---
    accelerations = compute_accelerations(bodies)

    for body, acc in zip(bodies, accelerations):
        body.velocity += d1 * acc * dt

    # --- Stage 3: Drift ---
    for body in bodies:
        body.position += c2 * body.velocity * dt

    # --- Stage 4: Kick ---
    accelerations_1 = compute_accelerations(bodies)

    for body, acc in zip(bodies, accelerations_1):
        body.velocity += d2 * acc * dt

    # --- Stage 5: Drift ---

    for body in bodies:
        body.position += c3 * body.velocity * dt

    # --- Stage 6: Kick ---
    accelerations_2 = compute_accelerations(bodies)

    for body, acc in zip(bodies, accelerations_2):
        body.velocity += d3 * acc * dt
    # --- Stage 7: Drift ---

    for body in bodies:
        body.position += c4 * body.velocity * dt
