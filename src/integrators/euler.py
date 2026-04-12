
from src.physics.system import compute_accelerations

def euler_integrator(bodies, dt):
    """
    Perform a single Euler integration step.

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
        body.position += body.velocity * dt  # Update position
        body.velocity += acceleration * dt  # Update velocity
       

    
