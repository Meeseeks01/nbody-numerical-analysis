from src.physics.body import Body
from src.integrators.euler import euler_integrator
import numpy as np


def test_euler_integrator_updates_position_without_force():
    body = Body(1.0, [0, 0], [1, 0])
    bodies = [body]

    euler_integrator(bodies, dt=1.0)

    assert np.allclose(bodies[0].position, np.array([1.0, 0.0]))


def test_euler_integrator_keeps_velocity_constant_without_force():
    body = Body(1.0, [0, 0], [1, 0])
    bodies = [body]

    euler_integrator(bodies, dt=1.0)

    assert np.allclose(bodies[0].velocity, np.array([1.0, 0.0]))
    
def test_leapfrog_integrator_updates_position_without_force():
    body = Body(1.0, [0, 0], [1, 0])
    bodies = [body]

    from src.integrators.leapfrog import leapfrog_integrator
    leapfrog_integrator(bodies, dt=1.0)

    assert np.allclose(bodies[0].position, np.array([1.0, 0.0]))