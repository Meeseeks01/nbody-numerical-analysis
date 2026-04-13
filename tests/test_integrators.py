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

def test_leapfrog_outperforms_euler_energy_conservation():
    # Euler bodies
    Euler_body = [Body(1.0, [0, 0], [0, 1]),
                  Body(1.0, [1, 0], [0, -1])]
    # Leapfrog bodies - identical BUT independent instances to avoid coupling
    Leapfrog_body = [Body(1.0, [0, 0], [0, 1]),
                     Body(1.0, [1, 0], [0, -1])]
    

    from src.integrators.leapfrog import leapfrog_integrator
    from src.physics.system import compute_total_energy

    initial_energy_euler = compute_total_energy(Euler_body)
    initial_energy_leapfrog = compute_total_energy(Leapfrog_body)

    # Simulate with Euler integrator
    for _ in range(1000):
        euler_integrator(Euler_body, dt=0.01)
    euler_energy = compute_total_energy(Euler_body)

    # Simulate with Leapfrog integrator
    for _ in range(1000):
        leapfrog_integrator(Leapfrog_body, dt=0.01)
    leapfrog_energy = compute_total_energy(Leapfrog_body)

    euler_error = abs(euler_energy - initial_energy_euler)
    leapfrog_error = abs(leapfrog_energy - initial_energy_leapfrog)

    assert leapfrog_error < euler_error, f"Leapfrog energy error {leapfrog_error} should be less than Euler energy error {euler_error}"

   