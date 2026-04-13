from src.physics.body import Body
from src.integrators.euler import euler_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.physics.system import compute_total_energy
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

    
    leapfrog_integrator(bodies, dt=1.0)

    assert np.allclose(bodies[0].position, np.array([1.0, 0.0]))

def test_leapfrog_integrator_keeps_velocity_constant_without_force():
    body = Body(1.0, [0, 0], [1, 0])
    bodies = [body]

    
    leapfrog_integrator(bodies, dt=1.0)

    assert np.allclose(bodies[0].velocity, np.array([1.0, 0.0]))



def test_leapfrog_outperforms_euler_energy_conservation():
    # Euler bodies
    euler_bodies = [Body(1.0, [0, 0], [0, 1]),
                  Body(1.0, [1, 0], [0, -1])]
    
    # Leapfrog bodies - identical BUT independent instances to avoid coupling
    leapfrog_bodies = [Body(1.0, [0, 0], [0, 1]),
                     Body(1.0, [1, 0], [0, -1])]
    

    # Compute initial energies for both systems before simulation to have a baseline for comparison
    initial_energy_euler = compute_total_energy(euler_bodies)
    initial_energy_leapfrog = compute_total_energy(leapfrog_bodies)

    # Simulate with Euler integrator for 1000 steps with a time step of 0.01
    for _ in range(1000):
        euler_integrator(euler_bodies, dt=0.01)
    euler_energy = compute_total_energy(euler_bodies)

    # Simulate with Leapfrog integrator for 1000 steps with a time step of 0.01
    for _ in range(1000):
        leapfrog_integrator(leapfrog_bodies, dt=0.01)
    leapfrog_energy = compute_total_energy(leapfrog_bodies)
    # Calculate the absolute energy errors for both integrators
    euler_error = abs(euler_energy - initial_energy_euler)
    leapfrog_error = abs(leapfrog_energy - initial_energy_leapfrog)

    # Assert that the energy error of the Leapfrog integrator is less than that of the Euler integrator
    assert leapfrog_error < euler_error, f"Leapfrog energy error {leapfrog_error} should be less than Euler energy error {euler_error}"

   