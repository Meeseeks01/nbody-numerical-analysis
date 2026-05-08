import pytest
import numpy as np

from src.physics.body import Body
from src.integrators.euler import euler_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.physics.system import compute_total_energy
from src.integrators.yoshida import yoshida_integrator

# Fixtures for testing integrators
@pytest.fixture
def single_body():
    return Body(1.0, [0, 0], [1, 0])

# Using a fixture factory to create independent instances of the two bodies for each test to avoid coupling effects
@pytest.fixture
def make_two_bodies():
    def _make():
        return [
                Body(1.0, [0, 0], [0, 1]), 
                Body(1.0, [1, 0], [0, -1])
            ]
    return _make

# Parametrize tests to run for all integrators
@pytest.mark.parametrize(
    "integrator",
    [euler_integrator, leapfrog_integrator, yoshida_integrator],
)
def test_integrator_updates_position_without_force(single_body, integrator):

    integrator([single_body], dt=1.0)

    assert np.allclose(single_body.position, np.array([1.0, 0.0]))


@pytest.mark.parametrize(
    "integrator",
    [euler_integrator, leapfrog_integrator, yoshida_integrator],
)
def test_integrator_keeps_velocity_constant_without_force(single_body, integrator):

    integrator([single_body], dt=1.0)

    assert np.allclose(single_body.velocity, np.array([1.0, 0.0]))



def test_leapfrog_outperforms_euler_energy_conservation(make_two_bodies):
    # Euler bodies
    euler_bodies = make_two_bodies()  # Create a new instance of the two bodies for Euler to ensure identical initial conditions

    # Leapfrog bodies - identical BUT independent instances to avoid coupling
    leapfrog_bodies = make_two_bodies()  # Create a new instance of the two bodies for Leapfrog to ensure identical initial conditions

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
    assert leapfrog_error < euler_error, (
        f"Leapfrog energy error {leapfrog_error} "
        f"should be less than Euler energy error {euler_error} "
    ) 



# Test comparison of Yoshida integrator against Euler and Leapfrog for energy conservation
def test_yoshida_outperforms_euler_and_leapfrog_energy_conservation(make_two_bodies):
    # Euler bodies
    euler_bodies = make_two_bodies() 

    # Leapfrog bodies - identical BUT independent instances to avoid coupling
    leapfrog_bodies = make_two_bodies() 

    # Yoshida bodies - identical BUT independent instances to avoid coupling
    yoshida_bodies = make_two_bodies()  # Create a new instance of the two bodies for Yoshida to ensure identical initial conditions

    # Compute initial energies for all systems before simulation to have a baseline for comparison
    initial_energy_euler = compute_total_energy(euler_bodies)
    initial_energy_leapfrog = compute_total_energy(leapfrog_bodies)
    initial_energy_yoshida = compute_total_energy(yoshida_bodies)

    # Simulate with Euler integrator for 1000 steps with a time step of 0.01
    for _ in range(1000):
        euler_integrator(euler_bodies, dt=0.01)
    euler_energy = compute_total_energy(euler_bodies)

    # Simulate with Leapfrog integrator for 1000 steps with a time step of 0.01
    for _ in range(1000):
        leapfrog_integrator(leapfrog_bodies, dt=0.01)
    leapfrog_energy = compute_total_energy(leapfrog_bodies)

    # Simulate with Yoshida integrator for 1000 steps with a time step of 0.01
    for _ in range(1000):
        yoshida_integrator(yoshida_bodies, dt=0.01)
    yoshida_energy = compute_total_energy(yoshida_bodies)

    # Calculate the absolute energy errors for all integrators
    euler_error = abs(euler_energy - initial_energy_euler)
    leapfrog_error = abs(leapfrog_energy - initial_energy_leapfrog)
    yoshida_error = abs(yoshida_energy - initial_energy_yoshida)

    # Assert that the energy error of the Yoshida integrator is less than that of both the Euler and Leapfrog integrators
    assert yoshida_error < euler_error, (
        f"Yoshida energy error {yoshida_error} "
        f"should be less than Euler energy error {euler_error} "
    )

    assert yoshida_error < leapfrog_error, (
        f"Yoshida energy error {yoshida_error} "
        f"should be less than Leapfrog energy error {leapfrog_error} "
    ) 


