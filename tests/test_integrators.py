import pytest
import numpy as np

from src.integrators.rk4 import rk4_integrator
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
        return [Body(1.0, [0, 0], [0, 1]), Body(1.0, [1, 0], [0, -1])]

    return _make


# Parametrize tests to run for all integrators
@pytest.mark.parametrize(
    "integrator",
    [euler_integrator, rk4_integrator, leapfrog_integrator, yoshida_integrator],
)
def test_integrator_updates_position_without_force(single_body, integrator):

    integrator([single_body], dt=1.0)

    assert np.allclose(single_body.position, np.array([1.0, 0.0]))


@pytest.mark.parametrize(
    "integrator",
    [euler_integrator, rk4_integrator, leapfrog_integrator, yoshida_integrator],
)
def test_integrator_keeps_velocity_constant_without_force(single_body, integrator):

    integrator([single_body], dt=1.0)

    assert np.allclose(single_body.velocity, np.array([1.0, 0.0]))


@pytest.mark.parametrize(
    "worse, better",
    [
        (euler_integrator, leapfrog_integrator),
        (euler_integrator, yoshida_integrator),
        (leapfrog_integrator, yoshida_integrator),
    ],
)
def test_energy_conservation_ranking(make_two_bodies, worse, better):

    worse_bodies = make_two_bodies()
    better_bodies = make_two_bodies()

    initial_worse = compute_total_energy(worse_bodies)
    initial_better = compute_total_energy(better_bodies)

    for _ in range(1000):
        worse(worse_bodies, dt=0.01)
    for _ in range(1000):
        better(better_bodies, dt=0.01)

    worse_error = abs(compute_total_energy(worse_bodies) - initial_worse)
    better_error = abs(compute_total_energy(better_bodies) - initial_better)

    assert better_error < worse_error, (
        f"{better.__name__} error {better_error} "
        f"should be less than {worse.__name__} error {worse_error}"
    )


@pytest.fixture
def circular_orbit_bodies():
    # Two equal masses in circular orbit around their common centre of mass
    # With G=1, m=1, r=0.5 (separation=1), the exact circular orbit velocity is v=sqrt(G*m/(2r)) = sqrt(0.5)
    v = np.sqrt(0.5)
    return [
        Body(1.0, [-0.5, 0.0], [0.0, v]),
        Body(1.0, [0.5, 0.0], [0.0, -v]),
    ]


def test_circular_orbit_energy_is_correct(circular_orbit_bodies):
    # For G=1, m1=m2=1, separation=1: E = KE + PE = 2*(0.5*m*v^2) + (-G*m*m/r)
    # E = 1*0.5 + (-1/1) = 0.5 - 1.0 = -0.5
    expected_energy = -0.5
    computed_energy = compute_total_energy(circular_orbit_bodies)

    assert (
        abs(computed_energy - expected_energy) < 1e-10
    ), f"Expected energy {expected_energy}, got {computed_energy}"


def test_angular_momentum_conserved(make_two_bodies):
    bodies = make_two_bodies()

    def total_angular_momentum(bs):
        return sum(b.mass * np.cross
                   (np.append(b.position, 0),
                    np.append(b.velocity, 0)
                    )[2]
                   for b in bs)

    initial_L = total_angular_momentum(bodies)

    for _ in range(1000):
        leapfrog_integrator(bodies, dt=0.01)

    final_L = total_angular_momentum(bodies)

    assert (
        abs(final_L - initial_L) < 1e-6
    ), f"Angular momentum not conserved: initial {initial_L}, final {final_L}"
