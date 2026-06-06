import numpy as np
from src.diagnostics.energy import relative_energy_error
from src.diagnostics.lyapunov import compute_lyapunov_exponent
from src.integrators.leapfrog import leapfrog_integrator
from src.physics.body import Body
from src.utils.initial_conditions import make_two_body_circular


def test_relative_energy_error():
    assert relative_energy_error(10, 11) == 0.1


def test_lyapunov_exponent():
    v = np.sqrt(0.5)
    bodies = make_two_body_circular()

    exponent = compute_lyapunov_exponent(
        bodies, leapfrog_integrator, dt=0.01, num_steps=1000, epsilon=1e-8, seed=42
    )

    assert abs(exponent) < 0.5
