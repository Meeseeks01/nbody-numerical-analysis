# improve test names

from src.physics.body import Body
from src.physics.system import compute_accelerations, compute_total_energy
import numpy as np
import pytest


def test_compute_accelerations():
    earth = Body(1.0, [1, 0], [0, 1])
    mars = Body(0.1, [-1, 0], [0, -1])
    bodies = [earth, mars]
    accelerations = compute_accelerations(bodies)
    a1 = accelerations[0]
    a2 = accelerations[1]

    assert np.allclose(earth.mass * a1, -mars.mass * a2)
    assert accelerations[0][0] < 0  # Earth accelerates toward Mars (left)
    assert accelerations[1][0] > 0  # Mars accelerates toward Earth (right)


def test_compute_total_energy():
    earth = Body(1.0, [1, 0], [0, 0])
    mars = Body(0.1, [-1, 0], [0, 0])
    bodies = [earth, mars]
    energy = compute_total_energy(bodies)
    assert energy < 0  # Total energy should be negative for a bound system


def test_body_position_must_be_2D():
    with pytest.raises(ValueError):
        body = Body(1.0, [1, 0, 0], [0, 1])
