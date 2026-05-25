from src.physics.body import Body
import numpy as np


def test_body_mass():
    earth = Body(1.0, [1, 0], [0, 1])
    assert earth.mass == 1.0


def test_body_position():
    earth = Body(1.0, [1, 0], [0, 1])
    assert np.array_equal(earth.position, np.array([1.0, 0.0]))


def test_body_velocity():
    earth = Body(1.0, [1, 0], [0, 1])
    assert np.array_equal(earth.velocity, np.array([0.0, 1.0]))
