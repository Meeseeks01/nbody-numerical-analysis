import numpy as np

from src.physics.body import Body


def make_two_body_circular():
    """
    Create a stable equal-mass circular two-body system.
    """
    v = np.sqrt(0.5)
    return [
        Body(1.0, [-0.5, 0.0], [0.0, -v]),
        Body(1.0, [0.5, 0.0], [0.0, v])
        ]


def pythagorean_three_bodies():
    """
    Create the Pythagorean three-body problem initial conditions. (Burrau's Problem)
    """
    return [
        Body(3.0, [-1.0, 0.0], [0.0, 0.0]),
        Body(4.0, [0.0, 2.0], [0.0, 0.0]),
        Body(5.0, [2.0, 0.0], [0.0, 0.0]),
    ]
