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

#TODO add 3 bodies as well 