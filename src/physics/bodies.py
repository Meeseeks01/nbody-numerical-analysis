""" 
    bodies.py
This module defines the Body class, which represents a physical body in the N-body gravitational simulation.
Each Body has properties such as mass, position, and velocity. The class is designed to be simple and flexible,
allowing for easy integration with the physics functions defined in system.py. The Body class does not contain any methods for updating its state. 
Instead, it serves as a data container that can be manipulated by external functions to compute accelerations and energies based on the current state of the system.

"""

import numpy as np

class Body:
    """
    A class representing a physical body in the simulation.
    Attributes:
        name (str): An optional name for the body.
        mass (float): The mass of the body.
        position (numpy.ndarray): A 2D vector representing the position of the body in space.
        velocity (numpy.ndarray): A 2D vector representing the velocity of the body.
    """

    def __init__(self, mass, position, velocity, name=None):

        self.name = name
        self.mass = float(mass)

        self.position = np.array(position, dtype=float) #dtype (= data type) Numpy array of floats for position, velocity, and acceleration
        assert len(self.position) == 2, "Position must be 2D"
        self.velocity = np.array(velocity, dtype=float)
        assert len(self.velocity) == 2, "Velocity must be 2D"
        
# Improve assert with an if statement and raise an exception with a clear error message instead of using assert, which can be disabled in optimized mode.
    