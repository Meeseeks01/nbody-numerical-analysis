"""
system.py

Stateless physics functions for the N-body gravitational system.

Functions are intentionally kept stateless (no class) so that
integrators can call compute_accelerations() independently
without coupling to a System object. Each function takes the
current list of Body objects as input and returns a computed
quantity without modifying or storing any state.
"""

import numpy as np

# We are working in dimentionless units, so we can set the gravitational constant to 1.0 for simplicity.
G = 1.0  # Gravitational constant


def compute_accelerations(bodies):
    """
    Compute the gravitational acceleration on each body due to all other bodies.
    The acceleration is calculated using Newton's law of universal gravitation.
    """

    accelerations = []  # List to store the computed accelerations for each body

    for body in bodies:
        acceleration = np.zeros(2)  # Initialize acceleration to zero
        for other_body in bodies:
            if body is not other_body:  # Avoid self-interaction
                r_vector = (
                    other_body.position - body.position
                )  # Vector from body to other_body
                r_magnitude = np.linalg.norm(
                    r_vector
                )  # Distance between the two bodies
                if r_magnitude > 0:  # Avoid division by zero
                    r_unit_vector = (
                        r_vector / r_magnitude
                    )  # Unit vector in the direction of r_vector
                    acceleration_contribution = (
                        G * other_body.mass / r_magnitude**2 * r_unit_vector
                    )  # Gravitational acceleration contribution
                    acceleration += acceleration_contribution  # Sum contributions from all other bodies
        accelerations.append(
            acceleration
        )  # Append the computed acceleration for this body
    return accelerations


def compute_total_energy(bodies):
    """
    Compute the total energy of the system, which is the sum of kinetic and potential energy.
    Kinetic energy is calculated as (1/2) * m * v^2 for each body.
    Potential energy is calculated using the formula -G * m1 * m2 / r for each unique pair of bodies.
    """
    total_kinetic_energy = 0.0
    total_potential_energy = 0.0

    # Calculate kinetic energy
    for body in bodies:
        speed_squared = np.dot(body.velocity, body.velocity)  # v^2
        total_kinetic_energy += 0.5 * body.mass * speed_squared  # (1/2) * m * v^2

    # Calculate potential energy
    for i in range(len(bodies)):
        for j in range(i + 1, len(bodies)):  # Avoid double counting pairs
            r_vector = (
                bodies[j].position - bodies[i].position
            )  # Vector from body i to body j
            r_magnitude = np.linalg.norm(r_vector)  # Distance between the two bodies
            if r_magnitude > 0:  # Avoid division by zero
                potential_energy_contribution = (
                    -G * bodies[i].mass * bodies[j].mass / r_magnitude
                )  # -G * m1 * m2 / r
                total_potential_energy += potential_energy_contribution

    total_energy = (
        total_kinetic_energy + total_potential_energy
    )  # Total energy is kinetic + potential
    return total_energy
