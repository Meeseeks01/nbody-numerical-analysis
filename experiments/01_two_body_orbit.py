import numpy as np
import matplotlib.pyplot as plt

from src.physics.body import Body
from src.integrators.euler import euler_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.yoshida import yoshida_integrator


def make_two_bodies():
    """Create a stable equal-mass circular two-body system."""
    v = np.sqrt(0.5)
    return [
        Body(1.0, [-0.5, 0.0], [0.0, -v]),
        Body(1.0, [0.5, 0.0], [0.0, v]),
    ]


def run_simulation(integrator, steps=1000, dt=0.01):
    bodies = make_two_bodies()
    positions = []
    

    for _ in range(steps):
        positions.append([body.position.copy() for body in bodies])
        integrator(bodies, dt)

    return np.array(positions)


def main():
    integrators = {
        "Euler": euler_integrator,
        "Leapfrog": leapfrog_integrator,
        "RK4": rk4_integrator,
        "Yoshida": yoshida_integrator,
    }

    fig, axes = plt.subplots(2, 2, figsize=(10, 10), sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, (name, integrator) in zip(axes, integrators.items()):
        positions = run_simulation(integrator)

        body1_x = positions[:, 0, 0]
        body1_y = positions[:, 0, 1]

        body2_x = positions[:, 1, 0]
        body2_y = positions[:, 1, 1]

        ax.plot(body1_x, body1_y, label="Body 1")
        ax.plot(body2_x, body2_y, label="Body 2")

        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_xlim(-0.8, 0.8)
        ax.set_ylim(-0.8, 0.8)
        ax.grid(True)
        ax.legend()

    fig.suptitle("Two-Body Orbit Comparison Across Integrators")
    plt.tight_layout()
    plt.savefig("dissertation/figures/two_body_orbit.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()