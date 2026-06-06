import numpy as np
import matplotlib.pyplot as plt

from src.integrators.euler import euler_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.yoshida import yoshida_integrator
from src.utils.initial_conditions import make_two_body_circular


def run_simulation(integrator, steps=1000, dt=0.01):
    """
    Evolve the circular two-body system and record body positions over time.
    """
    bodies = make_two_body_circular()
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

    # Plot results for each integrator
    print("=" * 60)
    print("Experiment_01: Two-Body Orbit Simulation")
    print("=" * 60)

    for ax, (name, integrator) in zip(axes, integrators.items()):
        positions = run_simulation(integrator)

        body1_x = positions[:, 0, 0]
        body1_y = positions[:, 0, 1]

        body2_x = positions[:, 1, 0]
        body2_y = positions[:, 1, 1]

        # ------------------------------------------------------------------
        r_initial = np.sqrt(body1_x[0] ** 2 + body1_y[0] ** 2)
        r_final = np.sqrt(body1_x[-1] ** 2 + body1_y[-1] ** 2)
        radial_deviation = np.abs(r_final - r_initial)

        print(f"Integrator: {name}")
        print(f"Initial Orbital Radius: {r_initial:.6f}")
        print(f"Final Orbital Radius:   {r_final:.6f}")
        print(f"Radial Deviation:   {radial_deviation:.8e}")
        print("-" * 50)
        # ------------------------------------------------------------------

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

        #
        # ------------------------------------------------------------------
        print("=" * 60 + "\n")
        # ------------------------------------------------------------------

    fig.suptitle("Two-Body Orbit Comparison Across Integrators")
    plt.tight_layout()
    plt.savefig("dissertation/figures/two_body_orbit.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
