import os
import numpy as np
import matplotlib.pyplot as plt

from src.physics.system import compute_total_energy
from src.integrators.euler import euler_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.yoshida import yoshida_integrator
from src.diagnostics.energy import relative_energy_error
from src.utils.initial_conditions import make_two_body_circular


def run_simulation(name, integrator, steps=1000, dt=0.01):

    bodies = make_two_body_circular()
    initial_energy = compute_total_energy(bodies)

    print(f"Integrator: {name}")
    print(f"Initial Energy: {initial_energy:.16f}")

    times = []
    energy_errors = []

    for step in range(steps):
        integrator(bodies, dt)

        current_energy = compute_total_energy(bodies)
        error = relative_energy_error(initial_energy, current_energy)

        if step < 3:
            print(
                f"  Step {step}: "
                f"Current Energy = {current_energy:.16f}, "
                f"Relative Error = {error:.8e}"
            )

        times.append(step * dt)
        energy_errors.append(error)

    return np.array(times), np.array(energy_errors)


def main():
    integrators = {
        "Euler": euler_integrator,
        "RK4": rk4_integrator,
        "Leapfrog": leapfrog_integrator,
        "Yoshida": yoshida_integrator,
    }

    plt.figure(figsize=(12, 6))

    print("\n" + "=" * 60)
    print("EXPERIMENT 02: ENERGY DRIFT ANALYSIS")
    print("=" * 60)

    for name, integrator in integrators.items():
        times, energy_errors = run_simulation(name, integrator)
        plt.plot(times, energy_errors, label=name, linewidth=1.5)

        print(f"\n[Analysis: {name}]")

        # Use metrics appropriate to each integrator type
        if name in ["Euler", "RK4"]:

            # Secular drift: Linear fit is appropriate on raw data to find drift rate
            slope, _ = np.polyfit(times, energy_errors, 1)
            print(f" Energy Drift Slope: {slope:.8e}")

        else:
            # Symplectic: Bounded variance and maximum amplitude are the true metrics
            max_error = np.max(energy_errors)
            mean_error = np.mean(energy_errors)
            std_error = np.std(energy_errors)

            print(f" Maximum Bound Energy Error: {max_error:.8e}")
            print(f"  Mean Energy Error: {mean_error:.8e} (± {std_error:.8e})")

        print("-" * 50)

    print("=" * 60 + "\n")

    # Plot Styling modifications for Academic Publication
    plt.xlabel("Time (dimensionless)", fontsize=12)
    plt.ylabel("Relative Energy Error (log scale)", fontsize=12)
    plt.title("Energy Drift Comparison Across Integrators", fontsize=14, pad=20)
    plt.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.yscale("log")
    plt.tight_layout()

    # Robust path generation
    output_dir = "dissertation/figures"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "energy_drift_comparison.png")

    plt.savefig(output_path, dpi=300)
    print(f"Figure saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
