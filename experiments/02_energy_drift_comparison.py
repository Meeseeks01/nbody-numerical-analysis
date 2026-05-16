import numpy as np
import matplotlib.pyplot as plt

from src.physics.body import Body
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
    print(f"Initial Energy: {initial_energy}")

    times = []
    energy_errors = []

    for step in range(steps):
        integrator(bodies, dt)

        current_energy = compute_total_energy(bodies)
        error = relative_energy_error(initial_energy, current_energy)

        if step<3:
            print(
                  f"Step {step}: "
                  f"Current Energy = {current_energy}, "
                  f"Relative Error = {error}"
                  )

        times.append(step * dt)
        energy_errors.append(error)

    return times, energy_errors

def main():
    integrators = {
        'Euler': euler_integrator,
        'RK4': rk4_integrator,
        'Leapfrog': leapfrog_integrator,
        'Yoshida': yoshida_integrator
    }

    plt.figure(figsize=(12, 6))

    for name, integrator in integrators.items():
        times, energy_errors = run_simulation(name, integrator)
        plt.plot(times, energy_errors, label=name)

    
    plt.xlabel('Time', fontsize=14)
    plt.ylabel('Relative Energy Error (log scale)', fontsize=14)
    plt.title('Energy Drift Comparison Across Integrators', fontsize=16, pad=20)
    plt.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
    plt.grid(True)
    plt.yscale("log")
    plt.tight_layout()

    plt.savefig("dissertation/figures/energy_drift_comparison.png", dpi=300)
    plt.show()
    
if __name__ == "__main__":    
    main()