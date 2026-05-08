import numpy as np
import matplotlib.pyplot as plt

from src.integrators.rk4 import rk4_integrator
from src.physics.body import Body
from src.physics.system import compute_total_energy
from src.integrators.euler import euler_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.yoshida import yoshida_integrator
from src.diagnostics.energy import relative_energy_error

def make_two_bodies():
    return [
        Body(1.0, [0.0, 0.0], [0.0, 0.1]),
        Body(1.0, [1.0, 0.0], [0.0, -1.0])
    ]

def run_simulation(integrator, steps=1000, dt=0.01):
    bodies = make_two_bodies()
    initial_energy = compute_total_energy(bodies)

    times = []
    energy_errors = []

    for step in range(steps):
        integrator(bodies, dt)

        current_energy = compute_total_energy(bodies)
        error = relative_energy_error(initial_energy, current_energy)

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

    # plt.figure(figsize=(10, 6))

    for name, integrator in integrators.items():
        times, energy_errors = run_simulation(integrator)
        plt.plot(times, energy_errors, label=name)

    plt.xlabel('Time')
    plt.ylabel('Relative Energy Error')
    plt.title('Energy Drift Comparison Across Integrators')
    plt.legend()
    plt.grid(True)

    plt.savefig("dissertation/figures/energy_drift_comparison.png", dpi=300)
    plt.show()

if __name__ == "__main__":    
    main()