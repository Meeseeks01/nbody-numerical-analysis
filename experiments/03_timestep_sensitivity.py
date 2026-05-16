import matplotlib.pyplot as plt
import numpy as np

from src.diagnostics.energy import relative_energy_error
from src.physics.body import Body
from src.integrators.euler import euler_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.yoshida import yoshida_integrator
from src.physics.system import compute_total_energy
from src.utils.initial_conditions import make_two_body_circular


def timestep_sensitivity(
    integrator,
    dt_values,
    total_time=10.0,
    instability_threshold=0.1,
):
    """
    Run the two-body problem with different time steps and compute energy errors.

    Parameters
    ----------
    integrator : callable
        Numerical integration function to test.
    dt_values : array-like
        Time step sizes to evaluate.
    total_time : float, optional
        Total simulated physical time.
    instability_threshold : float, optional
        Relative energy error above which the simulation is marked unstable.

    Returns
    -------
    tuple[list[float], list[bool]]
        Final or termination relative energy errors and instability flags for each timestep.
    """
    energy_errors = []
    unstable_flags = []

    for dt in dt_values:
        bodies = make_two_body_circular()
        initial_energy = compute_total_energy(bodies)
        steps = int(total_time / dt)

        unstable = False
        current_error = 0.0

        for _ in range(steps):
            integrator(bodies, dt)

            current_energy = compute_total_energy(bodies)
            current_error = relative_energy_error(initial_energy, current_energy)

            if current_error > instability_threshold:
                unstable = True
                break

        unstable_flags.append(unstable)
        # Store the actual error at the end or at the moment of failure
        energy_errors.append(current_error)

    return energy_errors, unstable_flags


def plot_convergence_analysis():
    # Setup logarithmically spaced dt values spanning 3-4 orders of magnitude
    dt_values = np.logspace(-4, -1.7, num=10)

    integrators = {
        "Euler": (euler_integrator, "o", "tab:blue"),
        "Leapfrog": (leapfrog_integrator, "s", "tab:green"),
        "RK4": (rk4_integrator, "^", "tab:orange"),
        "Yoshida": (yoshida_integrator, "d", "tab:red"),
    }

    plt.figure(figsize=(11, 7))

    for name, (integrator_func, marker, color) in integrators.items():
        errors, flags = timestep_sensitivity(
            integrator_func,
            dt_values,
            total_time=10.0,
            instability_threshold=0.1,
        )

        errors = np.array(errors)
        flags = np.array(flags)

        # Isolate stable runs for the regression line
        stable_mask = ~flags
        fit_dt = dt_values[stable_mask]
        fit_errors = errors[stable_mask]

        # Calculate Empirical Slope strictly over the stable, non-diverged data
        if len(fit_errors) > 1:
            slope, intercept = np.polyfit(
                np.log10(fit_dt), np.log10(fit_errors), 1
            )
            label_text = f"{name} (Empirical Slope: {slope:.2f})"
        else:
            label_text = f"{name} (All Steps Unstable)"

        # 1. Plot the stable empirical data points
        plt.loglog(
            fit_dt,
            fit_errors,
            marker=marker,
            linestyle="",
            color=color,
            markersize=8,
            label=label_text,
        )

        # 2. Plot the regression line over the stable data range
        if len(fit_errors) > 1:
            fit_line = 10**intercept * fit_dt**slope
            plt.loglog(
                fit_dt,
                fit_line,
                linestyle="--",
                color=color,
                alpha=0.7,
            )

        # 3. Plot unstable runs at their ACTUAL error value at termination
        unstable_dt = dt_values[flags]
        unstable_errors = errors[flags]
        if len(unstable_dt) > 0:
            plt.loglog(
                unstable_dt,
                unstable_errors,
                marker="x",
                color="black",
                markersize=10,
                markeredgewidth=2,
                linestyle="",
                label=f"{name} (Diverged)" if f"{name} (Diverged)" not in plt.gca().get_legend_handles_labels()[1] else ""
            )

    plt.gca().invert_xaxis()

    # Visual threshold line to give context to the divergence points
    plt.axhline(
        y=0.1, 
        color="grey", 
        linestyle=":", 
        alpha=0.7, 
        label="Instability Threshold (0.1)"
    )

    plt.title(
        "Empirical Verification of Integrator Convergence Orders",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    plt.xlabel(r"Timestep Size ($\Delta t$)", fontsize=13, labelpad=10)
    plt.ylabel(
        r"Relative Energy Error ($|E_{final} - E_{init}| / |E_{init}|$)",
        fontsize=13,
        labelpad=10,
    )

    plt.grid(True, which="both", linestyle=":", alpha=0.5)
    plt.legend(loc="upper left", fontsize=10, frameon=True, shadow=True)
    plt.tight_layout()

    plt.savefig("dissertation/figures/integrator_convergence_verification.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_convergence_analysis()