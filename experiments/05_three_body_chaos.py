import os
import matplotlib.pyplot as plt
import numpy as np

from src.integrators.euler import euler_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.yoshida import yoshida_integrator
from src.utils.initial_conditions import pythagorean_three_bodies


def generate_trajectories(integrator_func, dt, num_steps, name):
    """
    Evolve the Pythagorean three-body problem using a specific integrator
    and record the full spatial history of all bodies.
    """
    bodies = pythagorean_three_bodies()
    num_bodies = len(bodies)

    # Initialize history arrays: shape (num_bodies, num_steps + 1, 2) for 2D (x, y)
    history = np.zeros((num_bodies, num_steps + 1, 2))
    
    termination_status = "COMPLETED"
    failure_step = num_steps

    # Record initial conditions (t = 0)
    for b_idx in range(num_bodies):
        history[b_idx, 0, 0] = bodies[b_idx].position[0]
        history[b_idx, 0, 1] = bodies[b_idx].position[1]
    

    # Time-stepping loop
    for step in range(1, num_steps + 1):
        try:
            integrator_func(bodies, dt)

            # Record positions after integration step
            for b_idx in range(num_bodies):
                history[b_idx, step, 0] = bodies[b_idx].position[0]
                history[b_idx, step, 1] = bodies[b_idx].position[1]

        except (ValueError, OverflowError, FloatingPointError):
            # If an explicit method experiences a numerical explosion,
            # fill the remaining steps with NaN so Matplotlib stops plotting gracefully
            termination_status = "FAILED"
            failure_step = step - 1 if step > 1 else 0
            history[:, step:, :] = np.nan
            break
        
    print(f"Integrator: {name}")
    print(f"Status: {termination_status}")
    print(f"Steps Completed: {failure_step} / {num_steps}")
    
    for b_idx in range(num_bodies):
        fx = history[b_idx, failure_step if termination_status == "FAILED" else -1, 0]
        fy = history[b_idx, failure_step if termination_status == "FAILED" else -1, 1]
        print(f"Body {b_idx+1} Final Position: ({fx:.4f}, {fy:.4f})")
        print("-" * 50)

    return history


def execute_three_body_chaos_experiment():
    print("EXPERIMENT 05: THREE-BODY CHAOS TRAJECTORY COMPARISON")

    # Define strict parameters to preserve close encounter integrity
    dt = 0.001
    num_steps = 2000 

    integrators = {
        "Explicit Euler": euler_integrator,
        "Leapfrog (2nd-Order Symplectic)": leapfrog_integrator,
        "Runge-Kutta (RK4)": rk4_integrator,
        "Yoshida (4th-Order Symplectic)": yoshida_integrator,
    }

    # Color palette strictly mapped to body indexes across all subplots
    # Body 0 = m:3, Body 1 = m:4, Body 2 = m:5
    body_colors = ["tab:blue", "tab:green", "tab:red"]
    body_labels = [
        "Body 1 ($m=3$)",
        "Body 2 ($m=4$)",
        "Body 3 ($m=5$)",
    ]

    # Create a publication-quality 2x2 subplot layout
    fig, axs = plt.subplots(2, 2, figsize=(14, 12))
    axs_flat = axs.flatten()

    # Capture initial positions from a temporary generation for plotting starting points
    initial_bodies = pythagorean_three_bodies()
    init_positions = [b.position for b in initial_bodies]

    for idx, (name, int_func) in enumerate(integrators.items()):
        print(f"Simulating trajectory using {name}...")
        ax = axs_flat[idx]

        # Compute trajectories
        history = generate_trajectories(int_func, dt, num_steps, name)

        # Plot the continuous path of each body
        for b_idx in range(len(initial_bodies)):
            ax.plot(
                history[b_idx, :, 0],
                history[b_idx, :, 1],
                color=body_colors[b_idx],
                linewidth=1.5,
                alpha=0.85,
                label=body_labels[b_idx] if idx == 0 else "",  # Label only first to avoid legend duplication
            )

            # Mark initial position with an open circle
            ax.plot(
                init_positions[b_idx][0],
                init_positions[b_idx][1],
                marker="o",
                color=body_colors[b_idx],
                markersize=8,
                markerfacecolor="white",
                markeredgewidth=2,
                zorder=5,
            )

            # Mark final position with a solid star if the simulation didn't explode to NaN
            final_x = history[b_idx, -1, 0]
            final_y = history[b_idx, -1, 1]
            if not np.isnan(final_x):
                ax.plot(
                    final_x,
                    final_y,
                    marker="*",
                    color=body_colors[b_idx],
                    markersize=10,
                    zorder=5,
                )

        # Style individual subplots for consistency and professional aesthetics
        ax.set_title(name, fontsize=13, fontweight="bold", pad=10)
        ax.grid(True, which="both", linestyle=":", alpha=0.5)
        ax.set_aspect("equal", adjustable="box")

        # Fixed axis ranges for consistent comparison
        ax.set_xlim(-5, 5) 
        ax.set_ylim(-5, 5)

    # Set outer bounding labels across shared grids
    for i in range(2):
        axs[1, i].set_xlabel("Spatial Coordinate $x$", fontsize=12, labelpad=10)
        axs[i, 0].set_ylabel("Spatial Coordinate $y$", fontsize=12, labelpad=10)
    
    total_simulated_time = dt * num_steps
    plt.suptitle(
        f"Three-Body Trajectory Comparison Across Integrators ($T={total_simulated_time}$)",
        fontsize=16,
        fontweight="bold",
        y=0.96,
    )

    # Single unified legend placed cleanly outside the subplots
    fig.legend(
        loc="upper right",
        bbox_to_anchor=(0.98, 0.94),
        fontsize=11,
        frameon=True,
        shadow=True,
    )

    plt.tight_layout(rect=[0.02, 0.02, 0.85, 0.92])

    # Created output directory if needed
    output_dir = "dissertation/figures"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "three_body_chaos_trajectories.png")

    plt.savefig(save_path, dpi=300)
    print(f"Figure saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    execute_three_body_chaos_experiment()