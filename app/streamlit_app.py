import os
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# Core Engine Imports
from src.integrators.euler import euler_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.yoshida import yoshida_integrator
from src.physics.system import compute_total_energy
from src.diagnostics.energy import relative_energy_error
from src.utils.initial_conditions import (
    make_two_body_circular,
    pythagorean_three_bodies,
)


def run_simulation(integrator_func, system_init_func, dt, steps):
    """
    Run an N-body simulation and track:
    - orbital trajectories
    - relative energy errors
    """

    bodies = system_init_func()
    num_bodies = len(bodies)

    positions = []
    energy_errors = []

    initial_energy = compute_total_energy(bodies)

    for step in range(steps):

        positions.append([body.position.copy() for body in bodies])

        current_energy = compute_total_energy(bodies)

        error = relative_energy_error(
            initial_energy,
            current_energy,
        )

        energy_errors.append(error)

        try:
            integrator_func(bodies, dt)

        except (ValueError, OverflowError, FloatingPointError):

            remaining = steps - step

            positions.extend(
                [[np.array([np.nan, np.nan]) for _ in range(num_bodies)]] * remaining
            )

            energy_errors.extend([np.nan] * remaining)

            break

    return np.array(positions), np.array(energy_errors)


# ---------------------------------------------------------
# STREAMLIT PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    layout="wide",
    page_title="N-Body Numerical Laboratory",
)

st.title("🌌 N-Body Numerical & Chaotic Analysis Laboratory")

st.markdown("""
Interactive comparison of numerical integrators
for gravitational N-body simulations.

Explore:
- orbit trajectories
- energy conservation
- timestep stability
- chaotic behavior
""")

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------

st.sidebar.header("Simulation Configuration")

# System selection
system_type = st.sidebar.radio(
    "Select Dynamical System:",
    [
        "Two-Body Control Group (Circular)",
        "Three-Body Chaotic Regime (Pythagorean)",
    ],
)

system_map = {
    "Two-Body Control Group (Circular)": make_two_body_circular,
    "Three-Body Chaotic Regime (Pythagorean)": pythagorean_three_bodies,
}

selected_init_func = system_map[system_type]
num_bodies = len(selected_init_func())

# Integrator selection
st.sidebar.subheader("Solvers to Evaluate")

available_integrators = {
    "Explicit Euler": euler_integrator,
    "Leapfrog": leapfrog_integrator,
    "RK4": rk4_integrator,
    "Yoshida": yoshida_integrator,
}

selected_solver_names = []

for name in available_integrators.keys():

    if st.sidebar.checkbox(name, value=(name == "Yoshida")):
        selected_solver_names.append(name)

# Numerical parameters
st.sidebar.subheader("Discretization Parameters")

dt = st.sidebar.slider(
    "Timestep dt",
    0.0005,
    0.05,
    0.001,
    step=0.0005,
    format="%.4f",
)

steps = st.sidebar.slider(
    "Number of Steps",
    100,
    5000,
    2000,
    step=100,
)

# ---------------------------------------------------------
# MAIN APPLICATION
# ---------------------------------------------------------

if not selected_solver_names:

    st.warning("Please select at least one numerical integrator.")

else:

    tab1, tab2 = st.tabs(
        [
            "📊 Interactive Simulation",
            "📈 Lyapunov Diagnostics",
        ]
    )

    # =====================================================
    # TAB 1 — LIVE SIMULATION
    # =====================================================

    with tab1:

        col1, col2 = st.columns([1, 1.1])

        fig_orbit, ax_orbit = plt.subplots(figsize=(6.5, 6.5))
        fig_energy, ax_energy = plt.subplots(figsize=(7, 6))

        # Integrator colors
        integrator_colors = {
            "Explicit Euler": "tab:blue",
            "Leapfrog": "tab:green",
            "RK4": "tab:orange",
            "Yoshida": "tab:red",
        }

        # Body linestyles
        body_styles = ["-", "--", ":"]

        # -------------------------------------------------
        # RUN SIMULATIONS
        # -------------------------------------------------

        for solver_name in selected_solver_names:

            integrator_func = available_integrators[solver_name]

            positions, energy_errors = run_simulation(
                integrator_func,
                selected_init_func,
                dt,
                steps,
            )

            time_axis = np.arange(steps) * dt

            color = integrator_colors[solver_name]

            # ---------------------------------------------
            # ORBIT PLOTS
            # ---------------------------------------------

            for b_idx in range(num_bodies):

                linestyle = body_styles[b_idx % len(body_styles)]

                ax_orbit.plot(
                    positions[:, b_idx, 0],
                    positions[:, b_idx, 1],
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    alpha=0.9,
                    label=solver_name if b_idx == 0 else "",
                )

            # Mark initial positions
            if solver_name == selected_solver_names[0]:

                for b_idx in range(num_bodies):

                    ax_orbit.plot(
                        positions[0, b_idx, 0],
                        positions[0, b_idx, 1],
                        marker="o",
                        color="black",
                        markerfacecolor="white",
                        markeredgewidth=1.5,
                        markersize=6,
                        zorder=5,
                    )

            # ---------------------------------------------
            # ENERGY PLOTS
            # ---------------------------------------------

            safe_energy_errors = np.where(
                energy_errors <= 0,
                1e-16,
                energy_errors,
            )

            ax_energy.plot(
                time_axis,
                safe_energy_errors,
                color=color,
                linewidth=2,
                label=solver_name,
            )

        # -------------------------------------------------
        # ORBIT FIGURE FORMATTING
        # -------------------------------------------------

        ax_orbit.set_title(
            "Orbital Trajectories",
            fontsize=12,
            fontweight="bold",
        )

        ax_orbit.set_xlabel("x Position")
        ax_orbit.set_ylabel("y Position")

        ax_orbit.set_aspect("equal")

        ax_orbit.grid(
            True,
            which="both",
            linestyle=":",
            alpha=0.5,
        )

        if "Pythagorean" in system_type:

            ax_orbit.set_xlim(-5, 5)
            ax_orbit.set_ylim(-5, 5)

        ax_orbit.legend(
            loc="upper right",
            fontsize=9,
            framealpha=0.9,
            shadow=True,
        )

        with col1:
            st.pyplot(fig_orbit)

        # -------------------------------------------------
        # ENERGY FIGURE FORMATTING
        # -------------------------------------------------

        ax_energy.set_title(
            "Relative Energy Error |ΔE / E₀|",
            fontsize=12,
            fontweight="bold",
        )

        ax_energy.set_xlabel("Simulated Time")
        ax_energy.set_ylabel("Relative Energy Error |ΔE / E₀|")

        ax_energy.set_yscale("log")

        ax_energy.grid(
            True,
            which="both",
            linestyle=":",
            alpha=0.5,
        )

        ax_energy.legend(
            loc="lower right",
            fontsize=9,
            shadow=True,
        )

        with col2:
            st.pyplot(fig_energy)

    # =====================================================
    # TAB 2 — LYAPUNOV ANALYSIS
    # =====================================================

    with tab2:

        st.subheader("🔬 Quantitative Chaos Identification")

        st.markdown("""
Lyapunov exponent calculations are computationally intensive,
therefore precomputed experiment results are displayed below.
""")

        img_path = "dissertation/figures/" "lyapunov_exponent_comparison.png"

        if os.path.exists(img_path):

            st.image(
                img_path,
                caption=(
                    "Lyapunov exponent comparison across " "system configurations."
                ),
            )

        else:

            st.info("Run `04_lyapunov_analysis.py` " "to generate the figure.")

        st.info(r"""
**Summary of Lyapunov Analysis**

- Stable two-body systems produce near-zero
  Lyapunov exponents, indicating bounded motion.

- Chaotic three-body systems produce positive
  Lyapunov exponents, demonstrating sensitive
  dependence on initial conditions.
""")
