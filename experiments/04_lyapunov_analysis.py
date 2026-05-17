
import matplotlib.pyplot as plt
import numpy as np

from src.integrators.euler import euler_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.yoshida import yoshida_integrator
from src.utils.initial_conditions import make_two_body_circular, pythagorean_three_bodies

from src.diagnostics.lyapunov import compute_lyapunov_exponent


def execute_lyapunov_experiment():
    """
    Runs Experiment 04: Evaluates all 4 integrators across both a stable control 
    and a chaotic system, producing a comparative analysis bar chart.
    """
    print("Starting Experiment 04: Lyapunov Analysis...")
    
    # 1. Setup step-based parameters matching your core function constraints
    dt = 0.001
    total_time = 100.0
    num_steps = int(total_time / dt)  # Convert total_time to steps required
    epsilon = 1e-5                    # Perturbation distance handled internally
    seed = 42                         # Ensure reproducible stochastic choices if used
    renormalization_steps = 500        # How often to renormalize the perturbation

    integrators = {
        "Euler": euler_integrator,
        "Leapfrog": leapfrog_integrator,
        "RK4": rk4_integrator,
        "Yoshida": yoshida_integrator
    }

    systems = {
        "Stable 2-Body (Circular)": {
            "factory": make_two_body_circular,
            "total_time": 1000.0,
            "renormalization_steps": 1000,  
        },

        "Chaotic 3-Body (Pythagorean)": {
            "factory": pythagorean_three_bodies,
            "total_time": 100.0,
            "renormalization_steps": 500,
        }
        
    }

    # Data structure to hold final exponents
    results = {sys_name: [] for sys_name in systems}
    integrator_names = list(integrators.keys())

    # 2. Honest data generation phase
    for sys_name, system_config in systems.items():
        for int_name, int_func in integrators.items():
            print(f"  > Running {int_name} on {sys_name}...")
            
            # Generate a fresh initial condition list of bodies for every single test
            # bodies = init_func()
            bodies = system_config["factory"]()
            total_time = system_config["total_time"]
            renormalization_steps = system_config["renormalization_steps"]

            num_steps = int(total_time / dt)
            
            try:
                # Direct call matching your internal Benettin architecture perfectly
                lambda_val = compute_lyapunov_exponent(
                    bodies=bodies,
                    integrator=int_func,
                    dt=dt,
                    num_steps=num_steps,
                    epsilon=epsilon,
                    seed=seed,
                    renormalization_steps=renormalization_steps
                )
                results[sys_name].append(lambda_val)
            except Exception as e:
                print(f"    [!] Run failed for {int_name} on {sys_name}: {e}")
                results[sys_name].append(np.nan)

   
    # GENERATING THE BAR CHART
    
    x = np.arange(len(integrator_names))  # Label locations
    width = 0.35  # Bar width

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot bars for both physical regimes side-by-side
    bars1 = ax.bar(
        x - width/2, results["Stable 2-Body (Circular)"], width, 
        label="Stable 2-Body (Control)", color="tab:blue", edgecolor="black", alpha=0.8
    )
    bars2 = ax.bar(
        x + width/2, results["Chaotic 3-Body (Pythagorean)"], width, 
        label="Chaotic 3-Body (Pythagorean)", color="tab:red", edgecolor="black", alpha=0.8
    )

    # Formatting headers and axes to match publication-quality layouts
    ax.set_title(
        r"Quantitative Chaos Verification: Maximal Lyapunov Exponents ($\lambda$)", 
        fontsize=14, fontweight="bold", pad=20
    )
    ax.set_ylabel(r"Estimated Lyapunov Exponent ($\lambda$)", fontsize=12, labelpad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(integrator_names, fontsize=11)
    
    # Horizontal reference line at zero to highlight the non-chaotic limit
    ax.axhline(y=0.0, color="grey", linestyle="--", alpha=0.7)
    
    ax.grid(True, which="both", axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=11, frameon=True, shadow=True)
    
    # Automatically append value labels on top of the bars for quick readability
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            if not np.isnan(height):
                ax.annotate(
                    f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight="bold"
                )

    autolabel(bars1)
    autolabel(bars2)

    plt.tight_layout()
    
    # Ensure directory exists before saving figure output
    output_dir = "dissertation/figures"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, "lyapunov_exponent_comparison.png")
    
    plt.savefig(save_path, dpi=300)
    print(f"Success! Plot cleanly saved to: {save_path}")
    plt.show()


if __name__ == "__main__":
    execute_lyapunov_experiment()