# N-Body Numerical Analysis

A computational study investigating how numerical integrator choice affects
the quantitative onset of chaos in gravitational N-body systems.

The project evaluates both symplectic and non-symplectic integrators using
diagnostics such as energy drift and Lyapunov exponent estimation.

Bachelor's Thesis — Computer Science  
Athens Tech College, City College University of York Europe Campus

---

## Research Question

> How does integrator choice affect the quantitative onset of chaos in
> gravitational N-body systems?

---

## Features

- Gravitational N-body simulation engine
- Multiple numerical integrators
- Energy conservation diagnostics
- Long-term orbital stability analysis
- Experimental comparison framework
- Streamlit-based interactive visualization (planned)

---

## Integrators

### Implemented

- Euler (first-order, non-symplectic)
- Leapfrog / Störmer-Verlet (second-order, symplectic)
- RK4 (fourth-order, non-symplectic)
- Yoshida (fourth-order, symplectic)

### Planned

- Adaptive timestep methods
- Lyapunov exponent estimation

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Meeseeks01/nbody-numerical-analysis.git
cd nbody-numerical-analysis
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running Tests

Run the complete test suite:

```bash
python -m pytest tests/ -v
```

---

## Running Experiments

Example:

```bash
python -m experiments.01_energy_drift_comparison
```

Generated figures are stored in:

```text
dissertation/figures/
```

---

## Project Structure

```text
src/
├── physics/         # Body class and gravitational system mechanics
├── integrators/     # Numerical integration algorithms
├── diagnostics/     # Energy drift and chaos diagnostics

experiments/         # Reproducible experiment scripts
tests/               # Unit and integration tests
notebooks/           # Quarto/Jupyter research notebooks
app/                 # Streamlit interactive visualization
dissertation/        # Dissertation assets and generated figures
```

---

## Numerical Methods

This project compares integrators across several dimensions:

- Numerical stability
- Long-term energy conservation
- Computational cost
- Chaotic trajectory divergence
- Sensitivity to timestep size

Special focus is given to the distinction between:

- Symplectic integrators
- Non-symplectic integrators

and their suitability for long-term gravitational simulations.

---

## Current Experiments

- Energy drift comparison across integrators
- Long-term orbital stability analysis
- Timestep sensitivity experiments

Planned:

- Lyapunov exponent estimation
- Chaotic three-body simulations
- Interactive simulation controls

---

## Dependencies

Main libraries used:

- NumPy
- Matplotlib
- SciPy
- PyTest
- Streamlit
- Jupyter / Quarto

---

## License

This project is developed for academic and research purposes.