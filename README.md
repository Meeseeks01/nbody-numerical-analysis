# N-Body Numerical Analysis

A computational study of how numerical integrator choice affects 
the onset of chaos in gravitational N-body systems, measured via 
Lyapunov exponents and energy drift.

Bachelor's Thesis — Computer Science  
Athens Tech College, City College University of York Europe Campus

## Research Question

How does integrator choice affect the quantitative onset of chaos 
in N-body systems?

## Installation

git clone https://github.com/Meeseeks01/nbody-numerical-analysis.git
cd nbody-numerical-analysis
pip install -r requirements.txt

## Running Tests

Run tests
- python -m pytest tests/ -v

## Project Structure

src/physics/       — Body class and gravitational physics engine
src/integrators/   — Numerical integrators (Euler, Leapfrog, RK4, Yoshida)
src/diagnostics/   — Chaos metrics (Lyapunov exponents, energy drift)
experiments/       — Experiment scripts and results
notebooks/         — Quarto research notebooks
app/               — Streamlit interactive application
tests/             — Unit and integration tests

## Integrators Implemented

- Euler (first-order, non-symplectic)
- Leapfrog / Störmer-Verlet (second-order, symplectic)
- RK4 (fourth-order, non-symplectic) [in progress]
- Yoshida (fourth-order, symplectic) [in progress]

## Dependencies

numpy, matplotlib, scipy, pytest, streamlit, jupyter