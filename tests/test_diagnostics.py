import numpy as np
from src.diagnostics.energy import relative_energy_error, energy_error

def test_relative_energy_error():
    assert relative_energy_error(10, 11) == 0.1