def energy_error(initial_energy, current_energy):
    """
    Compute the absolute energy error relative to the initial energy.

    Returns the magnitude of the deviation and does not preserve the sign
    of the energy change.
    """
    return abs(current_energy - initial_energy)


def relative_energy_error(initial_energy, current_energy):
    """
    Compute the relative energy error.

    Returns the absolute fractional deviation from the initial energy:
    |E - E0| / |E0|.
    """
    return abs((current_energy - initial_energy) / initial_energy)
