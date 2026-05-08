def energy_error(initial_energy, current_energy):
    return abs(current_energy - initial_energy)


def relative_energy_error(initial_energy, current_energy):
    return abs((current_energy - initial_energy) / initial_energy)
