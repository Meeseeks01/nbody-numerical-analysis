import numpy as np
import matplotlib.pyplot as plt

from src.physics.body import Body
from src.integrators.euler import euler_integrator
from src.integrators.rk4 import rk4_integrator
from src.integrators.leapfrog import leapfrog_integrator
from src.integrators.yoshida import yoshida_integrator


