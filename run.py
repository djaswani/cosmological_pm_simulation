import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import simulation as S
from simulation import run_simulation

S.run_simulation(n_steps=200, box_size=1.0, resolution=256, mass=None, dims=2)