import simulation as S
from simulation import run_simulation
import matplotlib.pyplot as plt

m, p = run_simulation(n_steps=200, box_size_mpc=2000, resolution=256, particle_resolution=512, omega_m = 0.315, omega_lambda=0.685, seed=42)
