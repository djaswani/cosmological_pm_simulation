import simulation as S
from simulation import initial_displacements
import matplotlib.pyplot as plt
field = initial_displacements(resolution=64, dims=2, box_size=1.0, n=-3, seed=42)

plt.imshow(field.T, origin="lower")
plt.show()