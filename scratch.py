import numpy as np
import simulation as S
import matplotlib.pyplot as plt

m = S.Mesh()
p = S.Particles(box_size = m.box_size, n_particles = ((2*m.resolution)**2))

m.deposit_CIC(p)

m.solve_potential()
m.compute_acceleration()
m.interpolate_acceleration(p)
plt.imshow(m.density.T, origin="lower")
plt.show()
plt.imshow(m.potential.T, origin="lower")
plt.show()

plt.quiver(p.positions[::200,0], p.positions[::200,1], p.accelerations[::200,0], p.accelerations[::200,1])
plt.show()

