import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class Mesh:
    def __init__(self, cosmo, box_size=1.0, resolution=64, dims=2, a=0.01):
        self.box_size = box_size
        self.resolution = resolution
        self.cell_size = box_size / resolution
        self.density = np.zeros((resolution,) * dims)
        self.dims = dims
        self.a = a
        self.h_0 = cosmo.h_0
        self.t = (2/(3 * self.h_0)) * (a ** (3/2)) #initial time at which a = given value
        self.h = cosmo.hubble(a)
        self.omega_m = cosmo.omega_m
        self.omega_lambda = cosmo.omega_lambda
    def deposit_NGP(self, particles):
        self.density.fill(0)
        indices = np.floor((particles.positions / self.cell_size)).astype(int)
        indices = indices % self.resolution
        np.add.at(self.density, tuple(indices.T), particles.mass)
        self.density /= (self.cell_size ** self.dims)

    def cic_weights(self, positions):
        dims = positions.shape[1]
        grid_positions = positions / self.cell_size
        grid_indices = np.floor(grid_positions).astype(int)
        directional_fractions = grid_positions - grid_indices
        offset_tuples = list(product([0, 1], repeat=dims))
        for offset in offset_tuples:
            weight = 1.0
            for dim, off in enumerate(offset):
                f = directional_fractions[:, dim]
                weight = weight * (f if off == 1 else (1 - f))
            idx = (grid_indices + offset) % self.resolution
            yield tuple(idx.T), weight

    def deposit_CIC(self, particles):
        self.density.fill(0)

        for idx, weight in self.cic_weights(particles.positions):
            np.add.at(self.density, idx, weight * particles.mass)

        self.density /= (self.cell_size ** self.dims)

    def solve_potential(self, G=1.0):
        density_k = np.fft.fftn(self.density-self.density.mean())
        density_k /= self.a ** 3
        k = np.fft.fftfreq(self.resolution, d=self.cell_size) * 2 * np.pi
        k_grids = np.meshgrid(*([k] * self.dims), indexing='ij')
        k_squared = sum(k_axis**2 for k_axis in k_grids)
        k_squared[(0,) * self.dims] = 1.0
        potential_k = -4 * np.pi * G * density_k / k_squared
        potential_k[(0,) * self.dims] = 0.0
        self.potential = np.fft.ifftn(potential_k).real
        return self.potential
    
    def compute_acceleration(self):
        acceleration_fields = []
        for axis in range(self.dims):
            acceleration_fields.append(-(np.roll(self.potential, -1, axis=axis)-np.roll(self.potential, 1, axis=axis)) / (2 * self.cell_size))
        self.acceleration_fields = acceleration_fields

    def interpolate_acceleration(self, particles):
        dims = particles.positions.shape[1]
        acceleration = np.zeros_like(particles.positions)
        for idx, weight in self.cic_weights(particles.positions):
            for axis in range(self.dims):
                acceleration[:, axis] += weight * self.acceleration_fields[axis][idx]
        particles.accelerations = acceleration

    def evolve_a(self, step_a_factor, cosmo):
        a_old = self.a
        a_new = a_old * step_a_factor
        da = a_new - a_old
        self.a = a_new # Stored as end of step value
        a_mid = (a_old + a_new) / 2
        self.a_dot = cosmo.a_dot(a_mid)
        dt = da / self.a_dot
        self.h = cosmo.hubble(a_mid) # Stored as mid-step value
        self.t += dt # Stored as end of step value
        self.current_dt = dt


class Particles:
    def __init__(self, mesh, resolution, mass=None,):
        self.dims = mesh.dims
        self.box_size = mesh.box_size
        self.resolution = resolution
        self.n_particles = resolution ** self.dims
        self.initial_a = mesh.a
        self.mass = 1.0/self.n_particles if mass is None else mass

    def initialize_particles(self, cosmo, seed=None, n=-2.0, delta_rms=0.01):
        rng = np.random.default_rng(seed)
        d = self.box_size / self.resolution #cell size
        k_axes = [np.fft.fftfreq(self.resolution, d=d) * 2 * np.pi] * (self.dims - 1)
        k_axes.append(np.fft.rfftfreq(self.resolution, d=d) * 2 * np.pi)
        k_grids = np.meshgrid(*k_axes, indexing='ij')
        k_magnitude = np.sqrt(sum(kx**2 for kx in k_grids))
        shape_k = k_magnitude.shape
        k_real = rng.normal(size=shape_k)
        k_im = rng.normal(size=shape_k)
        delta_k = k_real + 1j * k_im
        k_magnitude[(0,)*self.dims] = 1.0
        power_spectrum = k_magnitude ** n
        delta_k *= np.sqrt(power_spectrum)
        delta_k[(0,)*self.dims] = 0.0

        # rescale so the linear density field starts at the requested delta_rms
        delta_x = np.fft.irfftn(delta_k, s=(self.resolution,) * self.dims, axes=range(self.dims))
        scale = delta_rms / delta_x.std()

        psi = []
        for axis in range(self.dims):
            psi.append(scale * np.fft.irfftn(1j * k_grids[axis] * delta_k / k_magnitude ** 2,s=(self.resolution,) * self.dims, axes=range(self.dims)))

        q = (np.arange(self.resolution) + 0.5) * d
        grid = np.meshgrid(*([q] * self.dims), indexing='ij')
        self.positions = np.stack([g.ravel() for g in grid], axis=-1)

        displacement = np.stack([p.ravel() for p in psi], axis=-1)
        self.positions = (self.positions + displacement) % self.box_size

        self.velocities = cosmo.growth_rate(self.initial_a) * cosmo.hubble(self.initial_a) * displacement
        self.accelerations = np.zeros_like(self.positions)

class Cosmology:
    def __init__(self, omega_m=1.0, omega_lambda=0.0):
        self.omega_m = omega_m
        self.omega_lambda = omega_lambda
        self.h_0 = np.sqrt(8 * np.pi / (3 * self.omega_m))

    def hubble(self, a):
        return self.h_0 * np.sqrt(self.omega_m / a**3 + self.omega_lambda)

    def a_dot(self, a):
        return self.h_0 * np.sqrt((self.omega_m/a) + (self.omega_lambda * a ** 2))

    def growth_rate(self, a):
        return ((self.omega_m / a**3) / (self.omega_m / a ** 3 + self.omega_lambda)) ** 0.55


def recompute_acceleration(mesh, particles):
    mesh.deposit_CIC(particles)
    mesh.solve_potential()
    mesh.compute_acceleration()
    mesh.interpolate_acceleration(particles)

def step(mesh, particles, dt):
    particles.velocities *= np.exp(-2 * mesh.h * dt / 2)
    particles.velocities += particles.accelerations * (dt / 2)
    particles.positions += particles.velocities * dt
    particles.positions %= mesh.box_size
    recompute_acceleration(mesh, particles)
    particles.velocities += particles.accelerations * (dt / 2)
    particles.velocities *= np.exp(-2 * mesh.h * dt / 2)
    
def project(density, thickness=0.125):
    if density.ndim == 2:
        return density
    else:
        return density[:, :, int((density.shape[2] // 2) - (thickness * density.shape[2] // 2)):int((density.shape[2] // 2) + (thickness * density.shape[2] // 2))].mean(axis=2)

def run_simulation(n_steps, box_size=1.0, resolution=64, particle_resolution=None,
                   mass=None, dims=2, start_a=0.01, end_a=1.0,
                   omega_m=1.0, omega_lambda=0.0,
                   spectral_index=-2.0, delta_rms=0.01, seed=None):

    cosmo = Cosmology(omega_m=omega_m, omega_lambda=omega_lambda)
    m = Mesh(cosmo, box_size=box_size, resolution=resolution, dims=dims, a=start_a)

    if particle_resolution is None:
        particle_resolution = resolution
    p = Particles(m, resolution=particle_resolution, mass=mass)
    p.initialize_particles(cosmo, seed=seed, n=spectral_index, delta_rms=delta_rms)

    recompute_acceleration(m, p)

    step_a_factor = np.exp(np.log(end_a / start_a) / n_steps)
    n_rows, n_cols = 2, 4
    snapshot_interval = max(1, n_steps // (n_rows * n_cols))
    snapshots = {}
    for step_num in range(1, n_steps + 1):
        m.evolve_a(step_a_factor, cosmo)
        step(m, p, m.current_dt)
        if step_num % snapshot_interval == 0:
            print(f"Step {step_num}/{n_steps}   a = {m.a:.4f}")
            snapshots[step_num] = (m.a, m.density.copy())

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
                             constrained_layout=True)
    for ax in axes.flat:
        ax.set_axis_off()
    for ax, key in zip(axes.flat, sorted(snapshots)):
        a_snap, dens = snapshots[key]
        ax.imshow(np.log10(project(dens) + 0.1).T, origin="lower",
                  cmap="viridis", vmin=-1, vmax=1.65)
        ax.set_title(f"a = {a_snap:.3f}")
    plt.show()