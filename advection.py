import numpy as np
import matplotlib.pyplot as plt

def advect(n_cells=64, box_size=1.0, v=1.0, C = 0.5, initial_condition='square', plot=False):
    cell_size = box_size / n_cells
    x = (np.arange(n_cells) + 0.5) * cell_size
    if initial_condition == 'square':
        q = np.zeros(n_cells)
        q[(int(np.ceil(n_cells / 2 - n_cells * 0.125))):int((np.floor(n_cells / 2 + n_cells * 0.125)))] = 1.0
    elif initial_condition == 'sine':
        q = np.sin(2 * np.pi * x)
    dt = C * cell_size / abs(v)
    q_start = q.copy()
    for _ in range(int((box_size / v) / dt)):
        slope = np.where((np.sign(q - np.roll(q, 1)) == np.sign(np.roll(q, -1) - q)), 
                         np.minimum(np.abs((q - np.roll(q, 1))), np.abs((np.roll(q, -1) - q))) * np.sign(q - np.roll(q, 1)),
                           0)
        f = v * (q + 0.5 * (1-C) * slope) # Better flux estimation via sloped cell values
        q = q - dt / cell_size * (f - np.roll(f, 1))

    q_end = q.copy()
    if plot == True:
        plt.plot(x, q_start, label="start")
        plt.plot(x, q_end, label="end")
        plt.legend()
        plt.show()
        print(f"Sum: {sum(q)}")
        print(f"Min: {np.min(q)}")
        print(f"Max: {np.max(q)}")
    return np.abs(q_end - q_start).mean()

for n_cells in [32, 64, 128, 256]:
    print(np.log2(advect(n_cells=n_cells, initial_condition='sine')/advect(n_cells=2*n_cells, initial_condition='sine')))