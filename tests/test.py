from time import sleep

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


coord = np.array([0.5, 0.5])

positions = []

fig, ax = plt.subplots(1, 1)

scatter_obj = ax.scatter(coord[0], coord[1], s=100)


def animate(t: int):

    coord[0] += (0.005 * np.sin(t)) + (0.005 * np.cos(t))
    coord[1] += (0.005 * np.sin(t)) - (0.005 * np.cos(t))
    scatter_obj.set_offsets([coord[0], coord[1]])
    sleep(0.1)

    return scatter_obj


anim = animation.FuncAnimation(fig, animate, interval=1, repeat=True)

plt.show()