import os
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import LineCollection
import numpy as np
from numpy.typing import NDArray

MAX_ANGLE = np.pi / 2

os.makedirs("frames", exist_ok=True)


class DragonIteration:

    def __init__(self, dragon: NDArray):
        self.dragon = dragon
        self.tail = dragon[-1]
        vec = dragon - self.tail[None]
        self.norm = np.hypot(vec[:,0], vec[:,1])
        self.phi0 = np.arctan2(vec[:,1], vec[:,0])

    def rotate(self, phi: float) -> NDArray:
        angle = self.phi0 + phi
        extra_dragon = np.empty((self.dragon.shape[0], 2))
        extra_dragon[:,0] = self.norm * np.cos(angle) + self.tail[0]
        extra_dragon[:,1] = self.norm * np.sin(angle) + self.tail[1]
        return extra_dragon[::-1]

n_iter = 21
n_sub_iter = 8
step = 0.5*np.pi / n_sub_iter
max_iter = n_iter * n_sub_iter
n_frames = max_iter + 5

cm = plt.get_cmap("rainbow")
colors = cm(np.linspace(0, 1, n_iter+1))

dragon = np.array([[0, -0.42], [0, 0.42]])
dragon_iter = DragonIteration(dragon)
args = [0, dragon_iter, 1.0]

fig, ax = plt.subplots(figsize=(5,5))

ax.add_collection(LineCollection([dragon], color=colors[0], linewidth=.6))
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_aspect('equal')
ax.set_frame_on(False)
ax.set_xticks([])
ax.set_yticks([])

def animate(iiter: int) -> None:
    if 0 < iiter < max_iter:
        phi, dragon_iter, scale0 = args
        if phi == 0:
            extra_dragon = dragon_iter.rotate(0)
            lc = LineCollection([extra_dragon], color=colors[iiter//n_sub_iter+1], linewidth=.6)
            ax.add_collection(lc)
        phi = min(phi+step, MAX_ANGLE)
        scale = scale0 * (1 + phi / MAX_ANGLE * (np.sqrt(2)-1))
        extra_dragon = dragon_iter.rotate(phi)
        lc = ax.collections[-1]
        lc.set_segments([extra_dragon])
        ax.set_xlim(-scale, scale)
        ax.set_ylim(-scale, scale)
        # points = lc.get_datalim(ax.transData)
        # ax.update_datalim(points)
        # ax.autoscale()
        if phi >= MAX_ANGLE:
            dragon = np.concatenate([dragon_iter.dragon, extra_dragon[1:]], axis=0)
            dragon_iter = DragonIteration(dragon)
            scale0 = scale
            phi = 0
        args[2] = scale0
        args[1] = dragon_iter
        args[0] = phi
    plt.savefig(f"frames/frame_{iiter:04d}.png", dpi=200)

ani = animation.FuncAnimation(fig, animate, frames=n_frames, interval=30, repeat=False)
plt.show()
