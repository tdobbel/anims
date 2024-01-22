import matplotlib.pyplot as plt
import numpy as np
from slimutil.visualization import fig_size_from_bounds
from boid import Domain, Boids, Obstacle
from numerics import ERK44


minx, miny = 0, 0
maxx, maxy = 320, 180

w, h = fig_size_from_bounds(11, minx, maxx, miny, maxy)
dt = 0.3

rk = ERK44()
domain = Domain(minx, miny, maxx, maxy)
obstacles = [
    Obstacle(0.65*(maxx+minx), 0.66*(miny+maxy), 5),
    Obstacle(0.15*(maxx+minx), 0.3*(miny+maxy), 2),
    Obstacle(0.35*(maxx+minx), 0.85*(miny+maxy), 3),
    Obstacle(0.75*(maxx+minx), 0.30*(miny+maxy), 10),
    Obstacle(0.90*(maxx+minx), 0.90*(miny+maxy), 1),
    Obstacle(0.10*(maxx+minx), 0.65*(maxy+miny), 1.5)
]
for o in obstacles:
    domain.add_obstacle(o)
boids = Boids(domain, 600, 10)

fig, ax = plt.subplots(num='Boids', figsize=(w, h), facecolor='k')
ax.set_xlim(minx, maxx)
ax.set_ylim(miny, maxy)
ax.set_aspect('equal')
pos = boids.get_positions().T
vel = boids.get_velocities().T
ax.axis('off')
ax.set_facecolor('k')
cmap = plt.get_cmap('Blues')
colors = cmap(np.linspace(0.0, 0.7, pos.shape[1]))
fig.tight_layout()
for o in domain.obstacles:
    o.plot(ax)
scatter = ax.scatter(pos[0], pos[1], s=10, marker='o', color=colors)
scale_units = 'width' if maxx-minx < maxy-miny else 'height'
arrows =  ax.quiver(
    pos[0], pos[1], vel[0], vel[1], angles='xy',
    scale_units=scale_units, scale=150, color=colors
)
for _ in range(10_000):
    for _ in range(rk.nsub):
        rk.sub_time_step(boids, boids.compute_dx(dt))
    rk.end_time_step(boids)
    scatter.set_offsets(boids.get_positions())
    arrows.set_offsets(boids.get_positions())
    arrows.set_UVC(*boids.get_velocities().T)
    plt.draw()
    plt.pause(0.01)
plt.show()
