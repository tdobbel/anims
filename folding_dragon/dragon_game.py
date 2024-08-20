import os
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import pygame

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
MAX_ANGLE = np.pi / 2

def rotate(dragon: NDArray, phi: float) -> NDArray:
    tail = dragon[-1]
    vec = dragon - tail[None]
    norm = np.hypot(vec[:,0], vec[:,1])
    phi0 = np.arctan2(vec[:,1], vec[:,0])
    angle = phi0 + phi
    extra_dragon = np.empty((dragon.shape[0], 2))
    extra_dragon[:,0] = norm * np.cos(angle) + tail[0]
    extra_dragon[:,1] = norm * np.sin(angle) + tail[1]
    return extra_dragon[::-1]

def scale_points(x0: int, y0: int, scale_factor: float, dragon: NDArray) -> NDArray:
    scaled_dragon = np.empty_like(dragon)
    scaled_dragon[:,0] = x0 + (dragon[:,0]-x0) / scale_factor
    scaled_dragon[:,1] = y0 + (dragon[:,1]-y0) / scale_factor
    return scaled_dragon

width, height = 400, 300
pygame.init()
screen = pygame.display.set_mode((width, height))

xo = np.array([width//2, height//2])
dragon = np.array([xo+np.array([0, 65]), xo+np.array([0,-65])])
scale = 1.0

phi = 0
iiter = 0
max_iter = 16
step = 0.005
scale0 = 1
acceleration = 0.05

iframe = 0

cmap = plt.get_cmap('rainbow')
colors = cmap(np.linspace(0, 1, max_iter+1))
segments = [dragon]

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            break
    if iiter >= max_iter:
        continue
    phi = min(MAX_ANGLE, phi+step)
    xi = phi / MAX_ANGLE
    scale = scale0 * (1-xi) + xi * scale0 * np.sqrt(2)
    screen.fill(WHITE)
    extra_dragon = rotate(dragon, -phi)
    for ip,points in enumerate(segments + [extra_dragon]):
        color = tuple(int(c*255) for c in colors[ip, :3])
        pygame.draw.lines(screen, color, False, scale_points(*xo, scale, points), 2)
    pygame.display.update()
    if phi >= np.pi/2:
        segments.append(extra_dragon)
        dragon = np.concatenate([dragon, extra_dragon[1:]], axis=0)
        iiter += 1
        phi = 0
        scale0 = scale
        step *= 1 + acceleration
    # pygame.image.save(screen, f'snapshots/frame_{iframe:04d}.png')

