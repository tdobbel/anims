from dataclasses import dataclass
import itertools
from functools import cmp_to_key
import matplotlib.pyplot as plt
import numpy as np

WEIGHTS = {
    'cohesion': 1.0,
    'alignment': 1.0,
    'wall': 0.8,
    'separation': 1.1,
    'obstacle': 1.0
}


def _compare(vec1: np.ndarray, vec2: np.ndarray) -> int:
    if vec1[0] == vec2[0]:
        return np.clip(vec1[1] - vec2[1], -1, 1)
    return np.clip(vec1[0] - vec2[0], -1, 1)

def normalize_vector(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec
    return vec / norm

def combine_vectors(weigth_vec: list[tuple[float, np.ndarray]]) -> np.ndarray:
    wtot = 0
    res = np.zeros(2)
    for weight, vec in weigth_vec:
        wtot += weight
        res += weight * normalize_vector(vec)
    return res / wtot

class Boid:

    def __init__(self, x: float, y: float, vel_norm: float) -> None:
        self.pos = np.array([x, y])
        self.neighbors = []
        self.distances = []
        angle = np.random.uniform(0, 2 * np.pi)
        self.vel = vel_norm * np.array([np.sin(angle), np.cos(angle)])

    def reset_neighbors(self) -> None:
        self.neighbors = []
        self.distances = []

    def add_neighbour(self, boid_id: int, distance: float) -> None:
        self.neighbors.append(boid_id)
        self.distances.append(distance)

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def plot(self, ax) -> None:
        ax.add_patch(
            plt.Circle(
                (self.x, self.y), self.radius, color='white',fill=True
            )
        )

class Domain:

    def __init__(
        self, minx: float, miny: float, maxx: float, maxy: float, radius: float
    ) -> None:
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.radius = radius
        self.x_grid = np.arange(minx-radius, maxx+1.9*radius, radius)
        self.y_grid = np.arange(miny-radius, maxy+1.9*radius, radius)
        self.nx = len(self.x_grid)-1
        self.ny = len(self.y_grid)-1
        self.obstacles: list[Obstacle] = []

    def seed(self, n_boids: int) -> list[Boid]:
        xs = np.random.uniform(self.minx, self.maxx, n_boids)
        ys = np.random.uniform(self.miny, self.maxy, n_boids)
        vel_norm = self.velocity_norm
        return [Boid(x, y, vel_norm) for x, y in zip(xs, ys)]

    def get_coordinates(self, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rows = np.searchsorted(self.y_grid, pos[:,1], side='right') - 1
        cols = np.searchsorted(self.x_grid, pos[:,0], side='right') - 1
        return rows, cols

    @property
    def velocity_norm(self) -> float:
        return max(self.maxx - self.minx, self.maxy - self.miny) / 100

    def wall_repulsion(self, pos: np.ndarray) -> np.ndarray:
        vec_wall = np.zeros_like(pos)
        vec_wall[np.abs(pos[:,0]-self.minx) < self.radius,0] = 1
        vec_wall[np.abs(pos[:,0]-self.maxx) < self.radius,0] = -1
        vec_wall[np.abs(pos[:,1]-self.miny) < self.radius,1] = 1
        vec_wall[np.abs(pos[:,1]-self.maxy) < self.radius,1] = -1
        return vec_wall

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    def obstacle_repulsion(self, pos: np.ndarray) -> np.ndarray:
        vec_obs = np.zeros_like(pos)
        wtot = np.zeros((pos.shape[0],1))
        for obs in self.obstacles:
            dx_obstacle = pos - obs.pos[None]
            dist = np.linalg.norm(dx_obstacle, axis=1)[:,None]
            dx_obstacle /= dist
            dx_obstacle[dist[:,0] > self.radius+obs.radius,:] = 0
            weight = 1 / (dist + 1e-6)
            vec_obs += weight * dx_obstacle
            wtot += weight
        return vec_obs / wtot

class Flock:

    def __init__(self, domain: Domain, n_boids: int) -> None:
        self.domain = domain
        self.boids = domain.seed(n_boids)

    @property
    def positions(self) -> np.ndarray:
        return np.array([b.pos for b in self.boids])

    @property
    def velocities(self) -> np.ndarray:
        return np.array([b.vel for b in self.boids])

    def _add_pair(self, bid1: int, bid2: int, dist: float) -> int:
        if bid1 == bid2:
            return 0
        self.boids[bid1].add_neighbour(bid2, dist)
        self.boids[bid2].add_neighbour(bid1, dist)
        return 1

    def _find_neighbours(self) -> None:
        boid_pos = self.positions
        rows, cols = self.domain.get_coordinates(boid_pos)
        pairs = np.unique(np.array([rows, cols]), axis=1).T
        pairs = sorted(pairs, key=cmp_to_key(_compare))
        # reset neighbors
        _ = [ boid.reset_neighbors() for boid in self.boids ]
        iterables = [range(self.domain.ny), range(self.domain.nx)]
        for row,col in itertools.product(*iterables):
            select = (cols >= col) & (cols <= col+1) & (rows >= row) & (rows <= row+1)
            other_ids = np.where(select)[0]
            select[select] = (cols[select] == col) & (rows[select] == row)
            boid_ids = np.where(select)[0]
            for bid in boid_ids:
                dist = np.linalg.norm(boid_pos[other_ids] - boid_pos[[bid]], axis=1)
                within_radius = dist < self.domain.radius
                if not np.any(within_radius):
                    continue
                oids = other_ids[within_radius]
                add_iter = (
                    self._add_pair(bid,oid,d) for oid,d in zip(oids,dist[within_radius])
                )
                np.fromiter(add_iter, count=len(oids), dtype=np.int)

    def compute_dx(self, dt: float) -> np.ndarray:
        pos = self.positions
        vel = self.velocities
        wall_repulsion = self.domain.wall_repulsion(pos)
        obst_replusion = self.domain.obstacle_repulsion(pos)
        vel_norm = self.domain.velocity_norm
        self._find_neighbours()
        for i,boid in enumerate(self.boids):
            weight_vecs = [
                (WEIGHTS['wall'], wall_repulsion[i]),
                (WEIGHTS['obstacle'], obst_replusion[i]),
            ]
            if len(boid.neighbors) > 0:
                cohesion_vec = np.mean(pos[boid.neighbors], axis=0) - boid.pos
                alignment_vec = np.mean(vel[boid.neighbors], axis=0)
                dist = np.array(boid.distances)[:,None]
                sep_vecs = (boid.pos[None] - pos[boid.neighbors]) / (dist+1e-6)
                weigths = 1 / (dist + 1e-6)
                separation_vec = np.sum(sep_vecs * weigths, axis=0) / np.sum(weigths)
                weight_vecs += [
                    (WEIGHTS['cohesion'], cohesion_vec),
                    (WEIGHTS['alignment'], alignment_vec),
                    (WEIGHTS['separation'], separation_vec),
                ]
            extra_vel = vel_norm * combine_vectors(weight_vecs)
            boid.vel = vel_norm * normalize_vector(boid.vel + extra_vel)
        return self.velocities * dt

    def set_positions(self, positions: np.ndarray) -> None:
        for pos,boid in zip(positions, self.boids):
            boid.pos = pos.copy()

    def move(self, dx: np.ndarray) -> None:
        for boid,d in zip(self.boids, dx):
            boid.pos += d
