from dataclasses import dataclass
import itertools
import matplotlib.pyplot as plt
import numpy as np

WEIGHTS = {
    'cohesion': 1.0,
    'alignment': 1.0,
    'wall': 0.8,
    'separation': 1.1,
    'obstacle': 1.0
}


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

    def _find_neighbours(self) -> None:
        rows, cols = self.domain.get_coordinates(self.positions)
        # reset neighbors
        for boid in self.boids:
            boid.reset_neighbors()
        shifts = [(0,0), (0,1), (1,0), (1,1)]
        iterables = [range(self.domain.ny), range(self.domain.nx)]
        for row,col in itertools.product(*iterables):
            boid_ids = np.where((cols == col) & (rows == row))[0]
            for shift_r, shift_c in shifts:
                other_ids = np.where(
                    (cols == col + shift_c) & (rows == row + shift_r)
                )[0]
                for i,j in itertools.product(boid_ids, other_ids):
                    if i == j:
                        continue
                    dist = np.linalg.norm(self.boids[i].pos - self.boids[j].pos)
                    if dist > self.domain.radius:
                        continue
                    self.boids[i].add_neighbour(j, dist)
                    self.boids[j].add_neighbour(i, dist)

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
