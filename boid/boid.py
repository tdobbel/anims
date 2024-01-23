from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree

DEFAULT_WEIGHTS = {
    'cohesion': 1.0,
    'alignment': 1.0,
    'wall': 0.8,
    'separation': 1.1,
    'obstacle': 1.0
}

def _distance_weight(dx: np.ndarray) -> np.ndarray:
    dist = np.linalg.norm(dx, axis=1)
    dx /= dist[:,None]
    weights = 1 / (dist + 1e-6)
    return np.sum(dx * weights[:,None], axis=0) / np.sum(weights)

@dataclass
class Obstacle:
    x: float
    y: float
    radius: float

    def plot(self, ax) -> None:
        angle = np.linspace(0, 2*np.pi, 101)
        xs = self.x + self.radius * np.cos(angle)
        ys = self.y + self.radius * np.sin(angle)
        ax.fill(xs, ys, color='white')

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x, self.y])

    def get_normal(self, pos: np.ndarray) -> np.ndarray:
        return pos - self.center[None,:]

    def get_distance(self, pos: np.ndarray) -> np.ndarray:
        return np.linalg.norm(self.get_normal(pos), axis=1)

class Domain:

    def __init__(
        self, minx: float, miny: float, maxx: float, maxy: float
    ) -> None:
        self.minx = minx
        self.miny = miny
        self.maxx = maxx
        self.maxy = maxy
        self.obstacles: list[Obstacle] = []

    def add_obstacle(self, obstacle: Obstacle) -> None:
        self.obstacles.append(obstacle)

    @property
    def bounds(self) -> None:
        return self.minx, self.miny, self.maxx, self.maxy

    def seed(self, n: int) -> np.ndarray:
        xs = np.random.uniform(self.minx, self.maxx, n)
        ys = np.random.uniform(self.miny, self.maxy, n)
        pos = np.vstack((xs,ys)).T
        for obstacle in self.obstacles:
            dx = obstacle.get_normal(pos)
            dist = np.linalg.norm(dx, axis=1)
            dx[dist > obstacle.radius] = 0
            amp = 1.1 * (obstacle.radius - dist)
            pos[:] += dx * amp[:,None] / dist[:,None]
        return pos

class Boids:

    def __init__(
        self, domain: Domain, n_boids: int, radius: float
    ) -> None:
        self.domain = domain
        self.positions = domain.seed(n_boids)
        angle = np.random.uniform(0, 2*np.pi, n_boids)
        minx, miny, maxx, maxy = self.domain.bounds
        self.vel_norm = max(maxy-miny, maxx-minx) / 100
        x_vel = self.vel_norm * np.cos(angle)
        y_vel = self.vel_norm * np.sin(angle)
        self.velocity = np.vstack((x_vel, y_vel)).T
        self.radius = radius
        self.weights = DEFAULT_WEIGHTS.copy()

    def get_positions(self) -> np.ndarray:
        return self.positions

    def set_positions(self, positions: np.ndarray) -> None:
        self.positions[:] = positions

    def get_velocities(self) -> np.ndarray:
        return self.velocity

    def _compute_interactions(
        self, pos: np.ndarray, isel: np.ndarray
    ) -> dict[str, np.ndarray]:
        if len(isel) == 0:
            return {
                'separation': np.zeros(2),
                'alignment': np.zeros(2),
                'cohesion': np.zeros(2)
            }
        return {
            'separation': _distance_weight(pos[None] - self.positions[isel]),
            'alignment': np.mean(self.velocity[isel], axis=0),
            'cohesion': np.mean(self.positions[isel], axis=0) - pos
        }

    def _compute_obstacles(self) -> np.ndarray:
        dx_tot = np.zeros((self.positions.shape[0],2))
        wtot = 0
        for obstacle in self.domain.obstacles:
            dx_obstacle = obstacle.get_normal(self.positions)
            dist = np.linalg.norm(dx_obstacle, axis=1)
            ignore = np.where(dist > self.radius + obstacle.radius)[0]
            dx_obstacle[ignore,:] = 0
            weight = 1 / (dist + 1e-6)
            dx_tot += weight[:,None] * dx_obstacle / (dist[:,None] + 1e-6)
            wtot += weight
        return dx_tot / wtot[:,None]

    def _compute_wall(self, pos: np.ndarray) -> np.ndarray:
        minx, miny, maxx, maxy = self.domain.bounds
        wall = np.array([0,0])
        if np.abs(pos[0] - minx) < self.radius:
            wall[0] = 1
        elif np.abs(pos[0] - maxx) < self.radius:
            wall[0] = -1
        if np.abs(pos[1] - miny) < self.radius:
            wall[1] = 1
        elif np.abs(pos[1] - maxy) < self.radius:
            wall[1] = -1
        return wall

    def _combine_velocities(self, velocities: dict[str, np.ndarray]) -> np.ndarray:
        assert all(key in velocities for key in self.weights)
        wtot = sum(self.weights.values())
        res = 0
        for key,vel in velocities.items():
            res += vel * self.weights[key] / (wtot * (np.linalg.norm(vel) + 1e-6))
        return res

    def compute_dx(self, dt: float) -> np.ndarray:
        new_vel = np.zeros_like(self.velocity)
        vel_obstacle = self._compute_obstacles()
        # critical_obstacle = np.any([
        #     o.get_distance(self.positions) < CRITICAL_OBSTACLE \
        #         for o in self.domain.obstacles
        # ], axis=0)
        for i,pos in enumerate(self.positions):
            # self.weights = DEFAULT_WEIGHTS.copy()
            dist = np.linalg.norm(self.positions - pos[None], axis=1)
            within_radius = dist < self.radius
            within_radius[i] = False
            # critical_separation = np.any(dist[within_radius] < CRITICAL_SEPARATION)
            # if critical_separation:
            #     self.weights['separation'] *= 1.5
            # if critical_obstacle[i]:
            #     self.weights['obstacle'] *= 2
            isel = np.where(within_radius)[0]
            extra_vel = self._compute_interactions(pos, isel)
            extra_vel['wall'] = self._compute_wall(pos)
            extra_vel['obstacle'] = vel_obstacle[i]
            extra_vel = self._combine_velocities(extra_vel)
            new_vel[i] += self.velocity[i] + self.vel_norm * extra_vel
        new_vel /= np.linalg.norm(new_vel, axis=1)[:,None] + 1e-6
        self.velocity = self.vel_norm * new_vel
        return dt * self.velocity

    def move(self, dx: np.ndarray) -> None:
        self.positions += dx
