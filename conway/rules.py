import numpy as np

def underpopulation(
    pixels: np.ndarray, active: np.ndarray, n_active_neighbours: np.ndarray
) -> tuple[np.ndarray,np.ndarray]:
    """If a cell is alive and has less than 2 active neighbours, it dies."""
    deactivate = (active == 1) & (n_active_neighbours < 2)
    return pixels[deactivate], np.zeros(deactivate.sum(), dtype=np.uint0)

def overpopulation(
    pixels: np.ndarray, active: np.ndarray, n_active_neighbours: np.ndarray
) -> tuple[np.ndarray,np.ndarray]:
    """If a cell is alive and has more than 3 active neighbours, it dies."""
    deactivate = (active == 1) & (n_active_neighbours > 3)
    return pixels[deactivate], np.zeros(deactivate.sum(), np.uint0)

def reproduction(
    pixels: np.ndarray, active: np.ndarray, n_active_neighbours: np.ndarray
) -> tuple[np.ndarray,np.ndarray]:
    """If a cell is dead and has exactly 3 active neighbours, it becomes alive."""
    activate = (active == 0) & (n_active_neighbours == 3)
    return pixels[activate], np.ones(activate.sum(), dtype=np.uint0)
