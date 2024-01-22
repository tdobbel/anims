from dataclasses import dataclass
from typing import Union, Callable
import itertools
import numpy as np

Index = Union[int, np.ndarray]
Rule = Callable[[np.ndarray, np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]

class Game:
    """Class to represent the game of life."""

    def __init__(
        self, n_row: int, n_col: int, rules: list[Rule]
    ) -> None:
        """
        Initialize the game.

        Parameters:
            n_row --- number of rows in the grid
            n_col --- number of columns in the grid
            rules --- list of functions that implement the rules of the game
        """
        self.n_row = n_row
        self.n_col = n_col
        self.grid = np.zeros((n_row, n_col), dtype=np.uint0)
        dx = (-1,0,1)
        self.neighbours = [ (x,y) for x,y in itertools.product(dx,dx) if any([x,y]) ]
        self.rules = rules

    def _coords_to_pixel(self, row: Index, col: Index) -> Index:
        """Convert row and column indices to pixel indices."""
        return row * self.n_col + col

    def get_grid_values(self) -> np.ndarray:
        """Return the current state of the grid."""
        return self.grid

    def _pixel_to_coords(self, pixel: Index) -> tuple[Index, Index]:
        """Convert pixel indices to row and column indices."""
        return np.divmod(pixel, self.n_col)

    def activate_cell(self, row: int, col) -> None:
        """Activate a cell."""
        self.grid[row,col] = 1

    def count_active_neighbours(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Count the number of active neighbours around active cells and inactive cells
        adjacent to active cells.

        Returns:
            pixels --- the indices of the active and inactive cells
            active --- the state of the active and inactive cells
            n_active_neighbours --- the number of active neighbours for each cell of pixels
        """
        rows, cols = np.where(self.grid > 0)
        active_ids = self._coords_to_pixel(rows, cols)
        n_active_around_active = np.zeros(len(rows), dtype=np.uint8)
        inactive_ids = []
        for shift_r,shift_c in self.neighbours:
            irow = (rows + shift_r) % self.n_row
            irow[irow < 0] += self.n_row
            icol = (cols + shift_c) % self.n_col
            icol[icol < 0] += self.n_col
            active = self.grid[irow, icol]
            n_active_around_active += active
            inactive_ids.append(self._coords_to_pixel(irow[active == 0], icol[active == 0]))
        inactive_ids = np.concatenate(inactive_ids)
        inactive_ids, n_active_around_inactive = np.unique(inactive_ids, return_counts=True)

        pixels = np.concatenate((active_ids, inactive_ids))
        active = self.grid[self._pixel_to_coords(pixels)]
        n_active_neighbours = np.concatenate((n_active_around_active, n_active_around_inactive))

        return pixels, active, n_active_neighbours

    def update(self) -> int:
        """
        Update the grid according to the rules of the game.

        Returns:
            n_changed --- the number of cells that changed state
        """
        args = self.count_active_neighbours()
        updates = [rule(*args) for rule in self.rules]
        n_changed = 0
        for pixels, new_state in updates:
            n_changed += len(pixels)
            self.grid[self._pixel_to_coords(pixels)] = new_state
        return n_changed
