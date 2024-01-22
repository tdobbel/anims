from typing import Protocol, Optional
import matplotlib.pyplot as plt
import numpy as np

MAX_SIZE = 11
plt.rcParams['font.family'] = 'Montserrat'

class Game(Protocol):

    def update(self) -> int:
        ...

    def get_grid_values(self) -> np.ndarray:
        ...

def visualize_plot(
    game: Game, max_iter: Optional[int] = None, show_grid: bool = True
) -> None:
    """
    Visualize the game of life using matplotlib.

    Parameters:
        game --- game of life object
        max_iter --- maximum number of iterations. If None, run until the game
                     reaches a steady state. (Default: None)
        show_grid --- if True, show the grid lines. (Default: True)
    """
    display_data = game.get_grid_values()
    ny, nx = display_data.shape
    ratio = ny / nx
    w, h = (MAX_SIZE, MAX_SIZE*ratio) if ratio < 1 else (MAX_SIZE/ratio, MAX_SIZE)
    fig, ax = plt.subplots(figsize=(w,h), num="Conway's game of Life")
    game_display = ax.pcolormesh(display_data, cmap='binary')
    if show_grid:
        for i in range(nx+1):
            ax.axvline(i, color='black', lw=0.1)
        for i in range(ny+1):
            ax.axhline(i, color='black', lw=0.1)
    ax.axis('off')
    text = ax.text(
        0, 1, 'Generation: 0', fontsize=12, color='white',
        fontweight='medium', ha='left', va='top', transform=ax.transAxes,
        bbox=dict(facecolor='navy', edgecolor='navy', pad=0)
    )
    ax.set_aspect('equal')
    fig.tight_layout()
    gen = 0
    max_iter = np.inf if max_iter is None else max_iter
    while game.update() > 0:
        gen += 1
        game_display.update({'array': display_data})
        game_display.set_linewidth(0.01)
        display_data = game.get_grid_values()
        text.set_text(f'Generation: {gen}')
        plt.draw()
        plt.pause(0.08)
        if gen > max_iter:
            break
    plt.show()
