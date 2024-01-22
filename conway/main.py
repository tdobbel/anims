import argparse
import numpy as np
from game import Game
from visualizer import visualize_plot
from rules import (
    underpopulation, overpopulation, reproduction
)

N_ROW = 100
N_COL = 100
MAX_ITER = 100

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrow', type=int, default=N_ROW)
    parser.add_argument('--ncol', type=int, default=N_COL)
    parser.add_argument('--nseed', type=int, default=None)
    parser.add_argument('--max_iter', type=int, default=MAX_ITER)
    parser.add_argument('--no_grid', action='store_false', dest='show_grid')
    args = parser.parse_args()
    if args.nseed is None:
        args.nseed = args.nrow * args.ncol // 6
    rules = [underpopulation, overpopulation, reproduction]
    game = Game(args.nrow, args.ncol, rules)
    rows = np.random.randint(0, args.nrow, size=args.nseed)
    cols = np.random.randint(0, args.ncol, size=args.nseed)
    for row, col in zip(rows, cols):
        game.activate_cell(row, col)
    visualize_plot(game, args.max_iter, show_grid=args.show_grid)

if __name__ == '__main__':
    main()
