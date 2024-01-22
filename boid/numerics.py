from typing import Protocol
import numpy as np


class Tracker(Protocol) :

    def get_positions(self) -> np.ndarray:
        ...

    def set_positions(self) -> None:
        ...

    def move(self, dx: np.ndarray) -> None:
        ...


class ERK44 :
    """Class to integrate the displacement of particles based
    on current velocity using an explicit Runge-Kutta 44 scheme
    """

    def __init__(self) :
        self._a = [[0  , 0  , 0,  0],
                   [0.5, 0  , 0,  0],
                   [0  , 0.5, 0,  0],
                   [0  , 0  , 1,  0]]
        self._b = [1./6, 1./3, 1./3, 1./6]
        self._c = [0, 0.5, 0.5, 1]
        self.nsub = 4
        self.isub = 0
        self._d = [None, None, None, None]

    def sub_time_step(self, tracker: Tracker, dx: np.ndarray) :
        self._d[self.isub] = dx.copy()
        if self.isub == 0 :
            self._state0 = tracker.get_positions().copy()
        else :
            tracker.set_positions(self._state0.copy())
        self.isub += 1
        if self.isub != self.nsub :
            d = sum([self._a[self.isub][i]*self._d[i] for i in range(self.isub)])
            tracker.move(d)
        else :
            self.isub = 0

    def end_time_step(self, tracker: Tracker) -> np.ndarray :
        d = sum([self._b[i]*self._d[i] for i in range(self.nsub)])
        tracker.move(d)
