# representation of the true world

from __future__ import annotations

import numpy as np

from config import (
    FREE,
    GRID_HEIGHT,
    GRID_WIDTH,
    OBSTACLE,
    VICTIM,
    VICTIM_POSITION,
)

class Environment:
    # stores map of the true world, obstacles, and victims

    def __init__(self):
        self.grid = np.full((GRID_HEIGHT, GRID_WIDTH),FREE)

        self._add_boundaries()
        self._add_internal_obstacles()
        self._place_victim()
        self.height_map = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=float)

    def set_obstacle(self, row, col, height):
        self.grid[row, col] = OBSTACLE
        self.height_map[row, col] = height

    def _add_boundaries(self):
        # add walls as obstacles on edges
        self.grid[0, :] = OBSTACLE
        self.grid[-1, :] = OBSTACLE
        self.grid[:, 0] = OBSTACLE
        self.grid[:, -1] = OBSTACLE

    def _add_internal_obstacles(self):
        # simulate obstacles/internal environment

        # horizontal wall w/ doorway
        self.grid[6, 3:24] = OBSTACLE
        self.grid[6, 12:15] = FREE

        # vertical wall w/ doorway
        self.grid[6:17, 20] = OBSTACLE
        self.grid[11:14, 20] = FREE

        # debris
        self.grid[13:16, 7:10] = OBSTACLE
        self.grid[2:5, 16:19] = OBSTACLE

    def _place_victim(self):
        row, column = VICTIM_POSITION

        if self.grid[row, column] == OBSTACLE:
            raise ValueError("victim can't be ins an obstacle")

        self.grid[row, column] = VICTIM

    def is_in_bounds(self, position: tuple[int, int]):
        row, column = position

        if 0 <= row < self.grid.shape[0] and 0 <= column < self.grid.shape[1]:
            return True
        
        return False

    def is_traversable(self, position: tuple[int, int]):
        row, column = position

        if self.is_in_bounds(position) and self.grid[row, column] != OBSTACLE:
            return True
        
        return False
    