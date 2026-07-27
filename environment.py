# representation of the true world

from __future__ import annotations

import numpy as np

from config import FREE, OBSTACLE, VICTIM

class Environment:
    # stores map of the true world, obstacles, and victims

    def __init__(self, map_file):
        self.map_file = map_file

        self.grid = None
        self.height_map = None

        self.robot_start = None
        self.victim_position = None

        self.load_map(map_file)

    def load_map(self, map_file):
        with open(map_file, "r") as f:
            lines = [l.rstrip("\n") for l in f if l.strip()]

        # checks for valid map
        if len(lines) == 0:
            raise ValueError("Map file is empty")

        width = len(lines[0])

        for line in lines:
            if len(line) != width:
                raise ValueError("Unequal map widths")

        height = len(lines)

        self.grid = np.full((height, width), FREE, dtype = int)
        self.height_map = np.zeros((height, width), dtype = float)

        for row, l in enumerate(lines):
            for col, char in enumerate(l):
                self.load_cell(row, col, char)

        if self.robot_start is None:
                    raise ValueError("Map does not contain a robot start 'S'.")
        
        if self.victim_position is None:
            raise ValueError("Map does not contain a victim 'V'.")

    def load_cell(self, row, col, symbol):
        if symbol == ".":
            self.grid[row, col] = FREE

        elif symbol == "#":
            self.set_obstacle(row, col, 3.0)

        elif symbol == "r":
            self.set_obstacle(row, col, 1.5)

        elif symbol == "T":
            self.set_obstacle(row, col, 5.0)

        elif symbol == "S":
            self.grid[row, col] = FREE
            self.robot_start = (row, col)

        elif symbol == "V":
            self.grid[row, col] = VICTIM
            self.victim_position = (row, col)

        else:
            raise ValueError(f"Unknown map symbol '{symbol}' at row {row}, column {col}")

    def set_obstacle(self, row, col, height):
        self.grid[row, col] = OBSTACLE
        self.height_map[row, col] = height

    def is_in_bounds(self, row, col):
        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]

    def is_traversable(self, position: tuple[int, int]):
        row, column = position

        if self.is_in_bounds(position) and self.grid[row, column] != OBSTACLE:
            return True
        
        return False
    