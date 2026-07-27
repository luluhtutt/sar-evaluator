import numpy as np
from config import FREE, OBSTACLE

class Victim:
    def __init__(self, row, col):
        self.position = (row, col)
        self.detected = False
        self.reached = False

class Environment:
    def __init__(self, map_filename):
        self.grid = []
        self.height_map = []

        self.robot_start = None
        self.victims = []

        self.load_map(map_filename)

    def load_map(self, map_filename):
        with open(map_filename, "r") as file:
            lines = [line.rstrip() for line in file]

        if len(lines) == 0:
            raise ValueError("Map file is empty")

        expected_width = len(lines[0])

        for line in lines:
            if len(line) != expected_width:
                raise ValueError("Every map row must have the same width")

        for row, line in enumerate(lines):
            grid_row = []
            height_row = []

            for col, cell in enumerate(line):

                if cell == "#":
                    grid_row.append(OBSTACLE)
                    height_row.append(3)

                else:
                    grid_row.append(FREE)
                    height_row.append(0)

                    if cell == "S":
                        self.robot_start = (row, col)

                    elif cell == "V":
                        self.victims.append(Victim(row, col))

                    elif cell == "T":
                        height_row[-1] = 2

                    elif cell == "r":
                        height_row[-1] = 1

            self.grid.append(grid_row)
            self.height_map.append(height_row)

        self.grid = np.array(self.grid, dtype=np.int8)
        self.height_map = np.array(self.height_map, dtype=np.int8)

        if self.robot_start is None:
            raise ValueError("Map must contain one robot start marked with S")

        if len(self.victims) == 0:
            raise ValueError("Map must contain at least one victim marked with V")

    def get_victim_positions(self):
        return [victim.position for victim in self.victims]

    def get_victim_at_position(self, position):
        for victim in self.victims:
            if victim.position == position:
                return victim

        return None

    def is_in_bounds(self, row, col):
        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]

    def count_detected_victims(self):
        return sum(victim.detected for victim in self.victims)

    def count_reached_victims(self):
        return sum(victim.reached for victim in self.victims)
    