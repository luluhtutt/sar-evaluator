import numpy as np
from config import (
    FREE,
    OBSTACLE,
    UNKNOWN
)

class OccupancyMap:

    def __init__(self, height, width):
        self.grid = np.full((height, width), UNKNOWN, dtype=np.int8)
        self.detected_victims = set()

    def update_from_sensor(self, environment, robot_position, sensor_range):
        # update cells near robot from sensors

        robot_row, robot_col = robot_position

        for row_offset in range(-sensor_range, sensor_range + 1):
            for col_offset in range(-sensor_range, sensor_range + 1):

                row = robot_row + row_offset
                col = robot_col + col_offset

                if not environment.is_in_bounds(row, col):
                    continue

                distance_squared = row_offset ** 2 + col_offset ** 2

                if distance_squared > sensor_range ** 2:
                    continue

                self.observe_cell(environment, row, col)

    def observe_cell(self, environment, row, col):
        # copy an observed terrain cell into the occupancy map

        self.grid[row, col] = environment.grid[row, col]

        victim = environment.get_victim_at_position((row, col))

        if victim is not None:
            victim.detected = True
            self.detected_victims.add(victim.position)

    def mark_victim_detected(self, environment, position):
        victim = environment.get_victim_at_position(position)

        if victim is None:
            return False

        victim.detected = True
        self.detected_victims.add(victim.position)

        row, col = position
        self.grid[row, col] = FREE

        return True

    def get_detected_victims(self):
        return sorted(self.detected_victims)

    def get_unreached_detected_victims(self, environment):
        victims = []

        for victim in environment.victims:
            if victim.detected and not victim.reached:
                victims.append(victim.position)

        return victims

    def count_detected_victims(self, environment):
        return sum(victim.detected for victim in environment.victims)

    def is_in_bounds(self, position):
        row, col = position

        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]

    def is_traversable(self, position):
        if not self.is_in_bounds(position):
            return False

        row, col = position

        return self.grid[row, col] == FREE

    def is_known_traversable(self, position):
        if not self.is_in_bounds(position):
            return False

        row, col = position

        return self.grid[row, col] == FREE

    def find_known_victim(self, environment):
        victims = self.get_unreached_detected_victims(environment)

        if len(victims) == 0:
            return None

        return victims[0]

    def count_known_cells(self):
        return int(np.sum(self.grid != UNKNOWN))

    def count_total_cells(self):
        return self.grid.size

    def percent_explored(self):
        known_cells = self.count_known_cells()
        total_cells = self.count_total_cells()

        return 100.0 * known_cells / total_cells
    