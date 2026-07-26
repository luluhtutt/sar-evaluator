import numpy as np

from config import (
    FREE,
    OBSTACLE,
    UNKNOWN,
)

class OccupancyMap:

    def __init__(self, height, width):
        self.grid = np.full((height, width), UNKNOWN, dtype=np.int8)

    def update_from_sensor(self, environment, robot_position, sensor_range):
        # update cells near robot from sensors

        robot_row, robot_col = robot_position

        for row_offset in range(-sensor_range, sensor_range + 1):
            for col_offset in range(-sensor_range, sensor_range + 1):

                row = robot_row + row_offset
                col = robot_col + col_offset
                position = (row, col)

                if not environment.is_in_bounds(position):
                    continue

                # circular sensing area
                distance_squared = row_offset ** 2 + col_offset ** 2

                if distance_squared > sensor_range ** 2:
                    continue

                self.grid[row, col] = environment.grid[row, col]

    def is_in_bounds(self, position):
        row, col = position

        return 0 <= row < self.grid.shape[0] and 0 <= col < self.grid.shape[1]

    def is_traversable(self, position):
        row, column = position

        if self.is_in_bounds(position) and self.grid[row, column] != OBSTACLE:
            return True
        
        return False

    def count_known_cells(self):
        # get number of cells that have been observed
        return np.sum(self.grid != UNKNOWN)

    def count_total_cells(self):
        return self.grid.size

    def percent_explored(self):
        known_cells = self.count_known_cells()
        total_cells = self.count_total_cells()

        return 100.0 * known_cells / total_cells
    