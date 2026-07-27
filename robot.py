import math

from config import GROUND_MOVE_ENERGY


class GroundRobot:

    def __init__(self, start_position):
        self.position = start_position

        self.path = []
        self.path_index = 0

        self.distance_traveled = 0.0
        self.energy_used = 0.0

    def set_path(self, path):
        # path for robot to follow

        if path is None or len(path) == 0:
            self.path = []
            self.path_index = 0
            return

        self.path = path
        self.path_index = 1

    def clear_path(self):
        self.path = []
        self.path_index = 0

    def has_path(self):
        return self.path_index < len(self.path)

    def move_one_step(self):
        if self.path_index >= len(self.path):
            return False

        old_row, old_col = self.position
        new_row, new_col = self.path[self.path_index]

        self.position = (new_row, new_col)
        self.path_index += 1

        step_distance = math.hypot(new_row - old_row,new_col - old_col)

        self.distance_traveled += step_distance
        self.energy_used += GROUND_MOVE_ENERGY * step_distance

        return True

    def rescue_victim(self, environment, occupancy_map):
        victim = environment.get_victim_at_position(self.position)

        if victim is None:
            return False

        if victim.reached:
            return False

        occupancy_map.mark_victim_reached(environment,self.position)

        return True
    