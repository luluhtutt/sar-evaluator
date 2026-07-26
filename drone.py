import numpy as np
import math

from config import DRONE_CRUISE_ALTITUDE, DRONE_VERTICAL_SPEED, DRONE_CLEARANCE, DRONE_SCAN_STEPS

class Drone:

    def __init__(self):
        self.position = None
        self.target = None

        self.state = "docked"

        self.active = False
        self.has_been_used = False

        self.distance_traveled = 0.0
        self.scan_steps_remaining = 0

    def deploy(self, robot_position, target):
        if self.active or self.has_been_used:
            return False

        if target is None:
            return False

        robot_row, robot_col = robot_position
        self.position = (float(robot_row), float(robot_col), 0.0)

        self.target = target
        self.state = "takeoff"
        self.active = True

        print("Drone deployed toward: ", target)

        return True

    def move_one_step(self, environment):
        # move one grid step towards target
        # drone can fly over obstacles
        # TODO 3D sim

        if not self.active:
            return False

        row, col, altitude = self.position

        if self.state == "takeoff":
            new_altitude = min(altitude + DRONE_VERTICAL_SPEED, DRONE_CRUISE_ALTITUDE)
            self.distance_traveled += new_altitude - altitude
            self.position = (row, col, new_altitude)
            if new_altitude >= DRONE_CRUISE_ALTITUDE:
                self.state = "flying"
            return True

        if self.state == "flying":
            current_row = int(round(row))
            current_col = int(round(col))
            new_row = current_row
            new_col = current_col

            target_row, target_col = self.target

            if current_row < target_row:
                new_row += 1
            elif current_row > target_row:
                new_row -= 1

            if current_col < target_col:
                new_col += 1
            elif current_col > target_col:
                new_col -= 1

            # target reached
            if new_row == current_row and new_col == current_col:
                self.state = "scanning"
                self.scan_steps_remaining = DRONE_SCAN_STEPS
                return True

            obstacle_height = environment.height_map[new_row, new_col]

            required_alt = obstacle_height + DRONE_CLEARANCE
            desired_alt = max(DRONE_CRUISE_ALTITUDE, required_alt)

            # move up if necessary
            if altitude < desired_alt:
                new_altitude = min(altitude + DRONE_VERTICAL_SPEED, desired_alt)

                self.distance_traveled += (new_altitude - altitude)

                self.position = (row, col, new_altitude)

                return True
                
            horizontal_distance = math.sqrt((new_row - row) ** 2 + (new_col - col) ** 2)
            self.position = (float(new_row), float(new_col), altitude)
            self.distance_traveled += horizontal_distance

            if new_row == target_row and new_col == target_col:
                self.state = "scanning"
                self.scan_steps_remaining = DRONE_SCAN_STEPS

            return True

        if self.state == "scanning":
            self.scan_steps_remaining -= 1
            
            if self.scan_steps_remaining <= 0:
                self.state = "finished"

            return True

        return False

    def finish_mission(self):
        print("Drone completed its sensing mission.")

        self.active = False
        self.has_been_used = True
        self.state = "finished"
