import numpy as np
import math

from config import DRONE_CRUISE_ALTITUDE, DRONE_VERTICAL_SPEED, DRONE_CLEARANCE, DRONE_SCAN_STEPS, FORWARD_CAMERA_RANGE, FORWARD_CAMERA_HALF_ANGLE_DEGREES

def detect_forward_obstacles(environment, drone_position, heading):

    drone_row, drone_col, drone_altitude = drone_position
    heading_row, heading_col = heading

    heading_length = math.sqrt(heading_row ** 2 + heading_col ** 2)

    if heading_length == 0:
        return []

    normalized_heading_row = heading_row / heading_length
    normalized_heading_col = heading_col / heading_length

    detected_obstacles = []

    minimum_row = max(0, int(math.floor(drone_row - FORWARD_CAMERA_RANGE)))
    maximum_row = min(environment.grid.shape[0], int(math.ceil(drone_row + FORWARD_CAMERA_RANGE + 1)))
    minimum_col = max(0, int(math.floor(drone_col - FORWARD_CAMERA_RANGE)))
    maximum_col = min( environment.grid.shape[1], int(math.ceil(drone_col + FORWARD_CAMERA_RANGE + 1)) )

    for row in range(minimum_row, maximum_row):
        for col in range(minimum_col, maximum_col):

            obstacle_height = environment.height_map[row, col]

            if obstacle_height <= 0:
                continue

            row_diff = row - drone_row
            col_diff = col - drone_col

            horizontal_distance = math.sqrt(row_diff ** 2 + col_diff ** 2)

            if horizontal_distance == 0 or horizontal_distance > FORWARD_CAMERA_RANGE:
                continue

            direction_row = row_diff / horizontal_distance
            direction_col = col_diff / horizontal_distance

            dot_product = (normalized_heading_row * direction_row + normalized_heading_col * direction_col)

            dot_product = max(-1.0, min(1.0, dot_product))

            angle_degrees = math.degrees(math.acos(dot_product))

            if angle_degrees > FORWARD_CAMERA_HALF_ANGLE_DEGREES:
                continue

            unsafe_height = obstacle_height + DRONE_CLEARANCE

            if unsafe_height >= drone_altitude:
                detected_obstacles.append((row, col))

    return detected_obstacles

class Drone:

    def __init__(self):
        self.position = None
        self.target = None

        self.state = "docked"

        self.active = False
        self.has_been_used = False

        self.distance_traveled = 0.0
        self.scan_steps_remaining = 0

        self.heading = (0, 1)
        self.forward_obstacles = []

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

        self.heading = (0, 1)
        self.forward_obstacles = []

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

            row_change = new_row - current_row
            col_change = new_col - current_col

            if row_change != 0 or col_change != 0:
                self.heading = (row_change, col_change)

            self.forward_obstacles = detect_forward_obstacles(environment, self.position, self.heading)

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
