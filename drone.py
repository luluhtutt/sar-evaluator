class Drone:

    def __init__(self):
        self.position = None
        self.target = None

        self.active = False
        self.has_been_used = False
        self.reached_target = False

        self.distance_traveled = 0

    def deploy(self, robot_position, target):
        if self.active or self.has_been_used:
            return False

        if target is None:
            return False

        self.position = robot_position
        self.target = target

        self.active = True
        self.reached_target = False

        print("Drone deployed toward: ", target)

        return True

    def move_one_step(self):
        # move one grid step towards target
        # drone can fly over obstacles
        # TODO 3D sim

        if not self.active:
            return False

        if self.position == self.target:
            self.reached_target = True
            return False

        current_row, current_col = self.position
        target_row, target_col = self.target

        new_row = current_row
        new_col = current_col

        if current_row < target_row:
            new_row += 1
        elif current_row > target_row:
            new_row -= 1

        if current_col < target_col:
            new_col += 1
        elif current_col > target_col:
            new_col -= 1

        self.position = (new_row, new_col)
        self.distance_traveled += 1

        if self.position == self.target:
            self.reached_target = True

        return True

    def finish_mission(self):
        print("Drone completed its sensing mission.")

        self.active = False
        self.has_been_used = True
        self.reached_target = False

        self.position = None
        self.target = None
