
class GroundRobot:

    def __init__(self, start_position):
        self.position = start_position
        self.path = []
        self.path_index = 0
        self.distance_traveled = 0
        self.reached_goal = False

    def set_path(self, path):
        # path for robot to follow

        if path is None or len(path) == 0:
            self.path = []
            self.path_index = 0
            return

        self.path = path

        self.path_index = 1

    def move_one_step(self):

        if self.path_index >= len(self.path):
            return False

        next_position = self.path[self.path_index]

        self.position = next_position
        self.path_index += 1
        self.distance_traveled += 1

        return True
    