from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import ROBOT_START, VICTIM_POSITION
from environment import Environment
from planner import astar
from robot import GroundRobot


def visualize(environment, robot, path, step_number):
    # visualize environment and paths
    plt.clf()

    plt.imshow(environment.grid, origin="upper")

    # planned path
    path_array = np.array(path)
    plt.plot(
        path_array[:, 1],
        path_array[:, 0],
        linewidth=2,
        label="A* path",
    )

    robot_row, robot_col = robot.position
    plt.scatter(
        robot_col,
        robot_row,
        marker="o",
        s=100,
        label="Ground robot",
    )

    victim_row, victim_col = VICTIM_POSITION
    plt.scatter(
        victim_col,
        victim_row,
        marker="x",
        s=120,
        label="Victim",
    )

    plt.title(f"Ground Robot Navigation\nStep: {step_number}, Distance: {robot.distance_traveled}")
    plt.xlabel("Column")
    plt.ylabel("Row")
    plt.legend()
    plt.grid(True)

    plt.pause(0.2)


def main():
    environment = Environment()
    robot = GroundRobot(ROBOT_START)

    path = astar(start=ROBOT_START, goal=VICTIM_POSITION, is_traversable=environment.is_traversable)

    if path is None:
        print("No path found")
        return

    robot.set_path(path)

    print("Starting simulation")
    print(f"Path found with {len(path) - 1} moves")

    plt.figure(figsize=(10, 7))
    step_number = 0

    visualize(environment, robot, path, step_number)

    while not robot.reached_goal:
        robot.move_one_step()
        step_number += 1

        visualize(environment, robot, path, step_number)
    
    print("Victim reached")
    print("Total distance traveled:", robot.distance_traveled)
    print("Total simulation steps:", step_number)

    plt.savefig("outputs/phase2_robot_navigation.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()