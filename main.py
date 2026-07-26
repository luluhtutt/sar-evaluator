from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from config import ROBOT_START, VICTIM_POSITION, GRID_HEIGHT, GRID_WIDTH, ROBOT_SENSOR_RANGE
from environment import Environment
from mapping import OccupancyMap
from planner import astar
from robot import GroundRobot


def visualize(environment, occupancy_map, robot, path, step_number):
    # visualize environment and paths
    plt.clf()

    map_colors = ListedColormap([
        "gray", # unknown
        "white", # free
        "black", # obstacle
        "red" #victim
    ])

    true_world_plot = plt.subplot(1, 2, 1)
    true_world_plot.imshow(
        environment.grid,
        cmap=map_colors,
        vmin=-1,
        vmax=2,
        origin="upper"
    )
    true_world_plot.set_title("Ground Truth")
    true_world_plot.set_xlabel("Column")
    true_world_plot.set_ylabel("Row")

    robot_row, robot_col = robot.position
    true_world_plot.scatter(
        robot_col,
        robot_row,
        marker="o",
        s=100,
        label="Robot"
    )

    victim_row, victim_col = VICTIM_POSITION
    true_world_plot.scatter(
        victim_col,
        victim_row,
        marker="x",
        s=120,
        label="Victim"
    )

    true_world_plot.legend()

    # occupancy map
    map_plot = plt.subplot(1, 2, 2)
    map_plot.imshow(
        occupancy_map.grid,
        cmap=map_colors,
        vmin=-1,
        vmax=2,
        origin="upper"
    )

    map_plot.set_title(f"Robot Occupancy Map\nExplored: {occupancy_map.percent_explored():.1f}%")
    map_plot.set_xlabel("Column")
    map_plot.set_ylabel("Row")
    if path is not None and len(path) > 0:
        path_array = np.array(path)

        map_plot.plot(
            path_array[:, 1],
            path_array[:, 0],
            linewidth=2,
            label="Current A* path"
        )
    map_plot.scatter(
        robot_col,
        robot_row,
        marker="o",
        s=100,
        label="Robot"
    )

    map_plot.legend()

    plt.suptitle(f"Step: {step_number} | Distance: {robot.distance_traveled}"
    )

    plt.tight_layout()
    plt.pause(0.2)


def main():
    environment = Environment()
    occupancy_map = OccupancyMap(GRID_HEIGHT, GRID_WIDTH)
    robot = GroundRobot(ROBOT_START)

    plt.figure(figsize=(14, 7))
    step_number = 0
    max_steps = 500

    while robot.position != VICTIM_POSITION:

        occupancy_map.update_from_sensor(
            environment,
            robot.position,
            ROBOT_SENSOR_RANGE
        )
        path = astar(start=robot.position, goal=VICTIM_POSITION, is_traversable=occupancy_map.is_traversable)
        if path is None:
            print("No path found")
            break

        robot.set_path(path)

        visualize(environment, occupancy_map, robot, path, step_number)

        moved = robot.move_one_step()

        if not moved:
            print("Robot could not move")
            break

        step_number += 1

        if step_number >= max_steps:
            print("Maximum number of steps reached.")
            break
    occupancy_map.update_from_sensor(environment, robot.position, ROBOT_SENSOR_RANGE)

    final_path = [robot.position]

    visualize(environment, occupancy_map, robot, final_path, step_number)
    step_number = 0

    
    if robot.position == VICTIM_POSITION:
        print("Victim reached!")
    else:
        print("Victim was not reached")
    print("Total distance traveled:", robot.distance_traveled)
    print("Total simulation steps:", step_number)
    print("Map explored:",f"{occupancy_map.percent_explored():.1f}%")

    plt.savefig("outputs/exploration_occupancy_map.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()