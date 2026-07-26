from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from config import ROBOT_START, VICTIM_POSITION, GRID_HEIGHT, GRID_WIDTH, ROBOT_SENSOR_RANGE
from environment import Environment
from exploration import find_frontiers, choose_frontier
from mapping import OccupancyMap
from planner import astar
from robot import GroundRobot


def visualize(environment, occupancy_map, robot, path, frontiers, selected_goal, step_number, victim_found):
    # visualize environment and paths
    plt.clf()

    map_colors = ListedColormap([
        "gray", # unknown
        "white", # free
        "black", # obstacle
        "red" #victim
    ])

    robot_row, robot_col = robot.position
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
    occupancy_plot = plt.subplot(1, 2, 2)
    occupancy_plot.imshow(
        occupancy_map.grid,
        cmap=map_colors,
        vmin=-1,
        vmax=2,
        origin="upper"
    )

    # draw frontier cells
    if len(frontiers) > 0:

        frontier_array = np.array(frontiers)

        occupancy_plot.scatter(
            frontier_array[:, 1],
            frontier_array[:, 0],
            marker=".",
            s=25,
            label="Frontiers"
        )

    # draw selected exploration goal
    if selected_goal is not None:

        goal_row, goal_col = selected_goal

        occupancy_plot.scatter(
            goal_col,
            goal_row,
            marker="*",
            s=160,
            label="Selected goal"
        )
    if path is not None and len(path) > 0:
        path_array = np.array(path)

        occupancy_plot.plot(
            path_array[:, 1],
            path_array[:, 0],
            linewidth=2,
            label="Current path"
        )
    occupancy_plot.scatter(
        robot_col,
        robot_row,
        marker="o",
        s=100,
        label="Robot"
    )

    if victim_found:
        status = "Victim detected"
    else:
        status = "Exploring"

    occupancy_plot.set_title(f"Robot Occupancy Map\nExplored: {occupancy_map.percent_explored():.1f}%")
    occupancy_plot.set_xlabel("Column")
    occupancy_plot.set_ylabel("Row")
    occupancy_plot.legend()

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
    max_steps = 1000
    victim_found = False

    while step_number < max_steps:

        occupancy_map.update_from_sensor(
            environment,
            robot.position,
            ROBOT_SENSOR_RANGE
        )

        known_victim_position = occupancy_map.find_known_victim()

        if known_victim_position is not None:
            victim_found = True

            path = astar(start=robot.position, goal=known_victim_position, is_traversable=occupancy_map.is_traversable)

            frontiers = []
            selected_goal = known_victim_position
        else:
            victim_found = False

            frontiers = find_frontiers(occupancy_map)

            selected_goal, path = choose_frontier(occupancy_map, robot.position, frontiers)

        visualize(environment, occupancy_map, robot, path, frontiers, selected_goal, step_number, victim_found)
        
        if known_victim_position is not None and robot.position == known_victim_position:
            print("Victim reached!")
            break

        if path is None:
            print("No path found")
            break

        robot.set_path(path)

        moved = robot.move_one_step()

        if not moved:
            print("Robot could not move")
            break

        step_number += 1

        if step_number >= max_steps:
            print("Maximum number of steps reached.")

    occupancy_map.update_from_sensor(environment, robot.position, ROBOT_SENSOR_RANGE)

    print("Total distance traveled:", robot.distance_traveled)
    print("Total simulation steps:", step_number)
    print("Map explored:",f"{occupancy_map.percent_explored():.1f}%")

    plt.savefig("outputs/exploration_occupancy_map.png", dpi=200)

    plt.show()


if __name__ == "__main__":
    main()