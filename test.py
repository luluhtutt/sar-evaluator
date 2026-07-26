"""Run the ground robot mapping simulation."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from config import (
    GRID_HEIGHT,
    GRID_WIDTH,
    ROBOT_SENSOR_RANGE,
    ROBOT_START,
    VICTIM_POSITION,
)
from environment import Environment
from mapping import OccupancyMap
from planner import astar
from robot import GroundRobot


def draw_simulation(
    environment,
    occupancy_map,
    robot,
    path,
    step_number
):
    """Draw the true environment and the robot's known map."""

    plt.clf()

    # Colors correspond to:
    # unknown, free, obstacle, victim
    map_colors = ListedColormap([
        "gray",
        "white",
        "black",
        "red"
    ])

    # Draw the true environment on the left.
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

    # Draw the robot's occupancy map on the right.
    map_plot = plt.subplot(1, 2, 2)

    map_plot.imshow(
        occupancy_map.grid,
        cmap=map_colors,
        vmin=-1,
        vmax=2,
        origin="upper"
    )

    map_plot.set_title(
        "Robot Occupancy Map\n"
        f"Explored: {occupancy_map.percent_explored():.1f}%"
    )

    map_plot.set_xlabel("Column")
    map_plot.set_ylabel("Row")

    # Draw the currently planned path.
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

    plt.suptitle(
        f"Step: {step_number} | "
        f"Distance: {robot.distance_traveled}"
    )

    plt.tight_layout()
    plt.pause(0.2)


def main():
    environment = Environment()

    occupancy_map = OccupancyMap(
        GRID_HEIGHT,
        GRID_WIDTH
    )

    robot = GroundRobot(ROBOT_START)

    plt.figure(figsize=(14, 7))

    step_number = 0
    max_steps = 500

    while robot.position != VICTIM_POSITION:

        # The robot observes nearby cells.
        occupancy_map.update_from_sensor(
            environment,
            robot.position,
            ROBOT_SENSOR_RANGE
        )

        # Replan using only the robot's current occupancy map.
        path = astar(
            robot.position,
            VICTIM_POSITION,
            occupancy_map.is_traversable
        )

        if path is None:
            print("No path could be found.")
            break

        robot.set_path(path)

        draw_simulation(
            environment,
            occupancy_map,
            robot,
            path,
            step_number
        )

        moved = robot.move_one_step()

        if not moved:
            print("Robot could not move.")
            break

        step_number += 1

        if step_number >= max_steps:
            print("Maximum number of steps reached.")
            break

    # Perform one final scan at the destination.
    occupancy_map.update_from_sensor(
        environment,
        robot.position,
        ROBOT_SENSOR_RANGE
    )

    final_path = [robot.position]

    draw_simulation(
        environment,
        occupancy_map,
        robot,
        final_path,
        step_number
    )

    if robot.position == VICTIM_POSITION:
        print("Victim reached!")
    else:
        print("Victim was not reached.")

    print("Distance traveled:", robot.distance_traveled)
    print("Simulation steps:", step_number)

    print(
        "Map explored:",
        f"{occupancy_map.percent_explored():.1f}%"
    )

    plt.savefig(
        "outputs/phase3_occupancy_mapping.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()