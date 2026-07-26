"""Run frontier exploration until the victim is detected and reached."""

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
from exploration import find_frontiers, choose_frontier
from mapping import OccupancyMap
from planner import astar
from robot import GroundRobot


def draw_simulation(
    environment,
    occupancy_map,
    robot,
    path,
    frontiers,
    selected_goal,
    step_number,
    victim_found
):
    """Draw the true environment and robot occupancy map."""

    plt.clf()

    map_colors = ListedColormap([
        "gray",
        "white",
        "black",
        "red"
    ])

    robot_row, robot_col = robot.position

    # Ground-truth world
    true_world_plot = plt.subplot(1, 2, 1)

    true_world_plot.imshow(
        environment.grid,
        cmap=map_colors,
        vmin=-1,
        vmax=2,
        origin="upper"
    )

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

    true_world_plot.set_title("Ground Truth")
    true_world_plot.set_xlabel("Column")
    true_world_plot.set_ylabel("Row")
    true_world_plot.legend()

    # Robot occupancy map
    occupancy_plot = plt.subplot(1, 2, 2)

    occupancy_plot.imshow(
        occupancy_map.grid,
        cmap=map_colors,
        vmin=-1,
        vmax=2,
        origin="upper"
    )

    # Draw all frontier cells.
    if len(frontiers) > 0:

        frontier_array = np.array(frontiers)

        occupancy_plot.scatter(
            frontier_array[:, 1],
            frontier_array[:, 0],
            marker=".",
            s=25,
            label="Frontiers"
        )

    # Draw selected exploration goal.
    if selected_goal is not None:

        goal_row, goal_col = selected_goal

        occupancy_plot.scatter(
            goal_col,
            goal_row,
            marker="*",
            s=160,
            label="Selected goal"
        )

    # Draw current path.
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

    occupancy_plot.set_title(
        f"Robot Map: {status}\n"
        f"Explored: {occupancy_map.percent_explored():.1f}%"
    )

    occupancy_plot.set_xlabel("Column")
    occupancy_plot.set_ylabel("Row")
    occupancy_plot.legend(loc="upper right")

    plt.suptitle(
        f"Step: {step_number} | "
        f"Distance: {robot.distance_traveled}"
    )

    plt.tight_layout()
    plt.pause(0.15)


def main():

    environment = Environment()

    occupancy_map = OccupancyMap(
        GRID_HEIGHT,
        GRID_WIDTH
    )

    robot = GroundRobot(ROBOT_START)

    plt.figure(figsize=(14, 7))

    step_number = 0
    max_steps = 1000
    victim_found = False

    while step_number < max_steps:

        # Reveal nearby cells.
        occupancy_map.update_from_sensor(
            environment,
            robot.position,
            ROBOT_SENSOR_RANGE
        )

        # Check whether the sensor has detected the victim.
        known_victim_position = occupancy_map.find_known_victim()

        if known_victim_position is not None:

            victim_found = True

            # Once detected, plan directly to the victim.
            path = astar(
                robot.position,
                known_victim_position,
                occupancy_map.is_known_traversable
            )

            frontiers = []
            selected_goal = known_victim_position

        else:

            victim_found = False

            # Find the known/unknown boundaries.
            frontiers = find_frontiers(occupancy_map)

            selected_goal, path = choose_frontier(
                occupancy_map,
                robot.position,
                frontiers
            )

        draw_simulation(
            environment,
            occupancy_map,
            robot,
            path,
            frontiers,
            selected_goal,
            step_number,
            victim_found
        )

        # Stop once the robot reaches the detected victim.
        if (
            known_victim_position is not None
            and robot.position == known_victim_position
        ):
            print("Victim reached!")
            break

        if path is None:
            print("No reachable exploration target remains.")
            break

        robot.set_path(path)

        moved = robot.move_one_step()

        if not moved:
            print("Robot could not move.")
            break

        step_number += 1

    if step_number >= max_steps:
        print("Maximum number of steps reached.")

    occupancy_map.update_from_sensor(
        environment,
        robot.position,
        ROBOT_SENSOR_RANGE
    )

    print("Distance traveled:", robot.distance_traveled)
    print("Simulation steps:", step_number)

    print(
        "Map explored:",
        f"{occupancy_map.percent_explored():.1f}%"
    )

    plt.savefig(
        "outputs/phase4_frontier_exploration.png",
        dpi=200
    )

    plt.show()


if __name__ == "__main__":
    main()