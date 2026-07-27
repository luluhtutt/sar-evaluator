from __future__ import annotations

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from config import (
    MAP_FILE,
    ROBOT_SENSOR_RANGE,
    DRONE_SENSOR_RANGE,
    DRONE_DEPLOY_STEP,
    DRONE_REDEPLOY_COOLDOWN,
    MAX_DRONE_DEPLOYMENTS,
    MAX_SIMULATION_STEPS,
    VISUALIZATION_DELAY,
    EXPERIMENT_MODE,
    GROUND_ONLY,
    CONSTANT_DRONE,
    LIMITED_DRONE,
    DRONE_CRUISE_ALTITUDE
)

from environment import Environment
from exploration import find_frontiers, choose_frontier
from mapping import OccupancyMap
from planner import astar
from robot import GroundRobot
from drone import Drone


def visualize(environment, occupancy_map, robot, drone, path, frontiers, selected_goal, step_number, victim_found):
    current_azimuth = None
    current_elevation = None

    if len(plt.gcf().axes) > 0:
        old_axis = plt.gcf().axes[0]

        if hasattr(old_axis, "azim"):
            current_azimuth = old_axis.azim
            current_elevation = old_axis.elev

    plt.clf()

    robot_row, robot_col = robot.position
    rows, cols = environment.grid.shape

    # ground truth
    true_world_plot = plt.subplot(1, 2, 1, projection="3d")

    floor_x, floor_y = np.meshgrid(np.arange(cols), np.arange(rows))
    floor_z = np.zeros_like(floor_x, dtype=float)

    true_world_plot.plot_surface(floor_x, floor_y, floor_z, alpha=0.15)

    obstacle_rows, obstacle_cols = np.where(environment.height_map > 0)

    if len(obstacle_rows) > 0:
        obstacle_heights = environment.height_map[obstacle_rows, obstacle_cols]

        true_world_plot.bar3d(
            obstacle_cols - 0.4,
            obstacle_rows - 0.4,
            np.zeros(len(obstacle_rows)),
            0.8,
            0.8,
            obstacle_heights,
            alpha=0.35,
            label="Obstacles"
        )

    true_world_plot.scatter(
        robot_col,
        robot_row,
        0.4,
        marker="o",
        s=140,
        depthshade=False,
        label="Robot"
    )

    true_world_plot.plot(
        [robot_col, robot_col],
        [robot_row, robot_row],
        [0, 1.5],
        linestyle="--",
        linewidth=2
    )

    victim_row, victim_col = environment.victim_position

    true_world_plot.scatter(
        victim_col,
        victim_row,
        0.4,
        marker="x",
        s=140,
        depthshade=False,
        label="Victim"
    )

    if drone.active and drone.position is not None:
        drone_row, drone_col, drone_altitude = drone.position

        true_world_plot.scatter(
            drone_col,
            drone_row,
            drone_altitude,
            marker="^",
            s=140,
            depthshade=False,
            label=f"Drone z={drone_altitude:.1f}"
        )

        heading_row, heading_col = drone.heading

        true_world_plot.quiver(
            drone_col,
            drone_row,
            drone_altitude,
            heading_col,
            heading_row,
            0,
            length=1.5,
            normalize=True
        )

        if len(drone.forward_obstacles) > 0:
            obstacle_array = np.array(drone.forward_obstacles)

            detection_heights = environment.height_map[
                obstacle_array[:, 0],
                obstacle_array[:, 1]
            ]

            true_world_plot.scatter(
                obstacle_array[:, 1],
                obstacle_array[:, 0],
                detection_heights + 0.2,
                marker="s",
                s=90,
                depthshade=False,
                label="Forward detections"
            )

    if path is not None and len(path) > 0:
        path_array = np.array(path)

        true_world_plot.plot(
            path_array[:, 1],
            path_array[:, 0],
            np.full(len(path_array), 0.1),
            linewidth=2,
            label="Ground path"
        )

    maximum_height = max(
        DRONE_CRUISE_ALTITUDE + 2,
        float(np.max(environment.height_map)) + 2
    )

    true_world_plot.set_xlim(-0.5, cols - 0.5)
    true_world_plot.set_ylim(rows - 0.5, -0.5)
    true_world_plot.set_zlim(0, maximum_height)

    true_world_plot.set_xlabel("Column")
    true_world_plot.set_ylabel("Row")
    true_world_plot.set_zlabel("Height")
    true_world_plot.set_title("3D Ground Truth")

    if current_azimuth is not None and current_elevation is not None:
        true_world_plot.view_init(elev=current_elevation, azim=current_azimuth)
    else:
        true_world_plot.view_init(elev=35, azim=-60)

    true_world_plot.legend()

    # 2D occupancy map
    occupancy_plot = plt.subplot(1, 2, 2)

    map_colors = ListedColormap([
        "gray",
        "white",
        "black",
        "red"
    ])

    occupancy_plot.imshow(
        occupancy_map.grid,
        cmap=map_colors,
        vmin=-1,
        vmax=2,
        origin="upper"
    )

    if len(frontiers) > 0:
        frontier_array = np.array(frontiers)

        occupancy_plot.scatter(
            frontier_array[:, 1],
            frontier_array[:, 0],
            marker=".",
            s=25,
            label="Frontiers"
        )

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

    if drone.active and drone.position is not None:
        drone_row, drone_col, drone_altitude = drone.position

        occupancy_plot.scatter(
            drone_col,
            drone_row,
            marker="^",
            s=120,
            label=f"Drone z={drone_altitude:.1f}"
        )

    if victim_found:
        status = "Victim detected"
    elif drone.active:
        status = "Aerial scout active"
    else:
        status = "Ground robot exploring"

    occupancy_plot.set_title(
        f"Robot Occupancy Map\nExplored: {occupancy_map.percent_explored():.1f}%"
    )

    occupancy_plot.set_xlabel("Column")
    occupancy_plot.set_ylabel("Row")
    occupancy_plot.legend()

    plt.suptitle(
        f"Mode: {EXPERIMENT_MODE} | Step: {step_number} | Status: {status}\n"
        f"Ground distance: {robot.distance_traveled:.1f} | "
        f"Drone distance: {drone.distance_traveled:.1f} | "
        f"Deployments: {drone.deployments_used}"
    )

    plt.tight_layout()
    plt.pause(VISUALIZATION_DELAY)


def deployment_is_allowed(drone):
    if EXPERIMENT_MODE == GROUND_ONLY:
        return False

    if EXPERIMENT_MODE == CONSTANT_DRONE:
        return True

    if EXPERIMENT_MODE == LIMITED_DRONE:
        return drone.deployments_used < MAX_DRONE_DEPLOYMENTS

    raise ValueError(f"Unknown experiment mode: {EXPERIMENT_MODE}")


def choose_drone_frontier(occupancy_map, robot_position, frontiers, drone):
    available_frontiers = [
        frontier
        for frontier in frontiers
        if frontier not in drone.visited_targets
    ]

    if len(available_frontiers) == 0:
        return None

    selected_goal, _ = choose_frontier(
        occupancy_map,
        robot_position,
        available_frontiers
    )

    return selected_goal


def print_results(occupancy_map, robot, drone, step_number, victim_reached):
    total_energy = robot.energy_used + drone.energy_used

    print("\n------------------------------")
    print("Mission Results")
    print("------------------------------")
    print("Experiment mode:", EXPERIMENT_MODE)
    print("Victim reached:", victim_reached)
    print("Total simulation steps:", step_number)
    print("Ground distance traveled:", f"{robot.distance_traveled:.2f}")
    print("Drone distance traveled:", f"{drone.distance_traveled:.2f}")
    print("Drone deployments:", drone.deployments_used)
    print("Ground energy:", f"{robot.energy_used:.2f}")
    print("Drone energy:", f"{drone.energy_used:.2f}")
    print("Total normalized energy:", f"{total_energy:.2f}")
    print("Map explored:", f"{occupancy_map.percent_explored():.1f}%")
    print("------------------------------")


def main():
    environment = Environment(MAP_FILE)

    grid_height, grid_width = environment.grid.shape
    occupancy_map = OccupancyMap(grid_height, grid_width)

    robot = GroundRobot(environment.robot_start)
    drone = Drone()

    plt.ion()
    plt.figure(figsize=(14, 7))

    step_number = 0
    victim_found = False
    victim_reached = False
    drone_cooldown = 0

    path = None
    frontiers = []
    selected_goal = None

    while step_number < MAX_SIMULATION_STEPS:

        if drone.active:
            drone.move_one_step(environment)

            if drone.state in ("flying", "scanning") and drone.position is not None:
                drone_row, drone_col, drone_altitude = drone.position

                occupancy_map.update_from_sensor(
                    environment,
                    (int(round(drone_row)), int(round(drone_col))),
                    DRONE_SENSOR_RANGE
                )

            visualize(
                environment,
                occupancy_map,
                robot,
                drone,
                None,
                [],
                drone.target,
                step_number,
                victim_found
            )

            if drone.state == "finished":
                drone.finish_mission()
                drone_cooldown = DRONE_REDEPLOY_COOLDOWN

            step_number += 1
            continue

        if drone_cooldown > 0:
            drone_cooldown -= 1

        occupancy_map.update_from_sensor(
            environment,
            robot.position,
            ROBOT_SENSOR_RANGE
        )

        known_victim_position = occupancy_map.find_known_victim()

        if known_victim_position is not None:
            victim_found = True

            path = astar(
                start=robot.position,
                goal=known_victim_position,
                is_traversable=occupancy_map.is_traversable
            )

            frontiers = []
            selected_goal = known_victim_position

        else:
            victim_found = False
            frontiers = find_frontiers(occupancy_map)

            selected_goal, path = choose_frontier(
                occupancy_map,
                robot.position,
                frontiers
            )

        should_deploy = (
            step_number >= DRONE_DEPLOY_STEP
            and deployment_is_allowed(drone)
            and drone_cooldown == 0
            and known_victim_position is None
            and len(frontiers) > 0
        )

        if should_deploy:
            drone_target = choose_drone_frontier(
                occupancy_map,
                robot.position,
                frontiers,
                drone
            )

            if drone_target is not None:
                deployed = drone.deploy(robot.position, drone_target)

                if deployed:
                    continue

        visualize(
            environment,
            occupancy_map,
            robot,
            drone,
            path,
            frontiers,
            selected_goal,
            step_number,
            victim_found
        )

        if known_victim_position is not None and robot.position == known_victim_position:
            print("Victim reached.")
            victim_reached = True
            break

        if path is None or len(path) == 0:
            print("No reachable path or frontier was found.")
            break

        robot.set_path(path)

        moved = robot.move_one_step()

        if not moved:
            print("Robot could not move.")
            break

        step_number += 1

    if step_number >= MAX_SIMULATION_STEPS:
        print("Maximum number of simulation steps reached.")

    occupancy_map.update_from_sensor(
        environment,
        robot.position,
        ROBOT_SENSOR_RANGE
    )

    print_results(
        occupancy_map,
        robot,
        drone,
        step_number,
        victim_reached
    )

    os.makedirs("outputs", exist_ok=True)

    map_name = os.path.splitext(os.path.basename(MAP_FILE))[0]
    output_file = f"outputs/{map_name}_{EXPERIMENT_MODE}.png"

    plt.savefig(output_file, dpi=200)

    print("Saved final visualization to:", output_file)

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
