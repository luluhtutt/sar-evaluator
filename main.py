from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from config import ROBOT_START, VICTIM_POSITION
from environment import Environment
from planner import astar


def visualize(environment: Environment, path: list[tuple[int, int]]):
    # visualize environment and paths
    figure, axis = plt.subplots(figsize=(10, 7))

    axis.imshow(environment.grid, origin="upper")

    path_array = np.asarray(path)

    axis.plot(
        path_array[:, 1],
        path_array[:, 0],
        linewidth=2,
        label="A* path",
    )

    axis.scatter(
        ROBOT_START[1],
        ROBOT_START[0],
        marker="o",
        s=100,
        label="Ground robot",
    )

    axis.scatter(
        VICTIM_POSITION[1],
        VICTIM_POSITION[0],
        marker="x",
        s=120,
        label="Victim",
    )

    axis.set_title("Ground Robot A*")
    axis.set_xlabel("Column")
    axis.set_ylabel("Row")
    axis.legend()
    axis.grid(True)

    figure.tight_layout()
    figure.savefig("outputs/phase1_astar.png", dpi=200)
    plt.show()


def main():
    environment = Environment()

    path = astar(start=ROBOT_START, goal=VICTIM_POSITION, is_traversable=environment.is_traversable)

    if path is None:
        raise RuntimeError("A* could not find a path")

    print(f"Path found with {len(path) - 1} moves.")
    visualize(environment, path)


if __name__ == "__main__":
    main()