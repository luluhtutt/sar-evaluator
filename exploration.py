from config import FREE, UNKNOWN
from planner import astar, get_neighbors

def is_frontier_cell(occupancy_map, position):
    row, col = position

    if not occupancy_map.is_in_bounds(position) or occupancy_map.grid[row, col] != FREE:
        return False

    neighbors = get_neighbors(position)

    for n in neighbors:
        if not occupancy_map.is_in_bounds(n):
            continue

        neighbor_row, neighbor_col = n

        if occupancy_map.grid[neighbor_row, neighbor_col] == UNKNOWN:
            return True

    return False

def find_frontiers(occupancy_map):
    # get all frontier cells
    frontiers = []

    height, width = occupancy_map.grid.shape

    for r in range(height):
        for c in range(width):

            position = (r, c)

            if is_frontier_cell(occupancy_map, position):
                frontiers.append(position)

    return frontiers

def choose_frontier(occupancy_map, robot_position, frontiers):
    # choose the nearest reachable frontier
    best_frontier = None
    best_path = None
    best_distance = float("inf")

    for f in frontiers:

        path = astar(robot_position, f, occupancy_map.is_known_traversable)

        if path is None:
            continue

        path_distance = len(path) - 1

        if path_distance == 0:
            continue

        if path_distance < best_distance:
            best_distance = path_distance
            best_frontier = f
            best_path = path

    return best_frontier, best_path
