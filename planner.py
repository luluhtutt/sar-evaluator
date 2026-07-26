from __future__ import annotations

import heapq
from math import inf
from typing import Callable


Position = tuple[int, int]


def manhattan_distance(a: Position, b: Position):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def get_neighbors(position: Position):
    row, column = position

    return [(row - 1, column), (row + 1, column), (row, column - 1), (row, column + 1)]


def reconstruct_path(came_from: dict[Position, Position], goal: Position):
    # reconstruct path from start to destination
    path = [goal]
    current = goal

    while current in came_from:
        current = came_from[current]
        path.append(current)

    path.reverse()
    return path


def astar(start: Position, goal: Position, is_traversable: Callable[[Position], bool]):
    # a star algo, returns path from start --> destination
    frontier: list[tuple[float, Position]] = []
    heapq.heappush(frontier, (0.0, start))

    came_from: dict[Position, Position] = {}
    cost_so_far: dict[Position, float] = {start: 0.0}

    while frontier:
        _, current = heapq.heappop(frontier)

        if current == goal:
            return reconstruct_path(came_from, goal)

        for neighbor in get_neighbors(current):
            if not is_traversable(neighbor):
                continue

            new_cost = cost_so_far[current] + 1.0

            if new_cost < cost_so_far.get(neighbor, inf):
                cost_so_far[neighbor] = new_cost
                came_from[neighbor] = current

                priority = new_cost + manhattan_distance(neighbor, goal)
                heapq.heappush(frontier, (priority, neighbor))

    return None
