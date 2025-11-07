from __future__ import annotations
from typing import List, Tuple, Dict, Iterable
import math

Grid = List[List[int]]  # 0 for open cell, 1 for wall
Cell = Tuple[int, int]
Path = List[Cell]
INF = float("inf")


def neighbors(grid: Grid, r: int, c: int, diagonals: bool = False) -> Iterable[Cell]:

    # yeilds neighboring cells that are available to move to

    rows, cols = len(grid), len(grid[0])
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    current_pos = (r, c)

    if diagonals:

        steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for x, y in steps:

        changed_pos = (current_pos[0] + x, current_pos[1] + y)

        if 0 <= changed_pos[0] < rows and 0 <= changed_pos[1] < cols:

            if grid[changed_pos[0]][changed_pos[1]] == 0:

                yield changed_pos


def reconstruct_path(came_from: Dict[Cell, Cell], start: Cell, goal: Cell) -> Path:

    current = goal
    path = []

    while current != start:

        path.append(current)

        current = came_from.get(current)

        if current is None:

            return []

    path.append(start)
    path.reverse()

    return path


def manhattan(a: Cell, b: Cell) -> float:

    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def grid_open_count(grid: Grid) -> int:

    return sum(row.count(0) for row in grid)


def estimate_branching_and_depth(
    came_from: Dict[Cell, Cell], start: Cell, goal: Cell
) -> Tuple[float, float]:

    # Estimate average branching factor and depth of the search tree

    depth = 0
    current = goal

    while current != start:

        current = came_from.get(current)

        if current is None:

            break

        depth += 1

    if depth == 0:

        return 0.0, 0.0

    total_nodes = len(came_from) + 1  # +1 for the start node
    branching_factor = total_nodes ** (1 / depth)

    return branching_factor, float(depth)


def build_result(
    algorithm: str,
    grid: Grid,
    start: Cell,
    goal: Cell,
    path: Path,
    visited_orders: List[Cell],
    came_from: Dict[Cell, Cell],
    time_ms: float,
    diagonals: bool = False,
    heuristic: str = "",
    frontier_peak=None,
    true_costs=None,
    if_grid=True,
) -> Dict:

    success = bool(path and goal in path)

    def _path_cost(p: Path) -> float:

        if not p or len(p) < 2:

            return 0.0

        total = 0.0
        for a, b in zip(p, p[1:]):
            dx = abs(a[0] - b[0])
            dy = abs(a[1] - b[1])
            if dx + dy == 1:
                total += 1.0
            elif dx == 1 and dy == 1:
                total += math.sqrt(2)
            else:
                total += math.hypot(dx, dy)
        return total

    cost = _path_cost(path) if success else INF

    visited_count = len(visited_orders) if visited_orders is not None else 0
    unique_visited = len(set(visited_orders)) if visited_orders is not None else 0
    revisits = visited_count - unique_visited

    total_open = grid_open_count(grid)
    exploration_ratio = unique_visited / total_open if total_open > 0 else 0.0

    manh = manhattan(start, goal)
    optimality_gap = (cost - manh) if (math.isfinite(cost) and success) else None
    path_straightness = (
        (manh / cost) if (math.isfinite(cost) and cost > 0 and success) else None
    )

    branching_factor_est, search_max_depth_est = estimate_branching_and_depth(
        came_from, start, goal
    )

    result = {
        "identity": {"algorithm": algorithm},
        "outcome": {
            "success": success,
            "path_length": len(path),
            "path": path,
            "cost": cost,
        },
        "effort": {
            "time_ms": time_ms,
            "visited_count": visited_count,
            "unique_visited": unique_visited,
            "revisits": revisits,
            "frontier_peak_size": frontier_peak,
        },
        "coverage": {
            "exploration_ratio": exploration_ratio,
            "total_open_cells": total_open,
        },
        "path_quality": {
            "manhattan": manh,
            "optimality_gap": optimality_gap,
            "path_straightness": path_straightness,
        },
        "search_structure": {
            "branching_factor_est": branching_factor_est,
            "search_max_depth_est": search_max_depth_est,
        },
        "config": {"diagonals": diagonals, "heuristic": heuristic},
    }

    return result
