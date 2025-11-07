from collections import deque
import time
import json
from pathlib import Path
import sys
import heapq

sys.path.append(str(Path(__file__).resolve().parent.parent))
import helpers


class solver:
    def __init__(self, grid, start, goal, diagonals=False):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.diagonals = diagonals

    def import_from_file(self, file_path: Path) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)
            self.grid = data["maze"]
            self.start = tuple(data["start"])
            self.goal = tuple(data["goal"])

    def solve(self) -> None:

        self.visited_orders = []
        self.came_from = {self.start: self.start}
        self.heap = [(0, self.start)]
        self.frontier_peak = 1
        seen = {self.start}
        self.start_h = helpers.manhattan(self.start, self.goal)
        t0 = time.time()

        while self.heap:

            self.frontier_peak = max(self.frontier_peak, len(self.heap))

            h, current = heapq.heappop(self.heap)

            self.visited_orders.append(current)

            if current == self.goal:

                break

            for neighbor in helpers.neighbors(
                self.grid, current[0], current[1], self.diagonals
            ):

                if neighbor not in seen:

                    seen.add(neighbor)
                    neighbor_h = helpers.manhattan(neighbor, self.goal)
                    heapq.heappush(self.heap, (neighbor_h, neighbor))
                    self.came_from[neighbor] = current

        self.time_ms = (time.time() - t0) * 1000.0
        self.path = helpers.reconstruct_path(self.came_from, self.start, self.goal)

    def get_result(self) -> dict:
        return helpers.build_result(
            algorithm="gbfs",
            grid=self.grid,
            start=self.start,
            goal=self.goal,
            path=self.path,
            visited_orders=self.visited_orders,
            came_from=self.came_from,
            time_ms=self.time_ms,
            diagonals=self.diagonals,
            frontier_peak=self.frontier_peak,
            heuristic=self.start_h,
        )

    def to_file(self, file_path: Path) -> None:
        result = self.get_result()
        with open(file_path, "w") as f:
            json.dump(result, f, indent=4)


if __name__ == "__main__":

    example_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "examples"
        / "example.json"
    )

    solver_instance = solver(grid=[], start=(0, 0), goal=(0, 0))
    solver_instance.import_from_file(example_path)
    solver_instance.solve()

    result = solver_instance.get_result()

    print(json.dumps(result, indent=4))

    solver_instance.to_file(Path("gbfs_result.json"))
