import time
import json
from pathlib import Path
import sys

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
        stack = [self.start]
        self.frontier_peak = 1

        t0 = time.perf_counter()

        while stack:
            if len(stack) > self.frontier_peak:
                self.frontier_peak = len(stack)

            current = stack.pop()
            self.visited_orders.append(current)

            if current == self.goal:
                break

            r, c = current
            for neighbor in helpers.neighbors(self.grid, r, c, self.diagonals):

                if neighbor not in self.came_from:
                    self.came_from[neighbor] = current
                    stack.append(neighbor)

        self.time_ms = (time.perf_counter() - t0) * 1000.0
        self.path = helpers.reconstruct_path(self.came_from, self.start, self.goal)

    def get_result(self) -> dict:
        return helpers.build_result(
            algorithm="dfs",
            grid=self.grid,
            start=self.start,
            goal=self.goal,
            path=self.path,
            visited_orders=self.visited_orders,
            came_from=self.came_from,
            time_ms=self.time_ms,
            diagonals=self.diagonals,
            frontier_peak=self.frontier_peak,
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
