import heapq
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
        self.heuristic = "manhattan"

    def import_from_file(self, file_path: Path) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)
            self.grid = data["maze"]
            self.start = tuple(data["start"])
            self.goal = tuple(data["goal"])

    def solve(self) -> None:
        # g-score: best known cost from start to node
        self.g = {self.start: 0.0}

        # frontier: (f_score, g_score, node)
        f_start = helpers.manhattan(self.start, self.goal)
        heap = [(f_start, 0.0, self.start)]

        # parent links
        self.came_from = {self.start: self.start}

        # metrics
        self.visited_orders = []
        self.frontier_peak = 1

        t0 = time.perf_counter()

        while heap:
            if len(heap) > self.frontier_peak:
                self.frontier_peak = len(heap)

            f, g_current, current = heapq.heappop(heap)

            if g_current != self.g.get(current, float("inf")):
                continue

            # record visitation
            self.visited_orders.append(current)

            if current == self.goal:
                break

            r, c = current
            for neighbor in helpers.neighbors(self.grid, r, c, self.diagonals):
                step_cost = 1.0  # adjust if you handle diagonals with different cost
                tentative_g = g_current + step_cost

                if tentative_g < self.g.get(neighbor, float("inf")):
                    self.g[neighbor] = tentative_g
                    self.came_from[neighbor] = current

                    h = helpers.manhattan(neighbor, self.goal)
                    f_score = tentative_g + h

                    heapq.heappush(heap, (f_score, tentative_g, neighbor))

        self.time_ms = (time.perf_counter() - t0) * 1000.0
        self.path = helpers.reconstruct_path(self.came_from, self.start, self.goal)

    def get_result(self) -> dict:
        true_cost = self.g.get(self.goal)
        return helpers.build_result(
            algorithm="astar",
            grid=self.grid,
            start=self.start,
            goal=self.goal,
            path=self.path,
            visited_orders=self.visited_orders,
            came_from=self.came_from,
            time_ms=self.time_ms,
            diagonals=self.diagonals,
            frontier_peak=self.frontier_peak,
            heuristic=self.heuristic,
            true_costs=true_cost,
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
