import heapq
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import helpers


class solver:
    def __init__(self, grid, start, goal, diagonals: bool = False):
        # JPS here assumes 4-directional, uniform-cost movement
        self.grid = grid
        self.start = start
        self.goal = goal
        self.diagonals = False
        self.heuristic = "manhattan"

    def import_from_file(self, file_path: Path) -> None:
        with open(file_path, "r") as f:
            data = json.load(f)
            self.grid = data["maze"]
            self.start = tuple(data["start"])
            self.goal = tuple(data["goal"])

    # ---------- core API ----------

    def solve(self) -> None:
        self.g = {self.start: 0.0}
        self.came_from = {self.start: self.start}
        self.visited_orders = []
        self.frontier_peak = 1

        h0 = helpers.manhattan(self.start, self.goal)
        open_heap = [(h0, 0.0, self.start)]

        t0 = time.perf_counter()

        while open_heap:
            if len(open_heap) > self.frontier_peak:
                self.frontier_peak = len(open_heap)

            f, g_current, current = heapq.heappop(open_heap)

            # stale check
            if g_current != self.g.get(current, float("inf")):
                continue

            self.visited_orders.append(current)

            if current == self.goal:

                break

            for succ in self.get_successors(current):

                step_cost = abs(succ[0] - current[0]) + abs(succ[1] - current[1])
                tentative_g = g_current + step_cost

                if tentative_g < self.g.get(succ, float("inf")):

                    self.g[succ] = tentative_g
                    self.came_from[succ] = current

                    f_succ = tentative_g + helpers.manhattan(succ, self.goal)
                    heapq.heappush(open_heap, (f_succ, tentative_g, succ))

        self.time_ms = (time.perf_counter() - t0) * 1000.0
        self.path = helpers.reconstruct_path(self.came_from, self.start, self.goal)

    def get_result(self) -> dict:

        true_cost = self.g.get(self.goal)

        return helpers.build_result(

            algorithm="jps",
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


    def walkable(self, x: int, y: int) -> bool:
        return (
            0 <= x < len(self.grid)
            and 0 <= y < len(self.grid[0])
            and self.grid[x][y] == 0
        )

    def jump(self, x: int, y: int, dx: int, dy: int):

        while True:
            x += dx
            y += dy

            if not self.walkable(x, y):
                return None

            if (x, y) == self.goal:
                return (x, y)

            if dx != 0:

                if self.walkable(x, y - 1) and not self.walkable(x - dx, y - 1):
                    return (x, y)
                # down forced
                if self.walkable(x, y + 1) and not self.walkable(x - dx, y + 1):
                    return (x, y)

            elif dy != 0:

                if self.walkable(x - 1, y) and not self.walkable(x - 1, y - dy):
                    return (x, y)

                if self.walkable(x + 1, y) and not self.walkable(x + 1, y - dy):
                    return (x, y)

            # otherwise keep going

    def get_successors(self, current):

        successors = []
        parent = self.came_from.get(current)

        if parent is None or parent == current:
            directions = []
            x, y = current
            if self.walkable(x + 1, y):
                directions.append((1, 0))
            if self.walkable(x - 1, y):
                directions.append((-1, 0))
            if self.walkable(x, y + 1):
                directions.append((0, 1))
            if self.walkable(x, y - 1):
                directions.append((0, -1))
        else:
            cx, cy = current
            px, py = parent
            dx = cx - px
            dy = cy - py

            # normalize to -1, 0, 1
            if dx != 0:
                dx = 1 if dx > 0 else -1
            if dy != 0:
                dy = 1 if dy > 0 else -1

            directions = []

            if dx != 0 and dy == 0:
                directions.append((dx, 0))
                if self.walkable(cx, cy - 1) and not self.walkable(cx - dx, cy - 1):
                    directions.append((0, -1))
                if self.walkable(cx, cy + 1) and not self.walkable(cx - dx, cy + 1):
                    directions.append((0, 1))

            elif dy != 0 and dx == 0:
                
                directions.append((0, dy))

                if self.walkable(cx - 1, cy) and not self.walkable(cx - 1, cy - dy):
                    directions.append((-1, 0))

                if self.walkable(cx + 1, cy) and not self.walkable(cx + 1, cy - dy):
                    directions.append((1, 0))

            else:

                directions = []

        x, y = current
        for dx, dy in directions:
            jp = self.jump(x, y, dx, dy)
            if jp is not None:
                successors.append(jp)

        return successors


if __name__ == "__main__":
    example_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "examples"
        / "example.json"
    )

    s = solver(grid=[], start=(0, 0), goal=(0, 0))
    s.import_from_file(example_path)
    s.solve()
    print(json.dumps(s.get_result(), indent=4))
