from collections import deque
import time
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
import helpers


class solver:
    def __init__(self, grid, start, goal, diagonals: bool = False):
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
        # forward search state
        self.came_from_front = {self.start: self.start}
        frontier_front = deque([self.start])
        self.visited_orders_f = []

        # backward search state
        self.came_from_back = {self.goal: self.goal}
        frontier_back = deque([self.goal])
        self.visited_orders_b = []

        self.meeting_point = None
        self.frontier_peak = 2

        t0 = time.perf_counter()

        # stop when one side is empty or we meet
        while frontier_front and frontier_back and self.meeting_point is None:
            # track combined frontier size
            self.frontier_peak = max(
                self.frontier_peak,
                len(frontier_front) + len(frontier_back),
            )

            # ---------- forward expansion ----------
            if frontier_front and self.meeting_point is None:
                current_f = frontier_front.popleft()
                self.visited_orders_f.append(current_f)

                r, c = current_f
                for nb in helpers.neighbors(self.grid, r, c, self.diagonals):
                    # meet: backward side has seen this node
                    if nb in self.came_from_back:
                        self.meeting_point = nb
                        if nb not in self.came_from_front:
                            self.came_from_front[nb] = current_f
                        break

                    if nb not in self.came_from_front:
                        self.came_from_front[nb] = current_f
                        frontier_front.append(nb)

            # ---------- backward expansion ----------
            if frontier_back and self.meeting_point is None:
                current_b = frontier_back.popleft()
                self.visited_orders_b.append(current_b)

                r, c = current_b
                for nb in helpers.neighbors(self.grid, r, c, self.diagonals):
                    # meet: forward side has seen this node
                    if nb in self.came_from_front:
                        self.meeting_point = nb
                        if nb not in self.came_from_back:
                            self.came_from_back[nb] = current_b
                        break

                    if nb not in self.came_from_back:
                        self.came_from_back[nb] = current_b
                        frontier_back.append(nb)

        self.time_ms = (time.perf_counter() - t0) * 1000.0

        # reconstruct path
        if self.meeting_point is None:
            self.path = []
        else:
            self.path = self._reconstruct_bidirectional_path()

        # merge for metrics
        self.visited_orders = self.visited_orders_f + self.visited_orders_b
        self.came_from = {**self.came_from_front, **self.came_from_back}

    def _reconstruct_bidirectional_path(self):
        meet = self.meeting_point

        # forward: meet -> start (then reverse)
        forward = []
        cur = meet
        while cur != self.start:
            forward.append(cur)
            cur = self.came_from_front[cur]
        forward.append(self.start)
        forward.reverse()  # start -> ... -> meet

        # backward: meet -> goal using backward tree
        backward = []
        cur = meet
        while cur != self.goal:
            cur = self.came_from_back[cur]
            backward.append(cur)
        # backward: meet_next -> ... -> goal

        return forward + backward

    def get_result(self) -> dict:
        return helpers.build_result(
            algorithm="bidirectional_bfs",
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
