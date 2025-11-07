import math
import random
from collections import deque
from typing import Tuple, List, Dict, Optional
import json

# return formats

Cell = Tuple[int, int]
Grid = List[List[int]]


def to_json(
    grid: Grid, start: Cell, goal: Cell, metrics: Dict, file_name="maze.json"
) -> Dict:

    json_data = {
        "maze": grid,
        "start": list(start),
        "goal": list(goal),
        "metrics": metrics,
    }

    with open(file_name, "w") as f:

        json.dump(json_data, f, indent=4)


class RandomMazeGenerator:

    def __init__(
        self,
        rows: int,
        cols: int,
        method: str = "dfs",
        seed: Optional[int] = None,
        braiding: float = 0.0,
    ) -> None:

        self.rows = rows
        self.cols = cols
        self.method = method
        self.seed = seed
        self.braiding = braiding

        if seed is not None:

            random.seed(seed)

    def generate(self) -> Tuple[Grid, Cell, Cell, Dict]:

        if self.seed is not None:

            random.seed(self.seed)

        rows = self.ensure_odd(max(5, self.rows))
        cols = self.ensure_odd(max(5, self.cols))

        grid = self.blank_grid(rows, cols)

        start = (1, 1)
        goal = (rows - 2, cols - 2)

        attempts = min(1000, max(100, rows * cols))
        best_score = -1
        best_grid: Optional[Grid] = None
        best_metrics: Optional[Dict] = None

        method = (self.method or "dfs").lower()

        for attempt in range(attempts):

            grid_try = self.blank_grid(rows, cols)

            if method == "prim":

                self.maze_prim(grid_try)

            else:

                self.maze_recursive_backtracker(grid_try)

            # ensure start/goal are passages (force open)
            self._open_cell(grid_try, start[0], start[1], force=True)
            self._open_cell(grid_try, goal[0], goal[1], force=True)

            if self.braiding > 0.0:

                self.braid(grid_try, self.braiding)

            if not self.is_solvable(grid_try, start, goal):

                if self.seed is not None:

                    random.seed(self.seed + attempt + 1)

                else:

                    random.seed()

                continue

            pl = self.path_length(grid_try, start, goal) or 0
            metrics_try = self.maze_metrics(grid_try)
            junctions = metrics_try.get("junctions", 0)
            dead_ends = metrics_try.get("dead_ends", 0)

            # emphasize path length and dead-ends to increase difficulty
            score = pl * 8 + junctions * 2 + dead_ends

            if score > best_score:
                best_score = score
                best_grid = grid_try
                best_metrics = metrics_try

            if self.seed is not None:
                random.seed(self.seed + attempt + 1)
            else:
                random.seed()

        if best_grid is None:

            # fallback to single generation

            grid = self.blank_grid(rows, cols)
            if method == "prim":

                self.maze_prim(grid)
            else:

                self.maze_recursive_backtracker(grid)

            # ensure start/goal are passages (force open)
            self._open_cell(grid, start[0], start[1], force=True)
            self._open_cell(grid, goal[0], goal[1], force=True)

            if self.braiding > 0.0:

                self.braid(grid, self.braiding)

            metrics = self.maze_metrics(grid)

        else:

            grid = best_grid
            metrics = best_metrics or {}


            try:
                densify_attempts = max(50, (rows * cols) // 4)
                # scale by braiding parameter to respect user's desired braiding
                intensity = int(
                    densify_attempts * (0.5 + 1.5 * min(1.0, self.braiding))
                )
                self.densify_loops(grid, attempts=intensity)
            except Exception:
                pass


            try:
                aggravate_attempts = max(10, (rows * cols) // 10)
                self.aggravate(grid, start, goal, aggravate_attempts)
                metrics = self.maze_metrics(grid)
            except Exception:
                metrics = self.maze_metrics(grid)

        if not self.is_solvable(grid, start, goal):
            self.carve_path(grid, start, goal)
            metrics = self.maze_metrics(grid)

        return grid, start, goal, metrics

    # helpers

    def blank_grid(self, rows: int, cols: int) -> Grid:

        return [[1] * cols for _ in range(rows)]

    def ensure_odd(self, value: int) -> int:

        return value if value % 2 == 1 else value + 1

    def neighbors_2step(self, r: int, c: int, rows: int, cols: int):

        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:

            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:

                hr, hc = dr // 2, dc // 2
                yield (nr, nc, hr, hc)

    def _creates_2x2_if_open(self, grid: Grid, r: int, c: int) -> bool:
        rows, cols = len(grid), len(grid[0])
        for dr in (0, -1):
            for dc in (0, -1):
                sr, sc = r + dr, c + dc
                if sr < 0 or sc < 0 or sr + 1 >= rows or sc + 1 >= cols:
                    continue

                zeros = 0
                for rr in (sr, sr + 1):
                    for cc in (sc, sc + 1):
                        if (rr, cc) == (r, c):
                            zeros += 1
                        elif grid[rr][cc] == 0:
                            zeros += 1
                if zeros == 4:
                    return True
        return False

    def _open_cell(self, grid: Grid, r: int, c: int, force: bool = False) -> bool:
        if grid[r][c] == 0:
            return True
        if not force and self._creates_2x2_if_open(grid, r, c):
            return False
        grid[r][c] = 0
        return True

    def maze_recursive_backtracker(self, grid: Grid) -> None:

        rows, cols = len(grid), len(grid[0])

        start = (1, 1)  # starting cell
        stack = [start]
        self._open_cell(grid, start[0], start[1], force=True)  # ensure start is open

        while stack:

            r, c = stack[-1]

            candidates = [
                (nr, nc, hr, hc)
                for (nr, nc, hr, hc) in self.neighbors_2step(r, c, rows, cols)
                if grid[nr][nc] == 1
            ]

            if not candidates:

                stack.pop()

                continue

            # prefer candidates that won't create 2x2 open blocks
            viable = []
            for nr, nc, hr, hc in candidates:
                wr, wc = r + hr, c + hc
                if not self._creates_2x2_if_open(
                    grid, wr, wc
                ) and not self._creates_2x2_if_open(grid, nr, nc):
                    viable.append((nr, nc, hr, hc))

            if viable:
                nr, nc, hr, hc = random.choice(viable)
                self._open_cell(grid, r + hr, c + hc)
                self._open_cell(grid, nr, nc)
            else:
                nr, nc, hr, hc = random.choice(candidates)
                # force open if no safe candidate exists
                self._open_cell(grid, r + hr, c + hc, force=True)
                self._open_cell(grid, nr, nc, force=True)

            stack.append((nr, nc))

    # prim maze generation

    def maze_prim(self, grid: Grid) -> None:

        rows, cols = len(grid), len(grid[0])

        start = (1, 1)  # starting cell
        self._open_cell(grid, start[0], start[1], force=True)  # starting again

        walls: List[Tuple[int, int, int, int]] = []

        def add_walls(cr: int, cc: int) -> None:

            for nr, nc, hr, hc in self.neighbors_2step(cr, cc, rows, cols):
                if grid[nr][nc] == 1:
                    wr, wc = cr + hr, cc + hc
                    walls.append((wr, wc, nr, nc))

        add_walls(*start)

        while walls:

            idx = random.randrange(len(walls))
            wr, wc, nr, nc = walls.pop(idx)

            if grid[nr][nc] == 1:

                # prefer openings that don't create 2x2 open blocks
                if not self._creates_2x2_if_open(
                    grid, wr, wc
                ) and not self._creates_2x2_if_open(grid, nr, nc):
                    self._open_cell(grid, wr, wc)
                    self._open_cell(grid, nr, nc)
                    add_walls(nr, nc)
                else:
                    # no safe option; force open to keep algorithm progressing
                    self._open_cell(grid, wr, wc, force=True)
                    self._open_cell(grid, nr, nc, force=True)
                    add_walls(nr, nc)

    # braid - to reduce dead ends

    def braid(self, grid: Grid, braiding: float) -> None:

        frac = max(0.0, min(1.0, braiding))

        if frac <= 0.0:

            return

        rows, cols = len(grid), len(grid[0])

        dead_ends: List[Cell] = []

        for r in range(1, rows - 1):

            for c in range(1, cols - 1):

                if grid[r][c] != 0:

                    continue

                deg = 0

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                    if 0 <= r + dr < rows and 0 <= c + dc < cols:

                        if grid[r + dr][c + dc] == 0:

                            deg += 1

                if deg == 1:

                    dead_ends.append((r, c))

        if not dead_ends:

            return

        random.shuffle(dead_ends)
        k = math.floor(len(dead_ends) * frac)

        for r, c in dead_ends[:k]:

            wall_candidates = []

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                wr, wc = r + dr, c + dc

                if 0 <= wr < rows and 0 <= wc < cols:

                    wall_candidates.append((wr, wc))

            # if there are no candidates, skip
            if not wall_candidates:
                continue

            wr, wc = random.choice(wall_candidates)
            # try safe open, otherwise force
            if not self._creates_2x2_if_open(grid, wr, wc):
                self._open_cell(grid, wr, wc)
            else:
                self._open_cell(grid, wr, wc, force=True)

    # solvability helpers

    def is_solvable(self, grid: Grid, start: Cell, goal: Cell) -> bool:

        # return True if there's a path from start to goal using passages (0).

        rows, cols = len(grid), len(grid[0])
        sr, sc = start
        gr, gc = goal

        if grid[sr][sc] != 0 or grid[gr][gc] != 0:

            return False

        q = deque([(sr, sc)])
        seen = {(sr, sc)}

        while q:

            r, c = q.popleft()

            if (r, c) == (gr, gc):

                return True

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                nr, nc = r + dr, c + dc

                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:

                    if grid[nr][nc] == 0:

                        seen.add((nr, nc))
                        q.append((nr, nc))

        return False

    def carve_path(self, grid: Grid, start: Cell, goal: Cell) -> None:
        rows, cols = len(grid), len(grid[0])
        sr, sc = start
        gr, gc = goal

        def bfs(allow_unsafe: bool):
            q = deque([(sr, sc)])
            prev = {(sr, sc): None}
            while q:
                r, c = q.popleft()
                if (r, c) == (gr, gc):
                    return prev
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in prev:
                        if grid[nr][nc] == 0:
                            prev[(nr, nc)] = (r, c)
                            q.append((nr, nc))
                        else:
                            # it's a wall; only consider if opening it won't
                            # create a 2x2, or if we allow unsafe openings.
                            if allow_unsafe or not self._creates_2x2_if_open(
                                grid, nr, nc
                            ):
                                prev[(nr, nc)] = (r, c)
                                q.append((nr, nc))
            return None

        prev = bfs(False)
        used_unsafe = False
        if prev is None:
            prev = bfs(True)
            used_unsafe = True

        if prev is None:
            # no path found even allowing unsafe openings
            return

        # Reconstruct path and carve it. Prefer safe opens; if we used unsafe
        # BFS then allow force opening.
        cur = (gr, gc)
        while cur is not None:
            r, c = cur
            if grid[r][c] == 1:
                self._open_cell(grid, r, c, force=used_unsafe)
            cur = prev[cur]

    def path_length(self, grid: Grid, start: Cell, goal: Cell) -> Optional[int]:

        rows, cols = len(grid), len(grid[0])
        sr, sc = start
        gr, gc = goal

        if grid[sr][sc] != 0 or grid[gr][gc] != 0:
            return None

        q = deque([(sr, sc, 0)])
        seen = {(sr, sc)}

        while q:
            r, c, d = q.popleft()
            if (r, c) == (gr, gc):
                return d

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in seen:
                    if grid[nr][nc] == 0:
                        seen.add((nr, nc))
                        q.append((nr, nc, d + 1))

        return None

    def aggravate(
        self, grid: Grid, start: Cell, goal: Cell, attempts: int = 50
    ) -> None:

        rows, cols = len(grid), len(grid[0])

        # compute baseline shortest path
        base_len = self.path_length(grid, start, goal) or 0

        passage_cells = [
            (r, c)
            for r in range(1, rows - 1)
            for c in range(1, cols - 1)
            if grid[r][c] == 0 and (r, c) != start and (r, c) != goal
        ]

        if not passage_cells:
            return

        for _ in range(attempts):
            r, c = random.choice(passage_cells)

            # temporarily close
            grid[r][c] = 1

            if self.is_solvable(grid, start, goal):
                new_len = self.path_length(grid, start, goal) or 0
                if new_len > base_len:
                    # keep the wall and update baseline & passage list
                    base_len = new_len
                    passage_cells.remove((r, c))
                else:
                    # revert
                    grid[r][c] = 0
            else:
                # revert
                grid[r][c] = 0

    def densify_loops(self, grid: Grid, attempts: int = 200) -> None:
        """Try to add loops by removing suitable wall cells. We prefer
        removals that connect two or more distinct passages and do not
        create 2x2 open blocks (keep passages one-cell-wide when possible).
        The method is opportunistic and randomized.
        """

        rows, cols = len(grid), len(grid[0])

        # Collect candidate wall cells (not border)
        candidates = [
            (r, c)
            for r in range(1, rows - 1)
            for c in range(1, cols - 1)
            if grid[r][c] == 1
        ]

        if not candidates:
            return

        for _ in range(attempts):
            r, c = random.choice(candidates)

            # count distinct passage neighbors and opposite-side connections
            neigh_pass = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    neigh_pass += 1

            # we want removals that actually create loops / alternative paths
            if neigh_pass < 2:
                continue

            # avoid creating 2x2 open blocks
            if self._creates_2x2_if_open(grid, r, c):
                # occasionally force if randomness desires more loops
                if random.random() < 0.05:
                    self._open_cell(grid, r, c, force=True)
                continue

            # open safely
            self._open_cell(grid, r, c)

    # maze metrics

    def maze_metrics(self, grid: Grid) -> Dict[str, int]:

        rows, cols = len(grid), len(grid[0])
        dead_ends = 0
        junctions = 0

        for r in range(1, rows - 1):

            for c in range(1, cols - 1):

                if grid[r][c] != 0:

                    continue

                deg = 0

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                    if grid[r + dr][c + dc] == 0:

                        deg += 1

                if deg == 1:

                    dead_ends += 1

                if deg >= 3:

                    junctions += 1

        return {
            "dead_ends": dead_ends,
            "junctions": junctions,
        }


# Example usage:

if __name__ == "__main__":

    generator = RandomMazeGenerator(
        rows=40, cols=40, method="dfs", seed=42, braiding=0.2
    )
    maze, start, goal, metrics = generator.generate()

    for r, row in enumerate(maze):
        line = []
        for c, cell in enumerate(row):
            if (r, c) == start:
                line.append("S")
            elif (r, c) == goal:
                line.append("G")
            else:
                line.append("#" if cell == 1 else " ")
        print("".join(line))

    print(f"Start: {start}, Goal: {goal}, Metrics: {metrics}")

    to_json(maze, start, goal, metrics, file_name="generated_maze.json")
