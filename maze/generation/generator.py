import math
import random
from collections import deque
from typing import Tuple, List, Dict, Optional

# return formats

Cell = Tuple[int, int]
Grid = List[List[int]]


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

            grid_try[start[0]][start[1]] = 0
            grid_try[goal[0]][goal[1]] = 0

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

            grid[start[0]][start[1]] = 0
            grid[goal[0]][goal[1]] = 0

            if self.braiding > 0.0:

                self.braid(grid, self.braiding)

            metrics = self.maze_metrics(grid)

        else:

            grid = best_grid
            metrics = best_metrics or {}

            # try to make the chosen maze harder without breaking
            # solvability by selectively closing passages
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

    def maze_recursive_backtracker(self, grid: Grid) -> None:

        rows, cols = len(grid), len(grid[0])

        start = (1, 1)  # starting cell
        stack = [start]
        grid[start[0]][start[1]] = 0  # so u can actually start at the start

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

            nr, nc, hr, hc = random.choice(candidates)

            grid[r + hr][c + hc] = 0  # remove wall
            grid[nr][nc] = 0  # mark cell as passage
            stack.append((nr, nc))

    # prim maze generation

    def maze_prim(self, grid: Grid) -> None:

        rows, cols = len(grid), len(grid[0])

        start = (1, 1)  # starting cell
        grid[start[0]][start[1]] = 0  # starting again

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

                grid[wr][wc] = 0  # remove wall
                grid[nr][nc] = 0  # mark cell as passage
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
            grid[wr][wc] = 0  # remove wall

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

        q = deque([(sr, sc)])
        prev = {(sr, sc): None}

        found = False

        while q:

            r, c = q.popleft()

            if (r, c) == (gr, gc):

                found = True
                break

            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:

                nr, nc = r + dr, c + dc

                if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in prev:

                    prev[(nr, nc)] = (r, c)
                    q.append((nr, nc))

        if not found:
            # Shouldn't happen on a bounded grid, but guard anyway.
            return

        # Reconstruct path and carve it.
        cur = (gr, gc)
        while cur is not None:
            r, c = cur
            grid[r][c] = 0
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
        rows=21, cols=21, method="dfs", seed=42, braiding=0.2
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
