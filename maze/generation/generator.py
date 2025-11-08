import math
import random
from collections import deque
from typing import Tuple, List, Dict, Optional
import json

Cell = Tuple[int, int]
Grid = List[List[int]]


def to_json(
    grid: Grid, start: Cell, goal: Cell, metrics: Dict, file_name: str = "maze.json"
) -> Dict:

    json_data = {
        "maze": grid,
        "start": list(start),
        "goal": list(goal),
        "metrics": metrics,
    }
    with open(file_name, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    return json_data


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
        self.method = (method or "dfs").lower()
        self.seed = seed
        self.braiding = max(0.0, min(1.0, braiding))
        if seed is not None:
            random.seed(seed)

    # Public API
    def generate(self) -> Tuple[Grid, Cell, Cell, Dict]:

        if self.seed is not None:
            random.seed(self.seed)

        rows = self._ensure_odd(max(5, self.rows))
        cols = self._ensure_odd(max(5, self.cols))

        start: Cell = (1, 1)
        goal: Cell = (rows - 2, cols - 2)

        attempts = min(300, max(1, (rows * cols) // 4))
        best_score = -math.inf
        best_grid: Optional[Grid] = None
        best_metrics: Optional[Dict] = None

        for attempt in range(attempts):
            grid = self._blank_grid(rows, cols)
            if self.method == "prim":
                self._maze_prim(grid)
            else:
                self._maze_recursive_backtracker(grid)

            # ensure start/goal are open
            self._open_cell(grid, start[0], start[1], force=True)
            self._open_cell(grid, goal[0], goal[1], force=True)

            # optionally add loops (braiding) — small, controlled amount
            if self.braiding > 0.0:
                self._braid(grid, self.braiding)

            # discard unsolvable candidates quickly
            if not self._is_solvable(grid, start, goal):
                continue

            pl = self._path_length(grid, start, goal) or 0
            metrics = self._maze_metrics(grid)
            junctions = metrics.get("junctions", 0)
            dead_ends = metrics.get("dead_ends", 0)

            # scoring heuristic: prefer long shortest-paths and many junctions
            score = pl * 6 + junctions * 2 + dead_ends

            if score > best_score:
                best_score = score
                best_grid = grid
                best_metrics = metrics

        # If nothing passed, fallback to a single generation
        if best_grid is None:
            grid = self._blank_grid(rows, cols)
            if self.method == "prim":
                self._maze_prim(grid)
            else:
                self._maze_recursive_backtracker(grid)
            self._open_cell(grid, start[0], start[1], force=True)
            self._open_cell(grid, goal[0], goal[1], force=True)
            if self.braiding > 0.0:
                self._braid(grid, self.braiding)
            metrics = self._maze_metrics(grid)
        else:
            grid = best_grid
            metrics = best_metrics or {}

        # small densify pass to add more loops if user requested more braiding
        if self.braiding > 0.0:
            try:
                densify_attempts = max(20, (rows * cols) // 8)
                intensity = int(densify_attempts * min(2.0, 0.5 + 2.0 * self.braiding))
                self._densify_loops(grid, attempts=intensity)
                metrics = self._maze_metrics(grid)
            except Exception:
                metrics = self._maze_metrics(grid)

        # final solvability guard
        if not self._is_solvable(grid, start, goal):
            self._carve_path(grid, start, goal)
            metrics = self._maze_metrics(grid)

        return grid, start, goal, metrics

    def _blank_grid(self, rows: int, cols: int) -> Grid:
        return [[1] * cols for _ in range(rows)]

    def _ensure_odd(self, v: int) -> int:
        return v if v % 2 == 1 else v + 1

    def _neighbors_2step(self, r: int, c: int, rows: int, cols: int):

        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                yield nr, nc, r + dr // 2, c + dc // 2

    def _creates_2x2_if_open(self, grid: Grid, r: int, c: int) -> bool:
        rows, cols = len(grid), len(grid[0])

        for sr in (r - 1, r):
            for sc in (c - 1, c):
                if sr < 0 or sc < 0 or sr + 1 >= rows or sc + 1 >= cols:
                    continue
                count_open = 0
                for rr in (sr, sr + 1):
                    for cc in (sc, sc + 1):
                        if rr == r and cc == c:
                            count_open += 1
                        elif grid[rr][cc] == 0:
                            count_open += 1
                if count_open == 4:
                    return True
        return False

    def _open_cell(self, grid: Grid, r: int, c: int, force: bool = False) -> bool:
        if grid[r][c] == 0:
            return True
        if not force and self._creates_2x2_if_open(grid, r, c):
            return False
        grid[r][c] = 0
        return True

    def _maze_recursive_backtracker(self, grid: Grid) -> None:
        rows, cols = len(grid), len(grid[0])
        start = (1, 1)
        stack = [start]
        self._open_cell(grid, start[0], start[1], force=True)

        while stack:
            r, c = stack[-1]
            neighbors = [
                t
                for t in self._neighbors_2step(r, c, rows, cols)
                if grid[t[0]][t[1]] == 1
            ]
            if not neighbors:
                stack.pop()
                continue

            # prefer neighbors that won't create wide open areas
            random.shuffle(neighbors)
            chosen = None
            for nr, nc, wr, wc in neighbors:
                if not self._creates_2x2_if_open(
                    grid, wr, wc
                ) and not self._creates_2x2_if_open(grid, nr, nc):
                    chosen = (nr, nc, wr, wc)
                    break
            if chosen is None:
                chosen = neighbors[0]

            nr, nc, wr, wc = chosen
            self._open_cell(grid, wr, wc, force=True)
            self._open_cell(grid, nr, nc, force=True)
            stack.append((nr, nc))

    def _maze_prim(self, grid: Grid) -> None:
        rows, cols = len(grid), len(grid[0])
        start = (1, 1)
        self._open_cell(grid, start[0], start[1], force=True)

        walls: List[Tuple[int, int, int, int]] = []

        def add_walls(cr: int, cc: int) -> None:
            for nr, nc, wr, wc in self._neighbors_2step(cr, cc, rows, cols):
                if grid[nr][nc] == 1:
                    walls.append((wr, wc, nr, nc))

        add_walls(*start)

        while walls:
            idx = random.randrange(len(walls))
            wr, wc, nr, nc = walls.pop(idx)
            if grid[nr][nc] == 1:
                # safe-first strategy
                if not self._creates_2x2_if_open(
                    grid, wr, wc
                ) and not self._creates_2x2_if_open(grid, nr, nc):
                    self._open_cell(grid, wr, wc)
                    self._open_cell(grid, nr, nc)
                    add_walls(nr, nc)
                else:
                    # if unsafe, still progress occasionally to avoid stalls
                    self._open_cell(grid, wr, wc, force=True)
                    self._open_cell(grid, nr, nc, force=True)
                    add_walls(nr, nc)

    # small post-processing to reduce dead-ends (braid)
    def _braid(self, grid: Grid, fraction: float) -> None:
        frac = max(0.0, min(1.0, fraction))
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
                    if grid[r + dr][c + dc] == 0:
                        deg += 1
                if deg == 1:
                    dead_ends.append((r, c))
        if not dead_ends:
            return
        random.shuffle(dead_ends)
        k = max(1, int(len(dead_ends) * frac))
        for r, c in dead_ends[:k]:
            # connect this dead-end to a random neighbor by opening an adjacent wall
            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                wr, wc = r + dr, c + dc
                if 0 <= wr < rows and 0 <= wc < cols and grid[wr][wc] == 1:
                    candidates.append((wr, wc))
            if not candidates:
                continue
            wr, wc = random.choice(candidates)
            if not self._creates_2x2_if_open(grid, wr, wc):
                self._open_cell(grid, wr, wc)
            else:
                # rarely force to increase loops if requested
                if random.random() < 0.05:
                    self._open_cell(grid, wr, wc, force=True)

    def _densify_loops(self, grid: Grid, attempts: int = 200) -> None:
        rows, cols = len(grid), len(grid[0])
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
            neigh = 0
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
                    neigh += 1
            if neigh < 2:
                continue
            if self._creates_2x2_if_open(grid, r, c):
                if random.random() < 0.02:
                    self._open_cell(grid, r, c, force=True)
                continue
            self._open_cell(grid, r, c)

    # ---------------- utility / metrics / solvability ---------------------
    def _is_solvable(self, grid: Grid, start: Cell, goal: Cell) -> bool:
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
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in seen
                    and grid[nr][nc] == 0
                ):
                    seen.add((nr, nc))
                    q.append((nr, nc))
        return False

    def _carve_path(self, grid: Grid, start: Cell, goal: Cell) -> None:
        # BFS that can step into walls (prefer safe opens) and then carve the path
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
            return
        cur = (gr, gc)
        while cur is not None:
            r, c = cur
            if grid[r][c] == 1:
                self._open_cell(grid, r, c, force=used_unsafe)
            cur = prev[cur]

    def _path_length(self, grid: Grid, start: Cell, goal: Cell) -> Optional[int]:
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
                if (
                    0 <= nr < rows
                    and 0 <= nc < cols
                    and (nr, nc) not in seen
                    and grid[nr][nc] == 0
                ):
                    seen.add((nr, nc))
                    q.append((nr, nc, d + 1))
        return None

    def _maze_metrics(self, grid: Grid) -> Dict[str, int]:
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
                elif deg >= 3:
                    junctions += 1
        return {"dead_ends": dead_ends, "junctions": junctions}


if __name__ == "__main__":
    # quick example and smoke test
    gen = RandomMazeGenerator(rows=41, cols=41, method="dfs", seed=42, braiding=0.12)
    maze, start, goal, metrics = gen.generate()
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
