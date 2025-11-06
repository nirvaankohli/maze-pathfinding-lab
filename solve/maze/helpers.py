from __future__ import annotations
from typing import List, Tuple, Dict, Iterable
import math

Grid = List[List[int]]  # 0 for open cell, 1 for wall
Cell = Tuple[int, int]
Path = List[Cell]
INF = float("inf")


def neighbors(grid: Grid, r: int, c: int, diagonals: bool = False) -> Iterable[Cell]:

    # returns neighboring cells that are available to move to

    rows, cols = len(grid), len(grid[0])
    steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]