# Maze Pathfinding Lab

A web-based tool for comparing different pathfinding algorithms on randomly generated mazes. Test how various search algorithms perform against each other on the same maze configurations.

## Features

- **Multiple Pathfinding Algorithms**: Compare DFS, BFS, A*, JPS, Dijkstra, and GBFS
- **Dynamic Maze Generation**: Generate mazes using DFS or Prim's algorithm with customizable parameters
- **Visual Comparison**: See algorithm paths overlaid with different colors and spacing
- **Performance Metrics**: Track execution time, path length, nodes visited, and frontier peak
- **Configurable Parameters**: Adjust maze size, generation method, seed, and braiding factor

## Getting Started

### Prerequisites
- Python 3.7+
- Flask

### Installation

1. Clone this repository
```
git clone https://github.com/nirvaankohli/maze-pathfinding-lab.git
cd maze-pathfinding-lab
```

2. Install dependencies
```
pip install flask
```

3. Run the application
```
flask run
```

4. Open your browser to `http://localhost:5000`

## How to Use

1. **Generate a Maze**: Set your desired parameters (rows, columns, generation method, seed, braiding) and click "Generate Maze"
2. **Select Algorithms**: Choose which pathfinding algorithms you want to compare
3. **Run Comparison**: Click "Run Algorithms" to see how each performs on the generated maze
4. **View Results**: Examine the visual paths and performance metrics on the results page

## Algorithm Overview

- **DFS (Depth-First Search)**: Explores as far as possible along each branch before backtracking
- **BFS (Breadth-First Search)**: Explores all neighbors before moving to the next level
- **A (A-Star)**: Uses heuristics to find optimal paths efficiently  
- **JPS (Jump Point Search)**: Optimized A* variant that reduces nodes explored
- **Dijkstra**: Guarantees shortest path by exploring nodes in order of distance
- **GBFS (Greedy Best-First Search)**: Uses heuristics to guide search towards goal

## API Endpoints

- `GET /maze/generate` - Generate a new maze with specified parameters
- `POST /maze/solve` - Solve maze using selected algorithms
- `GET /maze/blank` - Generate blank grid template

### Example API Usage

Generate a 21x21 maze:
```
GET /maze/generate?rows=21&cols=21&method=dfs&seed=42&braiding=0.2
```

## Configuration Options

- **Rows/Columns**: Maze dimensions (minimum 5x5)
- **Generation Method**: DFS or Prim's algorithm
- **Seed**: Random seed for reproducible mazes
- **Braiding**: Factor (0.0-1.0) that adds loops by removing dead ends

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.