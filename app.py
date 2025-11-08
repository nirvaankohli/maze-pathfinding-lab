from flask import Flask, render_template, request
from maze.generation.generator import RandomMazeGenerator

app = Flask(__name__)


@app.route("/")
def home():

    return render_template("index.html")


# maze api /maze


@app.route("/maze")
def maze():

    return "maze docs coming soon..."


@app.route("/maze/generate")
def generate_maze():
    """

    Example request:
        |
        |   GET
        |   /maze/generate?rows=21&cols=21&method=dfs&seed=42&braiding=0.2
        |
    Example response:
        |
        |    {
        |        "maze": [[0, 1, 0, ...], [0, 0, 1, ...], ...],
        |        "start": [1, 1],
        |        "goal": [19, 19],
        |        "metrics": {
        |            "path_length": 150,
        |            "junctions": 30,
        |            "dead_ends": 25,
        |            ...
        |        }
        |    }

    """

    params = request.args
    params_dict = params.to_dict()

    rows = int(params_dict.get("rows", 21))
    cols = int(params_dict.get("cols", 21))
    method = params_dict.get("method", "dfs")
    seed_raw = params_dict.get("seed")
    seed = int(seed_raw) if seed_raw is not None and seed_raw != "" else None
    try:
        braiding = float(params_dict.get("braiding", 0.2))
    except Exception:
        braiding = 0.2

    generator = RandomMazeGenerator(
        rows=rows, cols=cols, method=method, seed=seed, braiding=braiding
    )

    maze, start, goal, metrics = generator.generate()

    path_len = generator._path_length(maze, start, goal) or 0
    metrics_out = dict(metrics)
    metrics_out.setdefault("path_length", path_len)

    return {
        "maze": maze,
        "start": list(start),
        "goal": list(goal),
        "metrics": metrics_out,
    }


@app.route("/maze/blank")
def blank_maze():

    params = request.args
    params_dict = params.to_dict()

    rows = int(params_dict.get("rows", 21))
    cols = int(params_dict.get("cols", 21))
    generator = RandomMazeGenerator(
        rows=rows, cols=cols, method="dfs", seed=None, braiding=0.0
    )
    return generator._blank_grid(rows, cols)


if __name__ == "__main__":

    app.run(debug=True)
