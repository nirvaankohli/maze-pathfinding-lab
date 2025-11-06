from flask import Flask, request
from maze.generation.generator import RandomMazeGenerator

app = Flask(__name__)


@app.route("/")
def home():

    return "this gon be something"


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

    generator = RandomMazeGenerator(
        rows=params_dict.get("rows", 21),
        cols=params_dict.get("cols", 21),
        method=params_dict.get("method", "dfs"),
        seed=params_dict.get("seed", 42),
        braiding=params_dict.get("braiding", 0.2),
    )

    maze, start, goal, metrics = generator.generate()

    return {"maze": maze, "start": start, "goal": goal, "metrics": metrics}


@app.route("/maze/blank")
def blank_maze():

    params = request.args
    params_dict = params.to_dict()

    generator = RandomMazeGenerator(
        rows=params_dict.get("rows", 21),
        cols=params_dict.get("cols", 21),
        method="dfs",
        seed=42,
        braiding=0.2,
    )

    return generator.blank_grid(
        rows=params_dict.get("rows", 21),
        cols=params_dict.get("cols", 21),
    )


if __name__ == "__main__":

    app.run(debug=True)
