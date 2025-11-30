from __future__ import annotations

import os
import inspect
from pathlib import Path
from typing import List, Callable

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import graphviz as gr
import networkx as nx
from matplotlib import pyplot as plt


# ---- PATHS ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


# ---- DATA HELPERS ----
def load_data(dataset: str, delimiter: str = ";") -> pd.DataFrame:
    """
    Load a CSV from the project-level data directory.

    Parameters
    ----------
    dataset : str
        Base name of the CSV file (without extension).
    delimiter : str, optional
        Field delimiter, by default ";"
    """
    fname = f"{dataset}.csv"
    data_file = DATA_DIR / fname
    return pd.read_csv(data_file, sep=delimiter)


def crosstab(x: np.ndarray, y: np.ndarray, labels: list[str] | None = None) -> pd.DataFrame:
    """Simple cross tabulation of two discrete vectors x and y."""
    ct = pd.crosstab(x, y)
    if labels is not None:
        ct.index = labels
        ct.columns = labels
    return ct


def center(vals: np.ndarray) -> np.ndarray:
    return vals - np.nanmean(vals)


def standardize(vals: np.ndarray) -> np.ndarray:
    centered_vals = center(vals)
    return centered_vals / np.nanstd(vals)


def convert_to_categorical(vals: pd.Series | np.ndarray) -> np.ndarray:
    return pd.Series(vals).astype("category").cat.codes.values


def logit(p: float | np.ndarray) -> float | np.ndarray:
    return np.log(p / (1 - p))


def invlogit(x: float | np.ndarray) -> float | np.ndarray:
    return 1 / (1 + np.exp(-x))


# ---- GRAPH / DAG HELPERS ----
def draw_causal_graph(
    edge_list,
    node_props=None,
    edge_props=None,
    graph_direction: str = "UD",
):
    """Draw a causal (directed) graph using graphviz."""
    g = gr.Digraph(graph_attr={"rankdir": graph_direction})

    edge_props = {} if edge_props is None else edge_props
    for e in edge_list:
        props = edge_props.get(e, {})
        g.edge(e[0], e[1], **props)

    if node_props is not None:
        for name, props in node_props.items():
            g.node(name=name, **props)
    return g


# ---- MATPLOTLIB HELPERS ----
def plot_scatter(xs, ys, **scatter_kwargs):
    """Draw scatter plot with consistent style (e.g. unfilled points)."""
    defaults = {"alpha": 0.6, "lw": 3, "s": 80, "color": "C0", "facecolors": "none"}
    for k, v in defaults.items():
        scatter_kwargs.setdefault(k, v)
    plt.scatter(xs, ys, **scatter_kwargs)


def plot_line(xs, ys, **plot_kwargs):
    """Plot line with a white border + foreground line."""
    linewidth = plot_kwargs.get("linewidth", 3)
    plot_kwargs["linewidth"] = linewidth

    background_plot_kwargs = dict(plot_kwargs)
    background_plot_kwargs["linewidth"] = linewidth + 2
    background_plot_kwargs["color"] = "white"
    background_plot_kwargs.pop("label", None)

    plt.plot(xs, ys, **background_plot_kwargs, zorder=30)
    plt.plot(xs, ys, **plot_kwargs, zorder=31)


def plot_errorbar(
    xs,
    ys,
    error_lower,
    error_upper,
    colors="C0",
    error_width: float = 12,
    alpha: float = 0.3,
):
    """Draw thick vertical error bars with consistent style."""
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            yerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


def plot_x_errorbar(
    xs,
    ys,
    error_lower,
    error_upper,
    colors="C0",
    error_width: float = 12,
    alpha: float = 0.3,
):
    """Draw thick horizontal error bars with consistent style."""
    if isinstance(colors, str):
        colors = [colors] * len(xs)

    for ii, (x, y, err_l, err_u) in enumerate(zip(xs, ys, error_lower, error_upper)):
        marker, _, bar = plt.errorbar(
            x=x,
            y=y,
            xerr=np.array((err_l, err_u))[:, None],
            ls="none",
            color=colors[ii],
            zorder=1,
        )
        plt.setp(bar[0], capstyle="round")
        marker.set_fillstyle("none")
        bar[0].set_alpha(alpha)
        bar[0].set_linewidth(error_width)


def plot_graph(graph, **graph_kwargs):
    """Draw a network graph (networkx DiGraph or adjacency matrix)."""
    G = (
        nx.from_numpy_array(graph, create_using=nx.DiGraph)
        if isinstance(graph, np.ndarray)
        else graph
    )

    np.random.seed(123)
    if "layout" in graph_kwargs:
        graph_kwargs["pos"] = graph_kwargs["layout"](G)

    default_graph_kwargs = {
        "node_color": "C0",
        "node_size": 500,
        "arrowsize": 30,
        "width": 3,
        "alpha": 0.7,
        "connectionstyle": "arc3,rad=0.1",
        "pos": nx.kamada_kawai_layout(G),
    }
    for k, v in default_graph_kwargs.items():
        graph_kwargs.setdefault(k, v)

    nx.draw(G, **graph_kwargs)
    return graph_kwargs["pos"]


def plot_2d_function(xrange, yrange, func: Callable, ax=None, **contour_kwargs):
    """Evaluate func(xs, ys) over a grid and plot contours."""
    resolution = len(xrange)
    xs, ys = np.meshgrid(xrange, yrange)
    xs = xs.ravel()
    ys = ys.ravel()

    value = func(xs, ys)

    if ax is not None:
        plt.sca(ax)

    return plt.contour(
        xs.reshape(resolution, resolution),
        ys.reshape(resolution, resolution),
        value.reshape(resolution, resolution),
        **contour_kwargs,
    )


# ---- VARIABLE / DATAFRAME HELPERS ----
def get_variable_name(var):
    frame = inspect.currentframe().f_back
    for name, value in frame.f_locals.items():
        if value is var:
            return name
    return None


def create_variables_dataframe(*variables: List[np.ndarray]) -> pd.DataFrame:
    """Convert numpy arrays to a dataframe; infer column names from variable names."""
    column_names = [get_variable_name(v) for v in variables]
    return pd.DataFrame(np.vstack(variables).T, columns=column_names)


# ---- PYMC HELPERS ----
def plot_pymc_distribution(distribution: pm.Distribution, **distribution_params):
    """Plot a PyMC distribution with given parameters."""
    with pm.Model() as _:
        d = distribution(name=distribution.__name__, **distribution_params)
        draws = pm.draw(d, draws=10_000)
    return az.plot_dist(draws)


# ---- FIGURE / IMAGE HELPERS ----
def savefig(filename: str):
    """Save a figure to PROJECT_ROOT / 'images'."""
    image_path = PROJECT_ROOT / "images"
    if not image_path.exists():
        print(f"creating image directory: {image_path}")
        os.makedirs(image_path)

    figure_path = image_path / filename
    print(f"saving figure to {figure_path}")
    plt.savefig(figure_path, dpi=300, bbox_inches="tight")


def display_image(filename: str, width: int = 600):
    """Display an image saved under ./images relative to notebook."""
    from IPython.display import Image, display

    return display(Image(filename=f"images/{filename}", width=width))


# ---- SIMULATION / DEMO ----
def simulate_2_parameter_bayesian_learning_grid_approximation(
    x_obs,
    y_obs,
    param_a_grid,
    param_b_grid,
    true_param_a,
    true_param_b,
    model_func: Callable,
    posterior_func: Callable,
    n_posterior_samples: int = 3,
    param_labels=None,
    data_range_x=None,
    data_range_y=None,
):
    """Simulate Bayesian learning in a 2-parameter model using grid approximation."""
    param_labels = param_labels if param_labels is not None else ["param_a", "param_b"]
    data_range_x = (x_obs.min(), x_obs.max()) if data_range_x is None else data_range_x
    data_range_y = (y_obs.min(), y_obs.max()) if data_range_y is None else data_range_y

    resolution = len(param_a_grid)

    param_a_grid, param_b_grid = np.meshgrid(param_a_grid, param_b_grid)
    param_a_grid = param_a_grid.ravel()
    param_b_grid = param_b_grid.ravel()

    posterior = posterior_func(x_obs, y_obs, param_a_grid, param_b_grid)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # posterior over params
    plt.sca(axs[0])
    plt.contour(
        param_a_grid.reshape(resolution, resolution),
        param_b_grid.reshape(resolution, resolution),
        posterior.reshape(resolution, resolution),
        cmap="gray_r",
    )

    sample_idx = np.random.choice(
        np.arange(len(posterior)),
        p=posterior / posterior.sum(),
        size=n_posterior_samples,
    )

    param_a_list = []
    param_b_list = []
    for ii, idx in enumerate(sample_idx):
        param_a = param_a_grid[idx]
        param_b = param_b_grid[idx]
        param_a_list.append(param_a)
        param_b_list.append(param_b)

        plt.scatter(param_a, param_b, s=60, c=f"C{ii}", alpha=0.75, zorder=20)

    plt.scatter(
        true_param_a, true_param_b, color="k", marker="x", s=60, label="true parameters"
    )
    plt.xlabel(param_labels[0])
    plt.ylabel(param_labels[1])

    # data + sampled models
    plt.sca(axs[1])
    plt.scatter(x_obs, y_obs, s=60, c="k", alpha=0.5)

    xs = np.linspace(data_range_x[0], data_range_x[1], 100)
    for ii, (param_a, param_b) in enumerate(zip(param_a_list, param_b_list)):
        ys = model_func(xs, param_a, param_b)
        plt.plot(xs, ys, color=f"C{ii}", linewidth=4, alpha=0.5)

    groundtruth_ys = model_func(xs, true_param_a, true_param_b)
    plt.plot(xs, groundtruth_ys, color="k", linestyle="--", alpha=0.5, label="true trend")

    plt.xlim([data_range_x[0], data_range_x[1]])
    plt.ylim([data_range_y[0], data_range_y[1]])
    plt.xlabel("x value")
    plt.ylabel("y value")
    plt.title(f"N={len(y_obs)}")
    plt.legend(loc="upper left")

    return fig, axs