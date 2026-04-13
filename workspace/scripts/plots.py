"""Plotting utilities for Lab 5 CiM experiments."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def bar_side_by_side(data, xlabel="", ylabel="", title="", ax=None):
    """Create a grouped bar chart.

    Parameters
    ----------
    data : dict[str, dict[str, float]]
        Outer keys are x-axis categories, inner keys are series names,
        values are bar heights.
    xlabel, ylabel, title : str
        Axis labels and chart title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. Created if not given.
    """
    if ax is None:
        _, ax = plt.subplots()

    categories = list(data.keys())
    if not categories:
        return ax

    series_names = list(
        set(k for inner in data.values() for k in inner.keys())
    )

    x = np.arange(len(categories))
    width = 0.8 / max(len(series_names), 1)

    for i, series in enumerate(series_names):
        offset = (i - len(series_names) / 2 + 0.5) * width
        values = [data[cat].get(series, 0) for cat in categories]
        label = series if series else None
        ax.bar(x + offset, values, width, label=label)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
    if any(s for s in series_names):
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return ax


def plot(data, xlabel="", ylabel="", ax=None, title=""):
    """Create a simple bar chart from a dict.

    Parameters
    ----------
    data : dict[str, float] or dict[str, dict[str, float]]
        If values are floats, creates a simple bar chart.
        If values are dicts, creates a grouped bar chart.
    xlabel, ylabel, title : str
        Axis labels and title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    """
    if ax is None:
        _, ax = plt.subplots()

    categories = list(data.keys())
    values = list(data.values())

    if not categories:
        return ax

    # Check if nested dict
    if isinstance(values[0], dict):
        return bar_side_by_side(data, xlabel, ylabel, title, ax)

    x = np.arange(len(categories))
    ax.bar(x, values)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
    ax.grid(axis="y", alpha=0.3)
    return ax


def bar_stacked(data, xlabel="", ylabel="", title="", ax=None):
    """Create a stacked bar chart.

    Parameters
    ----------
    data : dict[str, dict[str, float]]
        Outer keys are x-axis categories, inner keys are stack categories.
    xlabel, ylabel, title : str
        Axis labels and title.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    categories = list(data.keys())
    stack_keys = list(set(k for inner in data.values() for k in inner.keys()))

    x = np.arange(len(categories))
    bottoms = np.zeros(len(categories))

    for stack_key in stack_keys:
        heights = [data[cat].get(stack_key, 0) for cat in categories]
        ax.bar(x, heights, label=stack_key, bottom=bottoms)
        bottoms += np.array(heights)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in categories], rotation=45, ha="right")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    return ax
