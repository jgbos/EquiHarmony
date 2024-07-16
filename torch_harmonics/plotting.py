""" plotting.py """

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy.typing as npt


def plot_circular_fn(
    data: npt.NDArray, fig: Figure = None, title: str = None, plot_neg: bool = False
):
    """Plot circular function"""
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="polar")

    x = np.linspace(0, 2 * np.pi, data.shape[0])
    if plot_neg:
        pos_x = x[data > 0]
        pos = data[data > 0]
        neg_x = x[data < 0]
        neg = data[data < 0]
        ax.plot(pos_x, np.abs(pos), color="blue")
        ax.plot(neg_x, np.abs(neg), color="orange")
    else:
        ax.plot(x, data, color="blue")

    ax.set_rmax(np.max(data) + 0.2)
    ax.set_rticks([])

    ax.set_title(title, va="bottom")
    ax.grid(True)
    return ax


def plot_polar_fn(data: npt.NDArray, fig: Figure = None, title: str = None):
    """Plot polar function"""
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="polar")

    r = np.linspace(0, np.max(data[0]), data.shape[0])
    phi = np.linspace(0, 2 * np.pi, data.shape[1])
    ax.pcolormesh(phi, r, data)

    ax.set_title(title, va="bottom")
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_xticklabels([])

    return ax


def plot_cylinder_fn(data: npt.NDArray, fig: Figure = None, title: str = None):
    """Plot cylinder function"""
    pass
