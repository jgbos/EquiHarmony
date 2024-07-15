""" plotting.py """

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt

import numpy.typing as npt


def plot_circular_fn(data: npt.NDArray, fig: Figure = None, title: str = None):
    """Plot circular function"""
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="polar")
    ax.plot(np.linspace(0, 2 * np.pi, data.shape[0]), data)

    ax.set_rmax(np.max(data) + 0.2)
    ax.set_rticks([])

    ax.set_title(title, va="bottom")
    ax.grid(True)
    return ax


def plot_polar_fn(data: npt.NDArray, fig: Figure = None, title: str = None):
    """Plot polar function"""
    pass


def plot_cylinder_fn(data: npt.NDArray, fig: Figure = None, title: str = None):
    """Plot cylinder function"""
    pass
