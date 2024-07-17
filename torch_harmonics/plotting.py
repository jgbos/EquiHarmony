""" plotting.py """

import numpy as np
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator, interpn

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
    if fig is None:
        fig = plt.figure()

    ax = fig.add_subplot(projection="3d")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    r = np.linspace(0, 1, 10)
    p = np.linspace(0, 2 * np.pi, 36)
    z = np.linspace(0, 1, 10)

    P, R, zs = np.meshgrid(p, r, z)
    xs = (R * np.cos(P)).flatten()
    ys = (R * np.sin(P)).flatten()
    zs = zs.flatten()

    X = np.linspace(-1, 1, 50)
    Y = np.linspace(-1, 1, 50)
    Z = np.linspace(0, 1, 50)
    X, Y, Z = np.meshgrid(X, Y, Z)

    interp = LinearNDInterpolator(list(zip(xs, ys, zs)), data.flatten())
    idata = interp(X, Y, Z)
    idata[np.isnan(idata)] = 0
    mask = idata != 0.0

    plot = ax.scatter(
        X.flatten(),
        Y.flatten(),
        Z.flatten(),
        c=idata.flatten(),
        s=10.0 * mask,
        edgecolor="face",
        alpha=0.2,
        marker="o",
        cmap="magma",
        linewidth=0,
    )

    ax.set_title(title, va="bottom")

    return ax
