""" grid.py """

from typing import Tuple
import torch


def grid1D(
    boxsize: float, ngrid: int, origin: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns the x coordinates of a cartesian grid.

    Args:
        boxsize : Box size.
        ngrid : Grid division along one axis.
        origin : Start point of the grid.
    """
    xedges = torch.linspace(0.0, boxsize, ngrid + 1) + origin
    x = 0.5 * (xedges[1:] + xedges[:-1])
    return xedges, x


def polar_grid(
    r_max: float, num_r: int, num_phi: int, r_min: float = 0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a 2D polar grid.

    Args:
        r_max - Maximum radius.
        Nr - Number of elements along the radial axis.
        Nphi- Number of elements along the angular axis.
        r_min - Minimum radius.
    """
    _, r = grid1D(r_max, num_r, origin=r_min)
    _, p = grid1D(2.0 * torch.pi, num_phi)
    r2d, p2d = torch.meshgrid(r, p, indexing="ij")
    return r2d, p2d


def cylinder_grid(
    r_max: float,
    z_max: float,
    num_r: int,
    num_phi: int,
    num_z: int,
    r_min: float = 0.0,
    z_min: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Returns a 3D cylinder grid.

    Args:
        r_max - Maximum radius.
        z_max - Maximum height.
        num_r - Number of elements along the radial axis.
        num_phi - Number of elements along the angular axis.
        num_z - Number of elements alongthe axial axis
        r_min - Minimum radius.
        z_min - Minimum height.
    """
    _, r = grid1D(r_max, num_r, origin=r_min)
    _, p = grid1D(2.0 * torch.pi, num_phi)
    _, z = grid1D(z_max, num_z, origin=z_min)
    r2d, p2d, z2d = torch.meshgrid(r, p, z, indexing="ij")
    return r2d, p2d, z2d


def spherical_grid(num_theta: int, num_phi: int):
    """Generates 2-Sphere grid.

    Args:
        num_theta - Number of elements along the longitude axis.
        num_phi - Number of elements along the latitude axis.
    """
    _, t = grid1D(2.0 * torch.pi, num_theta)
    _, p = grid1D(2.0 * torch.pi, num_phi)
    t2d, p2d = torch.meshgrid(t, p, indexing="ij")

    return t2d, p2d


def wrap_polar(f: torch.Tensor) -> torch.Tensor:
    """Wraps polar grid, which is useful for plotting purposes.

    Args:
        f - Field polar grid.
    """
    return torch.concatenate([f, torch.array([f[0]])])


def unwrap_polar(f: torch.Tensor) -> torch.Tensor:
    """Unwraps polar grid.

    Args:
        f - Wrapped field polar grid.
    """
    return f[:-1]


def wrap_phi(p2d: torch.Tensor) -> torch.Tensor:
    """Wraps polar grid, which is useful for plotting purposes.

    Args:
        p2d - Phi grid.
    """
    p2d = wrap_polar(p2d)
    p2d[-1] = 2.0 * torch.pi
    return p2d


def unwrap_phi(f: torch.Tensor) -> torch.Tensor:
    """Unwraps polar grid.

    Args:
        f - Wrapped Phi grid.
    """
    return f[:-1]
