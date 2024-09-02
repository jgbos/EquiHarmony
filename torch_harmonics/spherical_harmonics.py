""" spherical_harmonics.py  """

import numpy as np
import torch
from torch import nn
import torch.fft

from lie_learn.representations.SO3 import spherical_harmonics
from lie_learn.spaces import S2

from torch_harmonics.harmonic_function import HarmonicFunction


def lobatto_weights(n, a=-1.0, b=1.0, tol=1e-16, maxiter=100):
    r"""
    Helper routine which returns the Legendre-Gauss-Lobatto nodes and weights
    on the interval [a, b]
    """

    wlg = np.zeros((n,))
    tlg = np.zeros((n,))
    tmp = np.zeros((n,))

    # Vandermonde Matrix
    vdm = np.zeros((n, n))

    # initialize Chebyshev nodes as first guess
    for i in range(n):
        tlg[i] = -np.cos(np.pi * i / (n - 1))

    tmp = 2.0

    for i in range(maxiter):
        tmp = tlg

        vdm[:, 0] = 1.0
        vdm[:, 1] = tlg

        for k in range(2, n):
            vdm[:, k] = (
                (2 * k - 1) * tlg * vdm[:, k - 1] - (k - 1) * vdm[:, k - 2]
            ) / k

        tlg = tmp - (tlg * vdm[:, n - 1] - vdm[:, n - 2]) / (n * vdm[:, n - 1])

        if max(abs(tlg - tmp).flatten()) < tol:
            break

    wlg = 2.0 / ((n * (n - 1)) * (vdm[:, n - 1] ** 2))

    # rescale
    tlg = (b - a) * 0.5 * tlg + (b + a) * 0.5
    wlg = wlg * (b - a) * 0.5

    return tlg, wlg


def legpoly(mmax, lmax, x, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(x) at the positions specified by x.
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    # compute the tensor P^m_n:
    nmax = max(mmax, lmax)
    vdm = np.zeros((nmax, nmax, len(x)), dtype=np.float64)

    norm_factor = 1.0 if norm == "ortho" else np.sqrt(4 * np.pi)
    norm_factor = 1.0 / norm_factor if inverse else norm_factor

    # initial values to start the recursion
    vdm[0, 0, :] = norm_factor / np.sqrt(4 * np.pi)

    # fill the diagonal and the lower diagonal
    for l in range(1, nmax):
        vdm[l - 1, l, :] = np.sqrt(2 * l + 1) * x * vdm[l - 1, l - 1, :]
        vdm[l, l, :] = (
            np.sqrt((2 * l + 1) * (1 + x) * (1 - x) / 2 / l) * vdm[l - 1, l - 1, :]
        )

    # fill the remaining values on the upper triangle and multiply b
    for l in range(2, nmax):
        for m in range(0, l - 1):
            vdm[m, l, :] = (
                x
                * np.sqrt((2 * l - 1) / (l - m) * (2 * l + 1) / (l + m))
                * vdm[m, l - 1, :]
                - np.sqrt(
                    (l + m - 1)
                    / (l - m)
                    * (2 * l + 1)
                    / (2 * l - 3)
                    * (l - m - 1)
                    / (l + m)
                )
                * vdm[m, l - 2, :]
            )

    if norm == "schmidt":
        for l in range(0, nmax):
            if inverse:
                vdm[:, l, :] = vdm[:, l, :] * np.sqrt(2 * l + 1)
            else:
                vdm[:, l, :] = vdm[:, l, :] / np.sqrt(2 * l + 1)

    vdm = vdm[:mmax, :lmax]

    if csphase:
        for m in range(1, mmax, 2):
            vdm[m] *= -1

    return vdm


def _precompute_legpoly(mmax, lmax, t, norm="ortho", inverse=False, csphase=True):
    r"""
    Computes the values of (-1)^m c^l_m P^l_m(\cos \theta) at the positions specified by t (theta).
    The resulting tensor has shape (mmax, lmax, len(x)). The Condon-Shortley Phase (-1)^m
    can be turned off optionally.

    method of computation follows
    [1] Schaeffer, N.; Efficient spherical harmonic transforms aimed at pseudospectral numerical simulations, G3: Geochemistry, Geophysics, Geosystems.
    [2] Rapp, R.H.; A Fortran Program for the Computation of Gravimetric Quantities from High Degree Spherical Harmonic Expansions, Ohio State University Columbus; report; 1982;
        https://apps.dtic.mil/sti/citations/ADA123406
    [3] Schrama, E.; Orbit integration based upon interpolated gravitational gradients
    """

    return legpoly(mmax, lmax, np.cos(t), norm=norm, inverse=inverse, csphase=csphase)


class SphericalHarmonics(HarmonicFunction):
    """
    Torch module for computing a spherical function using Fourier coefficients and spherical
    harmonics basis functions. Pre-computes basis functions for a grid of values which can
    be used for faster evaluation.

    Args:
       L - Maximum angular frequency.
       num_lat - Number of elements on the latitudinal axis.
       num_lon - Number of elements on the longitudinal axis.
    """

    def __init__(self, L: int, M: int, num_lat: int = 360, num_lon: int = 360):
        super().__init__()

        self.num_lat = num_lat
        self.num_lon = num_lon
        self.mmax = M
        self.lmax = L

        cost, _ = lobatto_weights(self.num_lat, -1, 1)
        t = np.flip(np.arccos(cost))
        pct = _precompute_legpoly(self.mmax, self.lmax, t)
        pct = torch.from_numpy(pct)

        self.register_buffer("pct", pct, persistent=False)

    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        B = w.size(0)

        # w = torch.view_as_real(w)

        rl = torch.einsum("...lm, mlk->...km", w[..., 0], self.pct.to(w.dtype))
        im = torch.einsum("...lm, mlk->...km", w[..., 1], self.pct.to(w.dtype))
        xs = torch.stack((rl, im), -1)

        # apply the inverse (real) FFT
        x = torch.view_as_complex(xs)
        out = torch.fft.irfft(x, n=self.num_lon, dim=-1, norm="forward")

        return out
