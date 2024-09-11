""" spherical_harmonics.py  """

import numpy as np
import torch
from torch import nn

from lie_learn.representations.SO3 import spherical_harmonics
from lie_learn.spaces import S2

from torch_harmonics.harmonic_function import HarmonicFunction


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

        self.L = L
        self.num_lat = num_lat
        self.num_lon = num_lon

        #Y = self.generate_basis_fns().permute(0, 2, 1)
        Y = self.generate_basis_fns()
        self.register_buffer("Y", Y, persistent=False)

    def generate_basis_fns(self, coords: torch.Tensor = None) -> torch.Tensor:
        if coords is None:
            theta, phi = np.meshgrid(
                np.linspace(0, 2 * np.pi, self.num_lon),
                np.linspace(0, np.pi, self.num_lat),
            )
            # beta, alpha = S2.meshgrid(self.num_lat)
        else:
            theta = coords[:, 0].unsqueeze(1)
            phi = coords[:, 1].unsqueeze(1)

        irreps = np.arange(self.L + 1)
        ls = [[ls] * (2 * ls + 1) for ls in irreps]
        ls = np.array(
            [ll for sublist in ls for ll in sublist]
        )  # 0, 1, 1, 1, 2, 2, 2, 2, 2, ...

        ms = [list(range(-ls, ls + 1)) for ls in irreps]
        ms = np.array(
            [mm for sublist in ms for mm in sublist]
        )  # 0, -1, 0, 1, -2, -1, 0, 1, 2, ...

        Y = spherical_harmonics.sh(
            ls[:, None, None],
            ms[:, None, None],
            phi[None, :, :],
            theta[None, :, :],
            field="real",
            normalization="quantum",
            condon_shortley=True,
        )

        return torch.tensor(Y)

    def forward(self, w: torch.Tensor, coords: torch.Tensor = None) -> torch.Tensor:
        B = w.size(0)

        if coords is not None:
            Y = self.generate_basis_fns(coords.cpu()).permute(1, 0, 2).unsqueeze(3)
            Y = Y.to(w.device)
        else:
            Y = self.Y.unsqueeze(0)
            Y = Y.expand(B, Y.size(1), Y.size(2), Y.size(3))

        out = torch.einsum("bn,bncd->bncd", w, Y).sum(1)

        return out
