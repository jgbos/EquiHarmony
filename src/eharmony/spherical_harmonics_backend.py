"""Compatibility spherical harmonics backend.

This module provides a stable spherical harmonics API across SciPy versions.
It preserves EquiHarmony's expected conventions:
- theta: colatitude in [0, pi]
- phi: azimuth in [0, 2pi)
- field: real or complex
- normalization: quantum, seismology, geodesy, unnormalized, nfft
"""

import numpy as np
from scipy.special import factorial

try:
    # SciPy newer API
    from scipy.special import sph_harm_y as _scipy_sph_harm

    def _sph_harm_compat(m, l, phi, theta):
        # sph_harm_y signature: (n, m, theta, phi)
        return _scipy_sph_harm(l, m, theta, phi)

except ImportError:
    # SciPy older API
    from scipy.special import sph_harm as _scipy_sph_harm

    def _sph_harm_compat(m, l, phi, theta):
        # sph_harm signature: (m, n, theta, phi), where theta is azimuth and phi is colatitude.
        return _scipy_sph_harm(m, l, phi, theta)


def csh(l, m, theta, phi, normalization="quantum", condon_shortley=True):
    l, m, theta, phi = np.broadcast_arrays(l, m, theta, phi)

    if normalization == "quantum":
        y = ((-1.0) ** m) * _sph_harm_compat(m, l, phi, theta)
    elif normalization == "seismology":
        y = _sph_harm_compat(m, l, phi, theta)
    elif normalization == "geodesy":
        y = np.sqrt(4 * np.pi) * _sph_harm_compat(m, l, phi, theta)
    elif normalization == "unnormalized":
        y = _sph_harm_compat(m, l, phi, theta) / np.sqrt(
            (2 * l + 1) * factorial(l - m) / (4 * np.pi * factorial(l + m))
        )
    elif normalization == "nfft":
        y = _sph_harm_compat(m, l, phi, theta) / np.sqrt((2 * l + 1) / (4 * np.pi))
    else:
        raise ValueError(f"Unknown normalization convention: {normalization}")

    if condon_shortley:
        return y

    return y * ((-1.0) ** (m * (m > 0)))


def rsh(l, m, theta, phi, normalization="quantum", condon_shortley=True):
    l, m, theta, phi = np.broadcast_arrays(l, m, theta, phi)

    a = csh(l=l, m=m, theta=theta, phi=phi, normalization=normalization, condon_shortley=True)
    b = csh(l=l, m=-m, theta=theta, phi=phi, normalization=normalization, condon_shortley=True)

    y = (
        (m > 0) * np.array((b + ((-1.0) ** m) * a).real / np.sqrt(2.0))
        + (m < 0) * np.array((1j * a - 1j * ((-1.0) ** (-m)) * b).real / np.sqrt(2.0))
        + (m == 0) * np.array(a.real)
    )

    if condon_shortley:
        return y

    return y * ((-1.0) ** (m * (m > 0)))


def sh(l, m, theta, phi, field="real", normalization="quantum", condon_shortley=True):
    if field == "real":
        return rsh(l, m, theta, phi, normalization=normalization, condon_shortley=condon_shortley)
    if field == "complex":
        return csh(l, m, theta, phi, normalization=normalization, condon_shortley=condon_shortley)

    raise ValueError(f"Unknown field: {field}")
