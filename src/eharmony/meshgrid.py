import numpy as np
from numpy.polynomial.legendre import leggauss


def meshgrid(b, grid_type="Driscoll-Healy"):
    """
    Create a coordinate grid for the 2-sphere.
    There are various ways to setup a grid on the sphere.

    if grid_type == 'Driscoll-Healy', we follow the grid_type from [4], which is also used in [5]:
    beta_j = pi j / (2 b)     for j = 0, ..., 2b - 1
    alpha_k = pi k / b           for k = 0, ..., 2b - 1

    if grid_type == 'SOFT', we follow the grid_type from [1] and [6]
    beta_j = pi (2 j + 1) / (4 b)   for j = 0, ..., 2b - 1
    alpha_k = pi k / b                for k = 0, ..., 2b - 1

    if grid_type == 'Clenshaw-Curtis', we use the Clenshaw-Curtis grid, as defined in [2] (section 6):
    beta_j = j pi / (2b)     for j = 0, ..., 2b
    alpha_k = k pi / (b + 1)    for k = 0, ..., 2b + 1

    if grid_type == 'Gauss-Legendre', we use the Gauss-Legendre grid, as defined in [2] (section 6) and [7] (eq. 2):
    beta_j = the Gauss-Legendre nodes    for j = 0, ..., b
    alpha_k = k pi / (b + 1),               for k = 0, ..., 2 b + 1

    if grid_type == 'HEALPix', we use the HEALPix grid, see [2] (section 6):

    if grid_type == 'equidistribution', we use the equidistribution grid, as defined in [2] (section 6):

    [1] SOFT: SO(3) Fourier Transforms
    Kostelec, Peter J & Rockmore, Daniel N.

    [2] Fast evaluation of quadrature formulae on the sphere
    Jens Keiner, Daniel Potts

    [3] A Fast Algorithm for Spherical Grid Rotations and its Application to Singular Quadrature
    Zydrunas Gimbutas Shravan Veerapaneni

    [4] Computing Fourier transforms and convolutions on the 2-sphere
    Driscoll, JR & Healy, DM

    [5] Engineering Applications of Noncommutative Harmonic Analysis
    Chrikjian, G.S. & Kyatkin, A.B.

    [6] FFTs for the 2-Sphere – Improvements and Variations
    Healy, D., Rockmore, D., Kostelec, P., Moore, S

    [7] A Fast Algorithm for Spherical Grid Rotations and its Application to Singular Quadrature
    Zydrunas Gimbutas, Shravan Veerapaneni

    :param b: the bandwidth / resolution
    :return: a meshgrid on S^2
    """
    return np.meshgrid(*linspace(b, grid_type), indexing="ij")


def linspace(b, grid_type="Driscoll-Healy"):
    if grid_type == "Driscoll-Healy":
        beta = np.arange(2 * b) * np.pi / (2.0 * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == "SOFT":
        beta = np.pi * (2 * np.arange(2 * b) + 1) / (4.0 * b)
        alpha = np.arange(2 * b) * np.pi / b
    elif grid_type == "Clenshaw-Curtis":
        # beta = np.arange(2 * b + 1) * np.pi / (2 * b)
        # alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
        # Must use np.linspace to prevent numerical errors that cause beta > pi
        beta = np.linspace(0, np.pi, 2 * b + 1)
        alpha = np.linspace(0, 2 * np.pi, 2 * b + 2, endpoint=False)
    elif grid_type == "Gauss-Legendre":
        x, _ = leggauss(b + 1)  # TODO: leggauss docs state that this may not be only stable for orders > 100
        beta = np.arccos(x)
        alpha = np.arange(2 * b + 2) * np.pi / (b + 1)
    elif grid_type == "HEALPix":
        # TODO: implement this here so that we don't need the dependency on healpy / healpix_compat
        from healpix_compat import healpy_sphere_meshgrid

        return healpy_sphere_meshgrid(b)
    elif grid_type == "equidistribution":
        raise NotImplementedError("Not implemented yet; see Fast evaluation of quadrature formulae on the sphere.")
    else:
        raise ValueError("Unknown grid_type:" + grid_type)
    return beta, alpha
