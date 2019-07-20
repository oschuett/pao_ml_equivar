#!/usr/bin/python3


"""
Compute the Clebsch-Gordan coefficients of SO(3) numerically.
These coefficients specify how a product of Wigner D functions can be re-expressed as a linear combination of
Wigner D functions.

Wikipedia gives the formula:
D^l_mn(a,b,c) D^l'_m'n'(a,b,c) = sum_{L=|l-l'|^(l+l') sum_M=-L^L sum_N=-L^L <lml'm'|LM> <lnl'n'|LN> D^L_MN(a,b,c)
where D are the Wigner D functions and <....|..> are CG coefficients.

For our computations related to the representations of SO(3) on the projective plane, we are most interested in
the case l=l'=1
D^1_mn(a,b,c) D^1_m'n'(a,b,c) = sum_{L=0^2 sum_M=-L^L sum_N=-L^L <lml'm'|LM> <lnl'n'|LN> D^L_MN(a,b,c)

Since we typically employ the Pinchon-Hoggan basis of real spherical harmonics, we cannot use the formulas for the
CG coefficients that are given in the standard references. We could re-do the analysis to find these coefficients
for the real basis, but here we employ a simple numerical approach.

We view the products coefficients <1m1m'|LM><1n1n'|LN> as unknowns C(m, m', n, n', L, M, N)
We can randomly sample a large number of euler angles g_i = (alpha_i, beta_i, gamma_i), and solve the linear
system for C.

Looking at this system, it appears that the coefficients are ratios of integers or (square?) roots, such as
1/2, 1/3, 1/(2 sqrt(3)), etc.
This is reminiscent of the numbers Pinchon & Hoggan encounter in their J matrix, so that is something to look into.

I've saved dtThe results of the CG computation for l=1 to clebsch_gordan_l1.npy
"""

import numpy as np
import scipy.linalg


def compute_CG_3D(m1, n1, m2, n2, N=1000):

    l = 1

    l_min = 0
    l_max = 2
    num_coefs = sum([(2 * j + 1) ** 2 for j in range(l_min, l_max + 1)])

    g = np.random.rand(3, N) * np.array([[2 * np.pi], [np.pi], [2 * np.pi]])

    D1 = SO3_irrep(g, 1)[l + m1, l + n1, :]
    D2 = SO3_irrep(g, 1)[l + m2, l + n2, :]
    target = D1 * D2

    A = np.zeros((N, num_coefs))
    for i in range(N):
        Ds = np.concatenate([SO3_irrep(g[:, i][:, None], j).flatten() for j in range(l_min, l_max + 1)])
        A[i, :] = Ds

    return A, target, np.linalg.pinv(A).dot(target)


def compute_CG_matrix(N=1000):

    CG = np.zeros((1 + 3 * 3 + 5 * 5, 3, 3, 3, 3))
    l = 1
    for m1 in range(-l, l + 1):
        for n1 in range(-l, l + 1):
            for m2 in range(-l, l + 1):
                for n2 in range(-l, l + 1):
                    print(m1, n1, m2, n2)
                    _, _, w = compute_CG_3D(m1, n1, m2, n2, N)
                    CG[:, l + m1, l + n1, l + m2, l + n2] = w

    return CG



#===============================================================================


# This code is not very optimized,
# and can never become very efficient because it cannot exploit the sparsity of the J matrix.

import os

import numpy as np
from scipy.linalg import block_diag

# Load the J-matrices, which are stored in the same folder as this file
Jd = np.load(os.path.join(os.path.dirname(__file__), 'J_dense_0-278.npy'), encoding='latin1', allow_pickle=True)


def SO3_irreps(g, irreps):
    global Jd

    # First, compute sinusoids at all required frequencies, i.e.
    # cos(n x) for n=0, ..., max(irreps)
    # sin(n x) for n=-max(irreps), ..., max(irreps)
    # where x ranges over the three parameters of SO(3).

    # In theory, it may be faster to evaluate cos(x) once and then use
    # Chebyshev polynomials to obtain cos(n*x), but in practice this appears
    # to be slower than just evaluating cos(n*x).
    dim = np.sum(2 * np.array(irreps) + 1)
    T = np.empty((dim, dim, g.shape[1]))
    for i in range(g.shape[1]):
        T[:, :, i] = block_diag(*[rot_mat(g[0, i], g[1, i], g[2, i], l, Jd[l]) for l in irreps])
    return T


def SO3_irrep(g, l):
    global Jd
    g = np.atleast_2d(g)
    T = np.empty((2 * l + 1, 2 * l + 1, g.shape[1]))
    for i in range(g.shape[1]):
        T[:, :, i] = rot_mat(g[0, i], g[1, i], g[2, i], l, Jd[l])
    return T  # np.squeeze(T)


def z_rot_mat(angle, l):
    """
    Create the matrix representation of a z-axis rotation by the given angle,
    in the irrep l of dimension 2 * l + 1, in the basis of real centered
    spherical harmonics (RC basis in rep_bases.py).

    Note: this function is easy to use, but inefficient: only the entries
    on the diagonal and anti-diagonal are non-zero, so explicitly constructing
    this matrix is unnecessary.
    """
    M = np.zeros((2 * l + 1, 2 * l + 1))
    inds = np.arange(0, 2 * l + 1, 1)
    reversed_inds = np.arange(2 * l, -1, -1)
    frequencies = np.arange(l, -l - 1, -1)
    M[inds, reversed_inds] = np.sin(frequencies * angle)
    M[inds, inds] = np.cos(frequencies * angle)
    return M


def rot_mat(alpha, beta, gamma, l, J):
    """
    Compute the representation matrix of a rotation by ZYZ-Euler
    angles (alpha, beta, gamma) in representation l in the basis
    of real spherical harmonics.

    The result is the same as the wignerD_mat function by Johann Goetz,
    when the sign of alpha and gamma is flipped.
    """
    Xa = z_rot_mat(alpha, l)
    Xb = z_rot_mat(beta, l)
    Xc = z_rot_mat(gamma, l)
    return Xa.dot(J).dot(Xb).dot(J).dot(Xc)


def derivative_z_rot_mat(angle, l):
    M = np.zeros((2 * l + 1, 2 * l + 1))
    inds = np.arange(0, 2 * l + 1, 1)
    reversed_inds = np.arange(2 * l, -1, -1)
    frequencies = np.arange(l, -l - 1, -1)
    M[inds, reversed_inds] = np.cos(frequencies * angle) * frequencies
    M[inds, inds] = -np.sin(frequencies * angle) * frequencies
    return M


def derivative_rot_mat(alpha, beta, gamma, l, J):
    Xa = z_rot_mat(alpha, l)
    Xb = z_rot_mat(beta, l)
    Xc = z_rot_mat(gamma, l)
    dXa_da = derivative_z_rot_mat(alpha, l)
    dXb_db = derivative_z_rot_mat(beta, l)
    dXc_dc = derivative_z_rot_mat(gamma, l)

    dDda = dXa_da.dot(J).dot(Xb).dot(J).dot(Xc)
    dDdb = Xa.dot(J).dot(dXb_db).dot(J).dot(Xc)
    dDdc = Xa.dot(J).dot(Xb).dot(J).dot(dXc_dc)
    return dDda, dDdb, dDdc

#===============================================================================
def main():
    _, _, w = compute_CG_3D(m1=0, n1=1, m2=0, n2=0, N=1000)
    print(w)

#===============================================================================
def main_orig():
    CG = compute_CG_matrix(1000)
    CG_exact = np.zeros_like(CG)

    uniques = [0., 1. / 2., -1. / 2.,
               1. / 3., -1. / 3.,
               1. / 6., 2. / 3.,
               1. / (2 * np.sqrt(3)), -1. / (2 * np.sqrt(3)),
               1. / np.sqrt(3), -1. / np.sqrt(3)]
    print('Hypothetical exact uniques:')
    print(np.sort(uniques))
    print('Numerically obtained uniques (rounded to 5 decimals)')
    print(np.unique(np.round(CG, 5)))
    for value in uniques:
        inds = np.nonzero(np.isclose(CG, value))
        CG_exact[inds] = value

    print('Absolute error between exact and numerical:', np.sum(np.abs(CG_exact - CG)))


#===============================================================================
if __name__ == '__main__':
    main()
