import numpy as np

#===============================================================================
# Tesseral Harmonics
#https://github.com/cp2k/cp2k/blob/master/src/common/spherical_harmonics.F
# http://www2.cpfs.mpg.de/~rotter/homepage_mcphase/manual/node131.html
def Y_l(r_vec, l):
    """Real Spherical Harmonics"""

    in_shape = r_vec.shape
    assert in_shape[-1] == 3
    out_shape = in_shape[:-1] + (2*l+1,)
    result = np.zeros(out_shape)

    r = np.sqrt(np.sum(r_vec * r_vec, axis=-1))
    x = r_vec[..., 0] / r
    y = r_vec[..., 1] / r
    z = r_vec[..., 2] / r

    if l == 0:
        result[..., 0] = np.sqrt(1.0 / (4.0 * np.pi))
    elif l == 1:
        pf = np.sqrt(3.0 / (4.0 * np.pi))
        result[..., 0] = pf * y # m=-1
        result[..., 1] = pf * z # m=0
        result[..., 2] = pf * x # m=+1
        return result
    elif l == 2:
        # m = -2
        pf = np.sqrt(15.0 / (16.0 * np.pi))
        result[..., 0] = pf * 2.0 * x * y
        # m = -1
        pf = np.sqrt(15.0 / (4.0 * np.pi))
        result[..., 1] = pf * z * y
        # m = 0
        pf = np.sqrt(5.0 / (16.0 * np.pi))
        result[..., 2] = pf * (3.0 * z**2 - 1.0)
        # m = 1
        pf = np.sqrt(15.0 / (4.0 * np.pi))
        result[..., 3] = pf * z * x
        # m = 2
        pf = np.sqrt(15.0 / (16.0 * np.pi))
        result[..., 4] = pf * (x**2 - y**2)
    else:
        raise Exception("Not implemented")

    return result

#===============================================================================
#WARNING: Uses a different definition than above, which leads to other signs.
def Y_l_sympy(r_vec, l):
    from sympy.functions.special import spherical_harmonics
    import sympy

    x = r_vec[0]
    y = r_vec[1]
    z = r_vec[2]
    r = sympy.sqrt(x**2+y**2+z**2)
    theta = sympy.acos(z/r)
    phi = sympy.atan2(y, x)
    result = np.zeros(2*l+1)
    for m in range(-l, l+1):
        result[m+l] = sympy.re(spherical_harmonics.Znm(l, m, theta, phi)).evalf()
        condon_shortley_phase = (-1.0)**m
        result[m+l] *= condon_shortley_phase
    return result

#===============================================================================
#http://qutip.org/docs/3.1.0/modules/qutip/utilities.html
def clebsch(j1, j2, j3, m1, m2, m3):
    """Calculates the Clebsch-Gordon coefficient
    for coupling (j1,m1) and (j2,m2) to give (j3,m3).

    Parameters
    ----------
    j1 : float
        Total angular momentum 1.

    j2 : float
        Total angular momentum 2.

    j3 : float
        Total angular momentum 3.

    m1 : float
        z-component of angular momentum 1.

    m2 : float
        z-component of angular momentum 2.

    m3 : float
        z-component of angular momentum 3.

    Returns
    -------
    cg_coeff : float
        Requested Clebsch-Gordan coefficient.

    """
    from scipy.special import factorial

    if m3 != m1 + m2:
        return 0
    vmin = int(np.max([-j1 + j2 + m3, -j1 + m1, 0]))
    vmax = int(np.min([j2 + j3 + m1, j3 - j1 + j2, j3 + m3]))

    C = np.sqrt((2.0 * j3 + 1.0) * factorial(j3 + j1 - j2) *
                factorial(j3 - j1 + j2) * factorial(j1 + j2 - j3) *
                factorial(j3 + m3) * factorial(j3 - m3) /
                (factorial(j1 + j2 + j3 + 1) *
                factorial(j1 - m1) * factorial(j1 + m1) *
                factorial(j2 - m2) * factorial(j2 + m2)))
    S = 0
    for v in range(vmin, vmax + 1):
        S += (-1.0) ** (v + j2 + m2) / factorial(v) * \
            factorial(j2 + j3 + m1 - v) * factorial(j1 - m1 + v) / \
            factorial(j3 - j1 + j2 - v) / factorial(j3 + m3 - v) / \
            factorial(v + j1 - j2 - m3)
    C = C * S
    return C

#===============================================================================
cg_cache = dict()

def get_clebsch_gordan_coefficients_sympy(li, lj, lo):
    from sympy.physics.quantum.cg import CG
    #TODO maybe rename lo -> lk ?
    global cg_cache
    key = (li, lj, lo)
    if key not in cg_cache:
        assert abs(li-lj) <= lo <= abs(li+lj)
        coeffs = np.zeros(shape=(2*li+1, 2*lj+1, 2*lo+1))
        for mi in range(-li, li+1):
            for mj in range(-lj, lj+1):
                for mo in range(-lo, lo+1):
                    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
                    cg = CG(li, mi, lj, mj, lo, mo).doit()
                    coeffs[mi+li, mj+lj, mo+lo] = cg
        cg_cache[key] = coeffs
    return cg_cache[key]

#===============================================================================
cg_cache2 = dict()

def get_clebsch_gordan_coefficients_sympy(li, lj, lo):
    from sympy.physics.quantum.cg import CG
    #TODO maybe rename lo -> lk ?
    global cg_cache2
    key = (li, lj, lo)
    if key not in cg_cache2:
        assert abs(li-lj) <= lo <= abs(li+lj)
        coeffs = np.zeros(shape=(2*li+1, 2*lj+1, 2*lo+1))
        for mi in range(-li, li+1):
            for mj in range(-lj, lj+1):
                for mo in range(-lo, lo+1):
                    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
                    cg = CG(li, mi, lj, mj, lo, mo).doit()
                    coeffs[mi+li, mj+lj, mo+lo] = cg
        cg_cache2[key] = coeffs
    return cg_cache2[key]
    
##===============================================================================
#cg_cache2 = dict()
#
## Formular 1.18 from https://sahussaintu.files.wordpress.com/2014/03/spherical_harmonics.pdf
#
#def get_clebsch_gordan_coefficients_new(li, lj, lo):
#    from sympy.physics.quantum.cg import Wigner3j, CG
#    #TODO maybe rename lo -> lk ?
#    global cg_cache2
#    key = (li, lj, lo)
#    if key not in cg_cache2:
#        assert abs(li-lj) <= lo <= abs(li+lj)
#
#        #cg0 = CG(li, 0, lj, 0, lo, 0).doit() 
#        #numerator = (2*li+1) * (2*lj+1)
#        #denominator = 4 * np.pi * (2*lo+1)
#        #pre_factor = cg0 * np.sqrt(numerator / denominator)
#
#        coeffs = np.zeros(shape=(2*li+1, 2*lj+1, 2*lo+1))
#        for mi in range(-li, li+1):
#            for mj in range(-lj, lj+1):
#                for mo in range(-lo, lo+1):
#                    # https://docs.sympy.org/latest/modules/physics/quantum/cg.html
#                    #cg = Wigner3j(li, mi, lj, mj, lo, mo).doit()
#                    cg = CG(lo, mo, li, mi, lj, mj).doit()
#                    coeffs[mi+li, mj+lj, mo+lo] = cg
#        cg_cache2[key] = coeffs
#    return cg_cache2[key]
#

#===============================================================================
# This is most likely wrong because with p functions use yzx instead of xyz.
def get_clebsch_gordan_coefficients_table(li, lj, lo):
    assert abs(li-lj) <= lo <= abs(li+lj)
    coeffs = np.zeros(shape=(2*li+1, 2*lj+1, 2*lo+1))
    if li==0 and lj==0 and lo==0:
        # copy scalar
        coeffs[0,0,0] = 1.0
    elif li==1 and lj==1 and lo==0:
        # vector dot product
        coeffs[0,0,0] = 1.0
        coeffs[1,1,0] = 1.0
        coeffs[2,2,0] = 1.0
    elif li==0 and lj==1 and lo==1:
        # multiply vector by scalar
        coeffs[0,0,0] = 1.0
        coeffs[0,1,1] = 1.0
        coeffs[0,2,2] = 1.0
    elif li==1 and lj==0 and lo==1:
        # multiply vector by scalar
        coeffs[0,0,0] = 1.0
        coeffs[1,0,1] = 1.0
        coeffs[2,0,2] = 1.0
    elif li==1 and lj==1 and lo==1:
        # vector cross product
        #TODO: note the order is (y, z, x), but does it really matter?
        coeffs[0,1,2] = 1.0
        coeffs[1,2,0] = 1.0
        coeffs[2,0,1] = 1.0
        coeffs[2,1,0] = -1.0
        coeffs[0,2,1] = -1.0
        coeffs[1,0,2] = -1.0
    return coeffs


# https://sahussaintu.files.wordpress.com/2014/03/spherical_harmonics.pdf