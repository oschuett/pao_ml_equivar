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
def clebsch_gordan_coefficients(l1, l2, l3):
    assert abs(l1-l2) <= l3 <= abs(l1+l2)
    coeffs = np.zeros(shape=(2*l1+1, 2*l2+1, 2*l3+1))
    if l1==0 and l2==0 and l3==0:
        # copy scalar
        coeffs[0,0,0] = 1.0
    elif l1==1 and l2==1 and l3==0:
        # vector dot product
        coeffs[0,0,0] = 1.0
        coeffs[1,1,0] = 1.0
        coeffs[2,2,0] = 1.0
    elif l1==0 and l2==1 and l3==1:
        # multiply vector by scalar
        coeffs[0,0,0] = 1.0
        coeffs[0,1,1] = 1.0
        coeffs[0,2,2] = 1.0
    elif l1==1 and l2==0 and l3==1:
        # multiply vector by scalar
        coeffs[0,0,0] = 1.0
        coeffs[1,0,1] = 1.0
        coeffs[2,0,2] = 1.0
    elif l1==1 and l2==1 and l3==1:
        # vector cross product
        coeffs[0,1,2] = 1.0
        coeffs[1,2,0] = 1.0
        coeffs[2,0,1] = 1.0
        coeffs[2,1,0] = -1.0
        coeffs[0,2,1] = -1.0
        coeffs[1,0,2] = -1.0
    elif l1==0 and l2==2 and l3==2:
        coeffs[0,0,0] = 1.0
        coeffs[0,1,1] = 1.0
        coeffs[0,2,2] = 1.0
        coeffs[0,3,3] = 1.0
        coeffs[0,4,4] = 1.0
    elif l1==0 and l2==2 and l3==2:
        # multiply vector by scalar
        coeffs[0,0,0] = 1.0
        coeffs[0,1,1] = 1.0
        coeffs[0,2,2] = 1.0
        coeffs[0,3,3] = 1.0
        coeffs[0,4,4] = 1.0
    elif l1==1 and l2==1 and l3==2:
        c = 0.5 * np.sqrt(3)
        coeffs[0,0,2] = -0.5
        coeffs[0,0,4] = -c
        coeffs[0,1,1] = c
        coeffs[0,2,0] = c
        coeffs[1,0,1] = c
        coeffs[1,1,2] = 1.0
        coeffs[1,2,3] = c
        coeffs[2,0,0] = c
        coeffs[2,1,3] = c
        coeffs[2,2,2] = -0.5
        coeffs[2,2,4] = c
    elif l1==1 and l2==2 and l3==1:
        c = 0.5 * np.sqrt(3)
        coeffs[0,0,2] = -c
        coeffs[0,1,1] = -c
        coeffs[0,2,0] = 0.5
        coeffs[0,4,0] = c
        coeffs[1,1,0] = -c
        coeffs[1,2,1] = -1.0
        coeffs[1,3,2] = -c
        coeffs[2,0,0] = -c
        coeffs[2,2,2] = 0.5
        coeffs[2,3,1] = -c
        coeffs[2,4,2] = -c
    elif l1==1 and l2==2 and l3==2:
        c = 0.5 * np.sqrt(3)
        coeffs[0,0,1] = 0.5
        coeffs[0,1,0] = -0.5
        coeffs[0,2,3] = -c
        coeffs[0,3,2] = c
        coeffs[0,3,4] = -0.5
        coeffs[0,4,3] = 0.5
        coeffs[1,0,4] = 1.0
        coeffs[1,1,3] = 0.5
        coeffs[1,3,1] = -0.5
        coeffs[1,4,0] = -1.0
        coeffs[2,0,3] = -0.5
        coeffs[2,1,2] = -c
        coeffs[2,1,4] = -0.5
        coeffs[2,2,1] = c
        coeffs[2,3,0] = 0.5
        coeffs[2,4,1] = 0.5
    elif l1==2 and l2==0 and l3==2:
        # multiply vector by scalar
        coeffs[0,0,0] = 1.0
        coeffs[1,0,1] = 1.0
        coeffs[2,0,2] = 1.0
        coeffs[3,0,3] = 1.0
        coeffs[4,0,4] = 1.0
    elif l1==2 and l2==1 and l3==1:
        c = 0.5 * np.sqrt(3)
        coeffs[0,0,2] = -c
        coeffs[0,2,0] = -c
        coeffs[1,0,1] = -c
        coeffs[1,1,0] = -c
        coeffs[2,0,0] = 0.5
        coeffs[2,1,1] = -1.0
        coeffs[2,2,2] = 0.5
        coeffs[3,1,2] = -c
        coeffs[3,2,1] = -c
        coeffs[4,0,0] = c
        coeffs[4,2,2] = -c
    elif l1==2 and l2==2 and l3==0:
        # vector dot product
        coeffs[0,0,0] = 1.0
        coeffs[1,1,0] = 1.0
        coeffs[2,2,0] = 1.0
        coeffs[3,3,0] = 1.0
        coeffs[4,4,0] = 1.0
    elif l1==2 and l2==1 and l3==2:
        c = 0.5 * np.sqrt(3)
        coeffs[0,0,1] = 0.5
        coeffs[0,1,4] = 1.0
        coeffs[0,2,3] = -0.5
        coeffs[1,0,0] = -0.5
        coeffs[1,1,3] = 0.5
        coeffs[1,2,2] = -c
        coeffs[1,2,4] = -0.5
        coeffs[2,0,3] = -c
        coeffs[2,2,1] = c
        coeffs[3,0,2] = c
        coeffs[3,0,4] = -0.5
        coeffs[3,1,1] = -0.5
        coeffs[3,2,0] = 0.5
        coeffs[4,0,3] = 0.5
        coeffs[4,1,0] = -1.0
        coeffs[4,2,1] = 0.5
    elif l1==2 and l2==2 and l3==1:
        c = 0.5 * np.sqrt(3)
        coeffs[0,1,0] = 0.5
        coeffs[0,3,2] = -0.5
        coeffs[0,4,1] = 1.0
        coeffs[1,0,0] = -0.5
        coeffs[1,2,2] = -c
        coeffs[1,3,1] = 0.5
        coeffs[1,4,2] = -0.5
        coeffs[2,1,2] = c
        coeffs[2,3,0] = -c
        coeffs[3,0,2] = 0.5
        coeffs[3,1,1] = -0.5
        coeffs[3,2,0] = c
        coeffs[3,4,0] = -0.5
        coeffs[4,0,1] = -1.0
        coeffs[4,1,2] = 0.5
        coeffs[4,3,0] = 0.5
    elif l1==2 and l2==2 and l3==2:
        c = 0.5 * np.sqrt(3)
        coeffs[0,0,2] = 1.0
        coeffs[0,1,3] = -c
        coeffs[0,2,0] = 1.0
        coeffs[0,3,1] = -c
        coeffs[1,0,3] = -c
        coeffs[1,1,2] = -0.5
        coeffs[1,1,4] = c
        coeffs[1,2,1] = -0.5
        coeffs[1,3,0] = -c
        coeffs[1,4,1] = c
        coeffs[2,0,0] = 1.0
        coeffs[2,1,1] = -0.5
        coeffs[2,2,2] = -1.0
        coeffs[2,3,3] = -0.5
        coeffs[2,4,4] = 1.0
        coeffs[3,0,1] = -c
        coeffs[3,1,0] = -c
        coeffs[3,2,3] = -0.5
        coeffs[3,3,2] = -0.5
        coeffs[3,3,4] = -c
        coeffs[3,4,3] = -c
        coeffs[4,1,1] = c
        coeffs[4,2,4] = 1.0
        coeffs[4,3,3] = -c
        coeffs[4,4,2] = 1.0
    else:
        raise Exception("Not implemented")
    return coeffs


#===============================================================================
def test_cg_coeffs(lmax=2):
    combis = []
    for l1 in range(lmax+1):
        for l2 in range(lmax+1):
            l3_min = abs(l1-l2)
            l3_max = min(l1+l2, lmax)
            for l3 in range(l3_min, l3_max+1):
                combis.append([l1,l2,l3])

    N = 10 # number of samples
    for l1,l2,l3 in combis:
        #print(l1,l2,l3)
        cg_coeffs = clebsch_gordan_coefficients(l1,l2,l3)
        for i in range(N):
            g = np.random.rand(3) * 2 * np.pi
            D1 = wigner_d_num(g,l1)
            D2 = wigner_d_num(g,l2)
            D3 = wigner_d_num(g,l3)
            lhs = np.einsum("ai,bj,ijc->abc", D1, D2, cg_coeffs)
            rhs = np.einsum("abi,ic->abc", cg_coeffs, D3)
            residual = lhs - rhs
            r = np.max(np.abs(residual))
            if r > 1e-13:
                print("ERROR {},{},{}: {}".format(l1,l2,l3,r))


#===============================================================================
def rot_mat(g):
    from scipy.spatial.transform import Rotation
    # g = [alpha, beta, gamma]
    return  Rotation.from_euler('zyx', g).as_dcm()


#===============================================================================
def wigner_d_num(g, l):
    """Determine Wigner D-matrix numerically."""
    # g: euler angles
    # pick random directions
    N = 2*2*(l+1)  # use twice as many too be sure
    r1 = np.random.rand(N, 3)

    # build rotation matrix
    R = rot_mat(g)

    # rotate random direction
    r2 = np.einsum("ki,ij->kj", r1, R)

    y1 = Y_l(r1, l=l)
    y2 = Y_l(r2, l=l)

    return np.linalg.pinv(y1).dot(y2)

# https://sahussaintu.files.wordpress.com/2014/03/spherical_harmonics.pdf

#EOF
