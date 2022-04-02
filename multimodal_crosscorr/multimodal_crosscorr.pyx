""" 
Utility functions used by the Multimodal Cross Correlation (MCC) metric
Code includes variations of mine to the original Cross Correlation (crosscorr) code,
adding support for multimodality.

Credit for the original crosscorr code goes to the DIPY team (https://dipy.org/)

Cfir
"""

import numpy as np
from fused_types cimport floating
cimport cython
cimport numpy as cnp
from libc.math cimport pow


cdef inline int _int_max(int a, int b) nogil:
    r"""
    Returns the maximum of a and b
    """
    return a if a >= b else b


cdef inline int _int_min(int a, int b) nogil:
    r"""
    Returns the minimum of a and b
    """
    return a if a <= b else b


cdef enum:
    SI = 0
    SI2 = 1
    SJ = 2
    SJ2 = 3
    SIJ = 4
    CNT = 5

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline int _wrap(int x, int m)nogil:
    r""" Auxiliary function to `wrap` an array around its low-end side.
    Negative indices are mapped to last coordinates so that no extra memory
    is required to account for local rectangular windows that exceed the
    array's low-end boundary.

    Parameters
    ----------
    x : int
        the array position to be wrapped
    m : int
        array length
    """
    if x < 0:
        return x + m
    return x


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cdef inline void _update_factors(double[:, :, :, :, :] factors,
                                 floating[:, :, :, :] moving,
                                 floating[:, :, :, :] static,
                                 int ss, int rr, int cc,
                                 int s, int r, int c, int operation) nogil:
    r"""Updates the precomputed CC factors of a rectangular window

    Updates the precomputed CC factors of the rectangular window centered
    at (`ss`, `rr`, `cc`) by adding the factors corresponding to voxel
    (`s`, `r`, `c`) of input images `moving` and `static`.

    Parameters
    ----------
    factors : array, shape (S, R, C, V, 5)
        array containing the current precomputed factors to be updated
    moving : array, shape (S, R, C, V)
        the moving volume (notice that both images must already be in a common
        reference domain, in particular, they must have the same shape)
    static : array, shape (S, R, C, V)
        the static volume, which also defines the reference registration domain
    ss : int
        first coordinate of the rectangular window to be updated
    rr : int
        second coordinate of the rectangular window to be updated
    cc : int
        third coordinate of the rectangular window to be updated
    s: int
        first coordinate of the voxel the local window should be updated with
    r: int
        second coordinate of the voxel the local window should be updated with
    c: int
        third coordinate of the voxel the local window should be updated with
    operation : int, either -1, 0 or 1
        indicates whether the factors of voxel (`s`, `r`, `c`) should be
        added to (`operation`=1), subtracted from (`operation`=-1), or set as
        (`operation`=0) the current factors for the rectangular window centered
        at (`ss`, `rr`, `cc`).

    """
    cdef:
        floating[:] sval
        floating[:] mval    
    if s >= moving.shape[0] or r >= moving.shape[1] or c >= moving.shape[2]:
        if operation == 0:
            factors[ss, rr, cc, :, SI] = 0
            factors[ss, rr, cc, :, SI2] = 0
            factors[ss, rr, cc, :, SJ] = 0
            factors[ss, rr, cc, :, SJ2] = 0
            factors[ss, rr, cc, :, SIJ] = 0
    else:
        sval = static[s, r, c, :]
        mval = moving[s, r, c, :]
        if operation == 0:
            for v in range(moving.shape[3]):
                factors[ss, rr, cc, v, SI] = sval[v]
                factors[ss, rr, cc, v, SI2] = sval[v]**2
                factors[ss, rr, cc, v, SJ] = mval[v]
                factors[ss, rr, cc, v, SJ2] = mval[v]**2
                factors[ss, rr, cc, v, SIJ] = sval[v] *  mval[v]
        elif operation == -1:
            for v in range(moving.shape[3]):
                factors[ss, rr, cc, v, SI] -= sval[v]
                factors[ss, rr, cc, v, SI2] -= sval[v]**2
                factors[ss, rr, cc, v, SJ] -= mval[v]
                factors[ss, rr, cc, v, SJ2] -= mval[v]**2
                factors[ss, rr, cc, v, SIJ] -= sval[v] *  mval[v]
        elif operation == 1:
            for v in range(moving.shape[3]):
                factors[ss, rr, cc, v, SI] += sval[v]
                factors[ss, rr, cc, v, SI2] += sval[v]**2
                factors[ss, rr, cc, v, SJ] += mval[v]
                factors[ss, rr, cc, v, SJ2] += mval[v]**2
                factors[ss, rr, cc, v, SIJ] += sval[v] *  mval[v]


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d(floating[:, :, :, :] static,
                             floating[:, :, :, :] moving,
                             cnp.npy_intp radius, num_threads=None):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    Pre-computes the separate terms of the cross correlation metric and image
    norms at each voxel considering a neighborhood of the given radius to
    efficiently compute the gradient of the metric with respect to the
    deformation field [Ocegueda2016]_ [Avants2008]_ [Avants2011]_.

    Parameters
    ----------
    static : array, shape (S, R, C, V)
        the static volume, which also defines the reference registration domain
    moving : array, shape (S, R, C, V)
        the moving volume (notice that both images must already be in a common
        reference domain, i.e. the same S, R, C, V)
    radius : the radius of the neighborhood (cube of (2 * radius + 1)^3 voxels)

    Returns
    -------
    factors : array, shape (S, R, C, V, 5)
        the precomputed cross correlation terms:
        factors[:,:,:,:,0] : static minus its mean value along the neighborhood
        factors[:,:,:,:,1] : moving minus its mean value along the neighborhood
        factors[:,:,:,:,2] : sum of the pointwise products of static and moving
                             along the neighborhood
        factors[:,:,:,:,3] : sum of sq. values of static along the neighborhood
        factors[:,:,:,:,4] : sum of sq. values of moving along the neighborhood

    References
    ----------
    .. [Ocegueda2016]_ Ocegueda, O., Dalmau, O., Garyfallidis, E., Descoteaux,
        M., & Rivera, M. (2016). On the computation of integrals over
        fixed-size rectangles of arbitrary dimension, Pattern Recognition
        Letters. doi:10.1016/j.patrec.2016.05.008
    .. [Avants2008]_ Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C.
        (2008). Symmetric Diffeomorphic Image Registration with
        Cross-Correlation: Evaluating Automated Labeling of Elderly and
        Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    .. [Avants2011]_ Avants, B. B., Tustison, N., & Song, G. (2011). Advanced
        Normalization Tools (ANTS), 1-35.
    """
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp nv  =static.shape[3]
        
        cnp.npy_intp side = 2 * radius + 1
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        cnp.npy_intp s, r, c, it, sides, sider, sidec
        double cnt
        cnp.npy_intp ssss, sss, ss, rr, cc, prev_ss, prev_rr, prev_cc
        double[:] Imean = np.zeros((nv), dtype=np.float64)
        double[:] Jmean = np.zeros((nv), dtype=np.float64)
        double[:] IJprods = np.zeros((nv), dtype=np.float64)
        double[:] Isq = np.zeros((nv), dtype=np.float64)
        double[:] Jsq = np.zeros((nv), dtype=np.float64)
        double[:, :, :, :, :] temp = np.zeros((2, nr, nc, nv, 5), dtype=np.float64)
        floating[:, :, :, :, :] factors = np.zeros((ns, nr, nc, nv, 5),
                                                dtype=np.asarray(static).dtype)

    with nogil:
        sss = 1
        for s in range(ns+radius):
            ss = _wrap(s - radius, ns)
            sss = 1 - sss
            firsts = _int_max(0, ss - radius)
            lasts = _int_min(ns - 1, ss + radius)
            sides = (lasts - firsts + 1)
            for r in range(nr+radius):
                rr = _wrap(r - radius, nr)
                firstr = _int_max(0, rr - radius)
                lastr = _int_min(nr - 1, rr + radius)
                sider = (lastr - firstr + 1)
                for c in range(nc+radius):
                    cc = _wrap(c - radius, nc)
                    # New corner
                    _update_factors(temp, moving, static,
                                    sss, rr, cc, s, r, c, 0)
    
                    # Add signed sub-volumes
                    if s > 0:
                        prev_ss = 1 - sss
                        for it in range(5):
                            for v in range(nv):
                                temp[sss, rr, cc, v, it] += temp[prev_ss, rr, cc, v, it]
                        if r > 0:
                            prev_rr = _wrap(rr-1, nr)
                            for it in range(5):
                                for v in range(nv):
                                    temp[sss, rr, cc, v, it] -= \
                                        temp[prev_ss, prev_rr, cc, v, it]
                            if c > 0:
                                prev_cc = _wrap(cc-1, nc)
                                for it in range(5):
                                    for v in range(nv):
                                        temp[sss, rr, cc, v, it] += \
                                            temp[prev_ss, prev_rr, prev_cc, v, it]
                        if c > 0:
                            prev_cc = _wrap(cc-1, nc)
                            for it in range(5):
                                for v in range(nv):
                                    temp[sss, rr, cc, v, it] -= \
                                        temp[prev_ss, rr, prev_cc, v, it]
                    if(r > 0):
                        prev_rr = _wrap(rr-1, nr)
                        for it in range(5):
                            for v in range(nv):
                                temp[sss, rr, cc, v, it] += \
                                    temp[sss, prev_rr, cc, v, it]
                        if(c > 0):
                            prev_cc = _wrap(cc-1, nc)
                            for it in range(5):
                                for v in range(nv):
                                    temp[sss, rr, cc, v, it] -= \
                                        temp[sss, prev_rr, prev_cc, v, it]
                    if(c > 0):
                        prev_cc = _wrap(cc-1, nc)
                        for it in range(5):
                            for v in range(nv):
                                temp[sss, rr, cc, v, it] += temp[sss, rr, prev_cc, v, it]
    
                    # Add signed corners
                    if s >= side:
                        _update_factors(temp, moving, static,
                                        sss, rr, cc, s-side, r, c, -1)
                        if r >= side:
                            _update_factors(temp, moving, static,
                                            sss, rr, cc, s-side, r-side, c, 1)
                            if c >= side:
                                _update_factors(temp, moving, static, sss, rr,
                                                cc, s-side, r-side, c-side, -1)
                        if c >= side:
                            _update_factors(temp, moving, static,
                                            sss, rr, cc, s-side, r, c-side, 1)
                    if r >= side:
                        _update_factors(temp, moving, static,
                                        sss, rr, cc, s, r-side, c, -1)
                        if c >= side:
                            _update_factors(temp, moving, static,
                                            sss, rr, cc, s, r-side, c-side, 1)
    
                    if c >= side:
                        _update_factors(temp, moving, static,
                                        sss, rr, cc, s, r, c-side, -1)
                    # Compute final factors
                    if s >= radius and r >= radius and c >= radius:
                        firstc = _int_max(0, cc - radius)
                        lastc = _int_min(nc - 1, cc + radius)
                        sidec = (lastc - firstc + 1)
                        cnt = sides*sider*sidec
                        for v in range(nv):
                            Imean[v] = temp[sss, rr, cc, v, SI] / cnt
                            Jmean[v] = temp[sss, rr, cc, v, SJ] / cnt
                        for v in range(nv):
                            IJprods[v] = (temp[sss, rr, cc, v, SIJ] -
                                         Jmean[v] * temp[sss, rr, cc, v, SI] -
                                         Imean[v] * temp[sss, rr, cc, v, SJ] +
                                         cnt * Jmean[v] * Imean[v])
                            Isq[v] = (temp[sss, rr, cc, v, SI2] -
                                     Imean[v] * temp[sss, rr, cc, v, SI] -
                                     Imean[v] * temp[sss, rr, cc, v, SI] +
                                     cnt * Imean[v] * Imean[v])
                            Jsq[v] = (temp[sss, rr, cc, v, SJ2] -
                                     Jmean[v] * temp[sss, rr, cc, v, SJ] -
                                     Jmean[v] * temp[sss, rr, cc, v, SJ] +
                                     cnt * Jmean[v] * Jmean[v])
                        for v in range(nv):
                            factors[ss, rr, cc, v, 0] = static[ss, rr, cc, v] - Imean[v]
                            factors[ss, rr, cc, v, 1] = moving[ss, rr, cc, v] - Jmean[v]
                            factors[ss, rr, cc, v, 2] = IJprods[v]
                            factors[ss, rr, cc, v, 3] = Isq[v]
                            factors[ss, rr, cc, v, 4] = Jsq[v]
                    
    return factors


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def precompute_cc_factors_3d_test(floating[:, :, :, :] static,
                                  floating[:, :, :, :] moving, int radius):
    r"""Precomputations to quickly compute the gradient of the CC Metric

    This version of precompute_cc_factors_3d is for testing purposes, it
    directly computes the local cross-correlation factors without any
    optimization, so it is less error-prone than the accelerated version.
    """
    cdef:
        cnp.npy_intp ns = static.shape[0]
        cnp.npy_intp nr = static.shape[1]
        cnp.npy_intp nc = static.shape[2]
        cnp.npy_intp nv = static.shape[3]
        cnp.npy_intp s, r, c, k, i, j, t
        cnp.npy_intp firstc, lastc, firstr, lastr, firsts, lasts
        double[:] Imean = np.zeros((nv), dtype=np.float64)
        double[:] Jmean = np.zeros((nv), dtype=np.float64)
        floating[:, :, :, :, :] factors = np.zeros((ns, nr, nc, nv, 5),
                                                dtype=np.asarray(static).dtype)
        double[:, :] sums = np.zeros((6,nv), dtype=np.float64)

    with nogil:
        for s in range(ns):
            firsts = _int_max(0, s - radius)
            lasts = _int_min(ns - 1, s + radius)
            for r in range(nr):
                firstr = _int_max(0, r - radius)
                lastr = _int_min(nr - 1, r + radius)
                for c in range(nc):
                    firstc = _int_max(0, c - radius)
                    lastc = _int_min(nc - 1, c + radius)
                    for t in range(6):
                        sums[t] = 0
                    for k in range(firsts, 1 + lasts):
                        for i in range(firstr, 1 + lastr):
                            for j in range(firstc, 1 + lastc):
                                for v in range(nv):
                                    sums[SI, v] += static[k, i, j, v]
                                    sums[SI2, v] += static[k, i, j, v] * static[k, i, j, v]
                                    sums[SJ, v] += moving[k, i, j, v]
                                    sums[SJ2, v] += moving[k, i, j, v] * moving[k, i, j, v]
                                    sums[SIJ, v] += static[k, i, j, v] * moving[k, i, j, v]
                                    sums[CNT, v] += 1
                    for v in range(nv):
                        Imean[v] = sums[SI, v]/ sums[CNT, v]
                        Jmean[v] = sums[SJ, v]/ sums[CNT, v]
                    for v in range(nv):
                        factors[s, r, c, v, 0] = static[s, r, c, v] - Imean[v]
                        factors[s, r, c, v, 1] = moving[s, r, c, v] - Jmean[v]
                        factors[s, r, c, v, 2] = (sums[SIJ, v] - Jmean[v] * sums[SI, v] -
                                                 Imean[v] * sums[SJ, v] +
                                                 sums[CNT, v] * Jmean[v] * Imean[v])
                        factors[s, r, c, v, 3] = (sums[SI2, v] - Imean[v] * sums[SI, v] -
                                                 Imean[v] * sums[SI, v] +
                                                 sums[CNT, v] * Imean[v] * Imean[v])
                        factors[s, r, c, v, 4] = (sums[SJ2, v] - Jmean[v] * sums[SJ, v] -
                                                 Jmean[v] * sums[SJ, v] +
                                                 sums[CNT, v] * Jmean[v] * Jmean[v])
                    
    return np.asarray(factors)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_forward_step_3d(floating[:, :, :, :, :] grad_static,
                               floating[:, :, :, :, :] factors,
                               cnp.npy_intp radius):
    r"""Gradient of the CC Metric w.r.t. the forward transformation

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) [Avants2008]_ w.r.t. the displacement associated to
    the moving volume ('forward' step) as in [Avants2011]_

    Parameters
    ----------
    grad_static : array, shape (S, R, C, V, 3)
        the gradient of the static volume
    factors : array, shape (S, R, C, V, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.

    Returns
    -------
    out : array, shape (S, R, C, 3)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the moving volume
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    .. [Avants2008]_ Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C.
        (2008). Symmetric Diffeomorphic Image Registration with
        Cross-Correlation: Evaluating Automated Labeling of Elderly and
        Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    .. [Avants2011]_ Avants, B. B., Tustison, N., & Song, G. (2011). Advanced
        Normalization Tools (ANTS), 1-35.
    """
    cdef:
        cnp.npy_intp ns = grad_static.shape[0]
        cnp.npy_intp nr = grad_static.shape[1]
        cnp.npy_intp nc = grad_static.shape[2]
        cnp.npy_intp nv = grad_static.shape[3]
        double energy = 0
        cnp.npy_intp s, r, c
        double[:] Ii = np.zeros((nv), dtype=np.float64)
        double[:] Ji = np.zeros((nv), dtype=np.float64)
        double[:] sfm = np.zeros((nv), dtype=np.float64)
        double[:] sff = np.zeros((nv), dtype=np.float64)
        double[:] smm = np.zeros((nv), dtype=np.float64)
        double localCorrelation, temp_grad, temp
        double sff_smm_norm, sfm_norm, smm_norm, sff_norm
        floating[:, :, :, :] out =\
            np.zeros((ns, nr, nc, 3), dtype=np.asarray(grad_static).dtype)
            
    with nogil:
        
        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    sff_norm = 0
                    smm_norm = 0
                    for v in range(nv):
                        Ii[v] = factors[s, r, c, v, 0]
                        Ji[v] = factors[s, r, c, v, 1]
                        sfm[v] = factors[s, r, c, v, 2]
                        sff[v] = factors[s, r, c, v, 3]
                        smm[v] = factors[s, r, c, v, 4]
                        sff_norm += sff[v]
                        smm_norm += smm[v]
                    sff_norm = pow(sff_norm, 0.5)
                    smm_norm = pow(smm_norm, 0.5)
                    if(sff_norm == 0.0 or smm_norm == 0.0):
                        continue
                    localCorrelation = 0
                    
                    sff_smm_norm = 0
                    sfm_norm = 0
                    for v in range(nv):
                        sff_smm_norm += sff[v] * smm[v]
                        sfm_norm += sff[v]**2
                    sff_smm_norm = pow(sff_smm_norm, 0.5)
                    sfm_norm = pow(sfm_norm, 0.5)
                    
                    if(sff_smm_norm > 1e-5):
                        localCorrelation = sfm_norm / sff_smm_norm
                    if(localCorrelation < 1):  # avoid bad values...
                        energy -= localCorrelation

                    for idx in range(3):
                        temp_grad = 0
                        for v in range(nv):
                            if (sff[v] == 0 or smm[v] == 0):
                                continue
                            temp = 2.0 * sfm[v] / (sff[v] * smm[v]) * (Ji[v] - sfm[v] / sff[v] * Ii[v])
                            temp_grad += temp * grad_static[s, r, c, v ,idx]
                        out[s, r, c, idx] -= temp_grad
                            
                
    return np.asarray(out), energy


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_cc_backward_step_3d(floating[:, :, :, :, :] grad_moving,
                                floating[:, :, :, :, :] factors,
                                cnp.npy_intp radius):
    r"""Gradient of the CC Metric w.r.t. the backward transformation

    Computes the gradient of the Cross Correlation metric for symmetric
    registration (SyN) [Avants08]_ w.r.t. the displacement associated to
    the static volume ('backward' step) as in [Avants11]_

    Parameters
    ----------
    grad_moving : array, shape (S, R, C, V, 3)
        the gradient of the moving volume
    factors : array, shape (S, R, C, V, 5)
        the precomputed cross correlation terms obtained via
        precompute_cc_factors_3d
    radius : int
        the radius of the neighborhood used for the CC metric when
        computing the factors. The returned vector field will be
        zero along a boundary of width radius voxels.

    Returns
    -------
    out : array, shape (S, R, C, V, 3)
        the gradient of the cross correlation metric with respect to the
        displacement associated to the static volume
    energy : the cross correlation energy (data term) at this iteration

    References
    ----------
    [Avants08]_ Avants, B. B., Epstein, C. L., Grossman, M., & Gee, J. C. (2008)
               Symmetric Diffeomorphic Image Registration with
               Cross-Correlation: Evaluating Automated Labeling of Elderly and
               Neurodegenerative Brain, Med Image Anal. 12(1), 26-41.
    [Avants11]_ Avants, B. B., Tustison, N., & Song, G. (2011).
               Advanced Normalization Tools (ANTS), 1-35.
    """
    ftype = np.asarray(grad_moving).dtype
    cdef:
        cnp.npy_intp ns = grad_moving.shape[0]
        cnp.npy_intp nr = grad_moving.shape[1]
        cnp.npy_intp nc = grad_moving.shape[2]
        cnp.npy_intp nv = grad_moving.shape[3]
        cnp.npy_intp s, r, c
        double energy = 0
        floating[:] Ii = np.zeros((nv), dtype=ftype)
        floating[:] Ji = np.zeros((nv), dtype=ftype)
        floating[:] sfm = np.zeros((nv), dtype=ftype)
        floating[:] sff = np.zeros((nv), dtype=ftype)
        floating[:] smm = np.zeros((nv), dtype=ftype)
        double localCorrelation, temp_grad, temp
        double sff_smm_norm, sfm_norm, smm_norm, sff_norm
        floating[:, :, :, :] out = np.zeros((ns, nr, nc, 3), dtype=ftype)

    with nogil:

        for s in range(radius, ns-radius):
            for r in range(radius, nr-radius):
                for c in range(radius, nc-radius):
                    smm_norm = 0
                    sff_norm = 0
                    for v in range(nv):
                        Ii[v] = factors[s, r, c, v, 0]
                        Ji[v] = factors[s, r, c, v, 1]
                        sfm[v] = factors[s, r, c, v, 2]
                        sff[v] = factors[s, r, c, v, 3]
                        smm[v] = factors[s, r, c, v, 4]
                        sff_norm += sff[v]
                        smm_norm += smm[v]
                    sff_norm = pow(sff_norm, 0.5)
                    smm_norm = pow(smm_norm, 0.5)
                    if(sff_norm == 0.0 or smm_norm == 0.0):
                        continue
                    localCorrelation = 0
                    
                    sff_smm_norm = 0
                    sfm_norm = 0
                    for v in range(nv):
                        sff_smm_norm += sff[v] * smm[v]
                        sfm_norm += sff[v]**2
                    sff_smm_norm = pow(sff_smm_norm, 0.5)
                    sfm_norm = pow(sfm_norm, 0.5)           
                    
                    if(sff_smm_norm > 1e-5):
                        localCorrelation = sfm_norm/ sff_smm_norm
                    if(localCorrelation < 1):  # avoid bad values...
                        energy -= localCorrelation
                        
                    for idx in range(3):
                        temp_grad = 0
                        for v in range(nv):
                            if (sff[v] == 0 or smm[v] == 0):
                                continue
                            temp = 2.0 * sfm[v] / (sff[v] * smm[v]) * (Ii[v] - sfm[v] / smm[v] * Ji[v])
                            temp_grad += temp * grad_moving[s, r, c, v, idx]
                        out[s, r, c, idx] -= temp_grad
                    
    return np.asarray(out), energy

## TODO: Remove Norms calculations and run all process in a single nv loop