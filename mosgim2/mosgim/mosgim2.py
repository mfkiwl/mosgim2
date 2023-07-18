import numpy as np
from lemkelcp.lemkelcp import lemkelcp
from scipy.special import sph_harm
from scipy.sparse import lil_matrix, csr_matrix
from pathlib import Path
from datetime import timedelta
from mosgim2.utils.geom_utils import MF
from concurrent.futures import ProcessPoolExecutor, as_completed


# configuration
GB_CHUNK = 15000 
#################################


def calc_coefs(M, N, theta, phi, sf):
    """
    :param M: meshgrid of harmonics degrees
    :param N: meshgrid of harmonics orders
    :param theta: LT of IPP in rad
    :param phi: co latitude of IPP in rad
    :param sf: slant factor
    """
    n_coefs = len(M)
    a = np.zeros(n_coefs)
    Ymn = sph_harm(np.abs(M), N, theta, phi)  # complex harmonics on meshgrid
    #  introducing real basis according to scipy normalization
    a[M < 0] = Ymn[M < 0].imag * np.sqrt(2) * (-1.) ** M[M < 0]
    a[M > 0] = Ymn[M > 0].real * np.sqrt(2) * (-1.) ** M[M > 0]
    a[M == 0] = Ymn[M == 0].real
    del Ymn
    return a*sf
vcoefs = np.vectorize(calc_coefs, excluded=['M','N'], otypes=[np.ndarray])
 

def construct_normal_system(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, nT, ndays, sigma0, time, theta1, phi1, theta2, phi2, el, time_ref, theta1_ref, phi1_ref, theta2_ref, phi2_ref, el_ref, rhs, linear):
    """
    :param nbig_layer1: maximum order of spherical harmonic for layer1
    :param mbig_layer1: maximum degree of spherical harmonic for layer1
    :param nbig_layer2: maximum order of spherical harmonic for layer2
    :param mbig_layer2: maximum degree of spherical harmonic for layer2
    :IPPh_layer1: height of layer1 [m]
    :IPPh_layer2: height of layer2 [m]
    :param nT: number of time intervals
    :param ndays: number of days in analysis
    :param sigma0: standard deviation of unit weight [TECu]
    :param time: array of times of IPPs in secs
    :param theta1: array of LTs of IPPs in rads for layer1
    :param phi1: array of co latitudes of IPPs in rads for layer1
    :param theta2: array of LTs of IPPs in rads for layer2
    :param phi2: array of co latitudes of IPPs in rads for layer2
    :param el: array of elevation angles in rads
    :param time_ref: array of ref times of IPPs in sec
    :param theta1_ref: array of ref longitudes (LTs) of IPPs in rads for layer1
    :param phi1_ref: array of ref co latitudes of IPPs in rads for layer1
    :param theta2_ref: array of ref longitudes (LTs) of IPPs in rads for layer2
    :param phi2_ref: array of ref co latitudes of IPPs in rads for layer2
    :param el_ref: array of ref elevation angles in rads
    :param rhs: array of rhs (measurements TEC difference on current and ref rays)
    :param linear: bool defines const or linear
    """
    print('constructing normal system for series')
    tmc = time
    tmr = time_ref

    SF1 = MF(el, IPPh_layer1)
    SF1_ref = MF(el_ref, IPPh_layer1)

    SF2 = MF(el, IPPh_layer2)
    SF2_ref = MF(el_ref, IPPh_layer2)
 
    # Construct weight matrix for the observations
    len_rhs = len(rhs)
    P = lil_matrix((len_rhs, len_rhs))
    el_sin = np.sin(el)
    elr_sin = np.sin(el_ref)
    diagP = (el_sin ** 2) * (elr_sin ** 2) / (el_sin ** 2 + elr_sin ** 2)
    P.setdiag(diagP)
    P = P.tocsr()
 
    # Construct matrix of the problem (A)
    n_ind_layer1 = np.arange(0, nbig_layer1 + 1, 1)
    m_ind_layer1 = np.arange(-mbig_layer1, mbig_layer1 + 1, 1)
    M_layer1, N_layer1 = np.meshgrid(m_ind_layer1, n_ind_layer1)
    Y_layer1 = sph_harm(np.abs(M_layer1), N_layer1, 0, 0)
    idx_layer1 = np.isfinite(Y_layer1)
    M_layer1 = M_layer1[idx_layer1]
    N_layer1 = N_layer1[idx_layer1]
    n_coefs_layer1 = len(M_layer1)
 

    n_ind_layer2 = np.arange(0, nbig_layer2 + 1, 1)
    m_ind_layer2 = np.arange(-mbig_layer2, mbig_layer2 + 1, 1)
    M_layer2, N_layer2 = np.meshgrid(m_ind_layer2, n_ind_layer2)
    Y_layer2 = sph_harm(np.abs(M_layer2), N_layer2, 0, 0)
    idx_layer2 = np.isfinite(Y_layer2)
    M_layer2 = M_layer2[idx_layer2]
    N_layer2 = N_layer2[idx_layer2]
    n_coefs_layer2 = len(M_layer2)

    n_coefs = n_coefs_layer1 + n_coefs_layer2

    tic = (tmc * nT / (ndays * 86400.)).astype('int16')
    tir = (tmr * nT / (ndays * 86400.)).astype('int16')

    ac1 = vcoefs(M=M_layer1, N=N_layer1, theta=theta1, phi=phi1, sf=SF1)
    ar1 = vcoefs(M=M_layer1, N=N_layer1, theta=theta1_ref, phi=phi1_ref, sf=SF1_ref)
 

    ac2 = vcoefs(M=M_layer2, N=N_layer2, theta=theta2, phi=phi2, sf=SF2)
    ar2 = vcoefs(M=M_layer2, N=N_layer2, theta=theta2_ref, phi=phi2_ref, sf=SF2_ref)


    print('coefs done', n_coefs, nT, ndays, len_rhs)

    #prepare (A) in csr sparse format
    dims = 4 if linear else 2
    nT_add = 1 if linear else 0
    data = np.empty(len_rhs * n_coefs * dims)
    rowi = np.empty(len_rhs * n_coefs * dims)
    coli = np.empty(len_rhs * n_coefs * dims)

    if linear:
        hour_cc = (ndays * 86400.) * tic / nT    
        hour_cn = (ndays * 86400.) * (tic + 1) / nT    
        hour_rc = (ndays * 86400.) * tir / nT    
        hour_rn = (ndays * 86400.) * (tir + 1) / nT  
        dt = (ndays * 86400.) / nT 
        for i in range(0, len_rhs, 1): 

           data[i * (n_coefs) * dims + n_coefs_layer1 * 0: i * (n_coefs) * dims + n_coefs_layer1 * 1] = (  tmc[i] - hour_cc[i]) * ac1[i] / dt  
           data[i * (n_coefs) * dims + n_coefs_layer1 * 1: i * (n_coefs) * dims + n_coefs_layer1 * 2] = ( -tmc[i] + hour_cn[i]) * ac1[i] / dt
           data[i * (n_coefs) * dims + n_coefs_layer1 * 2: i * (n_coefs) * dims + n_coefs_layer1 * 3] = - (  tmr[i] - hour_rc[i]) * ar1[i] / dt
           data[i * (n_coefs) * dims + n_coefs_layer1 * 3: i * (n_coefs) * dims + n_coefs_layer1 * 4] = - ( -tmr[i] + hour_rn[i]) * ar1[i] / dt

           data[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 0: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 1] = (  tmc[i] - hour_cc[i]) * ac2[i] / dt
           data[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 1: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 2] = ( -tmc[i] + hour_cn[i]) * ac2[i] / dt
           data[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 2: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 3] = - (  tmr[i] - hour_rc[i]) * ar2[i] / dt
           data[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 3: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 4] = - ( -tmr[i] + hour_rn[i]) * ar2[i] / dt

           rowi[i * (n_coefs) * dims: (i+1) * (n_coefs) * dims] = i

           coli[i * (n_coefs) * dims + n_coefs_layer1 * 0: i * (n_coefs) * dims + n_coefs_layer1 * 1] = np.arange((tic[i] + 0) * n_coefs, (tic[i] + 0) * n_coefs + n_coefs_layer1, 1)
           coli[i * (n_coefs) * dims + n_coefs_layer1 * 1: i * (n_coefs) * dims + n_coefs_layer1 * 2] = np.arange((tic[i] + 1) * n_coefs, (tic[i] + 1) * n_coefs + n_coefs_layer1, 1)
           coli[i * (n_coefs) * dims + n_coefs_layer1 * 2: i * (n_coefs) * dims + n_coefs_layer1 * 3] = np.arange((tir[i] + 0) * n_coefs, (tir[i] + 0) * n_coefs + n_coefs_layer1, 1)
           coli[i * (n_coefs) * dims + n_coefs_layer1 * 3: i * (n_coefs) * dims + n_coefs_layer1 * 4] = np.arange((tir[i] + 1) * n_coefs, (tir[i] + 1) * n_coefs + n_coefs_layer1, 1)

           coli[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 0: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 1] = np.arange((tic[i] + 0) * n_coefs + n_coefs_layer1, (tic[i] + 0) * n_coefs + n_coefs_layer1 + n_coefs_layer2,  1)

           coli[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 1: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 2] = np.arange((tic[i] + 1) * n_coefs + n_coefs_layer1, (tic[i] + 1) * n_coefs + n_coefs_layer1 + n_coefs_layer2, 1)

           coli[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 2: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 3] = np.arange((tir[i] + 0) * n_coefs + n_coefs_layer1, (tir[i] + 0) * n_coefs + n_coefs_layer1 + n_coefs_layer2, 1)

           coli[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 3: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2 * 4] = np.arange((tir[i] + 1) * n_coefs + n_coefs_layer1, (tir[i] + 1) * n_coefs + n_coefs_layer1 + n_coefs_layer2, 1)

    else:
        for i in range(0, len_rhs, 1): 

           data[i * (n_coefs) * dims: i * (n_coefs) * dims + n_coefs_layer1] = ac1[i]
           data[i * (n_coefs) * dims + n_coefs_layer1: i * (n_coefs) * dims + n_coefs_layer1 * dims] = -ar1[i]
           data[i * (n_coefs) * dims + n_coefs_layer1 * dims: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2] = ac2[i]
           data[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2: (i + 1) * (n_coefs) * dims] = -ar2[i]

           rowi[i * (n_coefs) * dims: (i+1) * (n_coefs) * dims] = i 

           coli[i * (n_coefs) * dims: i * (n_coefs) * dims + n_coefs_layer1] = np.arange(tic[i] * (n_coefs), tic[i] * (n_coefs) + n_coefs_layer1, 1)

           coli[i * (n_coefs) * dims + n_coefs_layer1: i * (n_coefs) * dims + n_coefs_layer1 * dims] = np.arange(tir[i] * (n_coefs), tir[i] * (n_coefs) + n_coefs_layer1, 1)

           coli[i * (n_coefs) * dims + n_coefs_layer1 * dims: i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2] = np.arange(tic[i] * (n_coefs)  + n_coefs_layer1, (tic[i] + 1) * (n_coefs), 1)

           coli[i * (n_coefs) * dims + n_coefs_layer1 * dims + n_coefs_layer2: (i + 1) * (n_coefs) * dims] = np.arange(tir[i] * (n_coefs)  + n_coefs_layer1, (tir[i] + 1) * (n_coefs), 1)



    A = csr_matrix((data, (rowi, coli)), 
                   shape=(len_rhs, (nT + nT_add) * n_coefs))
    print('matrix (A) for subset done')
 
    # define normal system
    AP = A.transpose().dot(P)
    N = AP.dot(A).todense()    
    b = AP.dot(rhs)
    rhsT_P_rhs = np.dot(diagP, np.square(rhs))
    print('normal matrix (N) for subset done')

    return N, b, rhsT_P_rhs, len_rhs


def stack_constrain_solve_ns(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, nT, ndays, sigma0, sigma_v, time_chunks, mlt1_chunks, mcolat1_chunks, mlt2_chunks, mcolat2_chunks, el_chunks, 
                             time_ref_chunks, mlt1_ref_chunks, mcolat1_ref_chunks, mlt2_ref_chunks, mcolat2_ref_chunks, el_ref_chunks, rhs_chunks, nworkers=3, linear=True):


    nT_add = 1 if linear else 0
    n_coefs_layer1 = (nbig_layer1 + 1)**2 - (nbig_layer1 - mbig_layer1) * (nbig_layer1 - mbig_layer1 + 1)
    n_coefs_layer2 = (nbig_layer2 + 1)**2 - (nbig_layer2 - mbig_layer2) * (nbig_layer2 - mbig_layer2 + 1)

    n_coefs = n_coefs_layer1 + n_coefs_layer2

    N = np.zeros((n_coefs * (nT + nT_add), n_coefs * (nT + nT_add)))
    b = np.zeros(n_coefs * (nT + nT_add))
    rhsT_P_rhs = 0.
    DOF = -n_coefs # degrees of freedom    


    with ProcessPoolExecutor(max_workers=nworkers) as executor:
        queue = []
        for chunk  in zip(time_chunks, mlt1_chunks, mcolat1_chunks, mlt2_chunks, mcolat2_chunks,
                          el_chunks, time_ref_chunks, mlt1_ref_chunks, mcolat1_ref_chunks, mlt2_ref_chunks, mcolat2_ref_chunks, el_ref_chunks, rhs_chunks):

            params = (nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, nT, ndays, sigma0) + chunk + (linear, )
            query = executor.submit(construct_normal_system, *params)
            queue.append(query)
        for v in as_completed(queue):
            NN, bb, rPr, lrhs = v.result()
            N += NN
            b += bb
            rhsT_P_rhs += rPr
            DOF += lrhs

    # imposing frozen conditions on consequitive maps coeffs
    for ii in range(0, nT - 1 + nT_add, 1): 
        for kk in range(0, n_coefs):
            N[ii*n_coefs + kk, ii*n_coefs + kk] += (sigma0 / sigma_v)**2
            N[(ii + 1) * n_coefs + kk, (ii+1) * n_coefs + kk] += (sigma0 / sigma_v)**2
            N[(ii + 1) * n_coefs + kk, ii * n_coefs + kk] += -(sigma0 / sigma_v)**2
            N[ii * n_coefs + kk, (ii + 1) * n_coefs + kk] += -(sigma0 / sigma_v)**2
            DOF += 1
    print('normal matrix (N) constraints added')


    # # solve normal system
    Ninv = np.linalg.inv(N)     
    res = np.dot(Ninv, b)  
    print('normal system solved')
    
    disp_scale = (rhsT_P_rhs - np.dot(res, b)) / DOF  

    print(disp_scale)

    return res, disp_scale, Ninv


# non paralel version, keept for large memory problems
def stack_constrain_solve_ns_np(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, nT, ndays, sigma0, sigma_v, time_chunks, mlt1_chunks, mcolat1_chunks, mlt2_chunks, mcolat2_chunks, el_chunks, 
                             time_ref_chunks, mlt1_ref_chunks, mcolat1_ref_chunks, mlt2_ref_chunks, mcolat2_ref_chunks, el_ref_chunks, rhs_chunks, linear=True):


    nT_add = 1 if linear else 0
    n_coefs_layer1 = (nbig_layer1 + 1)**2 - (nbig_layer1 - mbig_layer1) * (nbig_layer1 - mbig_layer1 + 1)
    n_coefs_layer2 = (nbig_layer2 + 1)**2 - (nbig_layer2 - mbig_layer2) * (nbig_layer2 - mbig_layer2 + 1)

    n_coefs = n_coefs_layer1 + n_coefs_layer2

    N = np.zeros((n_coefs * (nT + nT_add), n_coefs * (nT + nT_add)))
    b = np.zeros(n_coefs * (nT + nT_add))
    rhsT_P_rhs = 0.
    DOF = -n_coefs # degrees of freedom    


    for chunk  in zip(time_chunks, mlt1_chunks, mcolat1_chunks, mlt2_chunks, mcolat2_chunks,
                      el_chunks, time_ref_chunks, mlt1_ref_chunks, mcolat1_ref_chunks, mlt2_ref_chunks, mcolat2_ref_chunks, el_ref_chunks, rhs_chunks):

        params = (nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, nT, ndays, sigma0) + chunk + (linear, )
        NN, bb, rPr, lrhs = construct_normal_system(*params)
        N += NN
        b += bb
        rhsT_P_rhs += rPr
        DOF += lrhs

    # imposing frozen conditions on consequitive maps coeffs
    for ii in range(0, nT - 1 + nT_add, 1): 
        for kk in range(0, n_coefs):
            N[ii*n_coefs + kk, ii*n_coefs + kk] += (sigma0 / sigma_v)**2
            N[(ii + 1) * n_coefs + kk, (ii+1) * n_coefs + kk] += (sigma0 / sigma_v)**2
            N[(ii + 1) * n_coefs + kk, ii * n_coefs + kk] += -(sigma0 / sigma_v)**2
            N[ii * n_coefs + kk, (ii + 1) * n_coefs + kk] += -(sigma0 / sigma_v)**2
            DOF += 1
    print('normal matrix (N) constraints added')


    # # solve normal system
    Ninv = np.linalg.inv(N)     
    res = np.dot(Ninv, b)  
    print('normal system solved')
    
    disp_scale = (rhsT_P_rhs - np.dot(res, b)) / DOF  

    print(disp_scale)

    return res, disp_scale, Ninv


# constructing observation matrix to use in LCP problem
def constructG(nbig1, mbig1, nbig2, mbig2, nT, theta, phi, timeindex, linear):
    """
    :param nbig1: maximum order of spherical harmonic layer1
    :param mbig1: maximum degree of spherical harmonic layer1
    :param nbig2: maximum order of spherical harmonic layer2
    :param mbig2: maximum degree of spherical harmonic layer2
    :param nT: number of time intervals for whole period
    :param theta: array of LTs of IPPs in rads
    :param phi: array of co latitudes of IPPs in rads
    :param timeindex: array of timeindex
    :param linear: boolean for linear or const interpolation

    """
    nT_add = 1 if linear else 0
 
    # Construct observation matrix (G)
    n_ind1 = np.arange(0, nbig1 + 1, 1)
    m_ind1 = np.arange(-mbig1, mbig1 + 1, 1)
    M1, N1 = np.meshgrid(m_ind1, n_ind1)
    Y1 = sph_harm(np.abs(M1), N1, 0, 0)
    idx1 = np.isfinite(Y1)
    M1 = M1[idx1]
    N1 = N1[idx1]
    n_coefs_layer1 = len(M1)

    a1 = vcoefs(M=M1, N=N1, theta=theta, phi=phi, sf=np.ones(len(phi)))

    n_ind2 = np.arange(0, nbig2 + 1, 1)
    m_ind2 = np.arange(-mbig2, mbig2 + 1, 1)
    M2, N2 = np.meshgrid(m_ind2, n_ind2)
    Y2 = sph_harm(np.abs(M2), N2, 0, 0)
    idx2 = np.isfinite(Y2)
    M2 = M2[idx2]
    N2 = N2[idx2]
    n_coefs_layer2 = len(M2)

    a2 = vcoefs(M=M2, N=N2, theta=theta, phi=phi, sf=np.ones(len(phi)))


    len_rhs = len(phi)

    n_coefs = n_coefs_layer1 + n_coefs_layer2

    print('coefs done', n_coefs)

    #prepare (G) in csr sparse format
    data = np.empty(len_rhs * n_coefs)
    rowi = np.empty(len_rhs * n_coefs)
    coli = np.empty(len_rhs * n_coefs )

    for i in range(0, len_rhs,1): 
        data[i * n_coefs_layer1: (i + 1) * n_coefs_layer1] = a1[i]

        rowi[i * n_coefs_layer1: (i + 1) * n_coefs_layer1] = i 
        coli[i * n_coefs_layer1: (i + 1) * n_coefs_layer1] = np.arange(timeindex[i] * n_coefs, timeindex[i] * n_coefs + n_coefs_layer1, 1) 

    for i in range(0, len_rhs,1): 
        data[len_rhs * n_coefs_layer1 + i * n_coefs_layer2: len_rhs * n_coefs_layer1 + (i + 1) * n_coefs_layer2] = a2[i]

        rowi[len_rhs * n_coefs_layer1 + i * n_coefs_layer2: len_rhs * n_coefs_layer1 + (i + 1) * n_coefs_layer2] = (len_rhs + i) 
        coli[len_rhs * n_coefs_layer1 + i * n_coefs_layer2: len_rhs * n_coefs_layer1 + (i + 1) * n_coefs_layer2] = np.arange(timeindex[i] * n_coefs + n_coefs_layer1, (timeindex[i] + 1) * n_coefs, 1) 


    G = csr_matrix((data, (rowi, coli)), shape=(2 * len_rhs, (nT + nT_add) * n_coefs))
    print('matrix (G) done')

    return G


def LCPCorrection(res, Ninv, nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, nT, ndays, linear):

    tint = int(nT / ndays)
    iterations = 1
    proceed = True        

    while proceed:

        colat = np.arange(5, 180, 10 - 0.75 * iterations)
        mlt = np.arange(0., 360, 20 - 1.5 * iterations)
        mlt_m, colat_m  = np.meshgrid(mlt, colat)

        nT_add = 1 if linear else 0

        if ndays==3:
            mlt_m = np.tile(mlt_m.flatten(), tint + nT_add)
            colat_m = np.tile(colat_m.flatten(), tint  + nT_add)
            time_m = np.array([int(_ / (len(colat) * len(mlt))) for _ in range(len(colat) * len(mlt) * tint, len(colat) * len(mlt) * (2 * tint + nT_add),1)])

        if ndays==1:
            mlt_m = np.tile(mlt_m.flatten(), nT + nT_add)
            colat_m = np.tile(colat_m.flatten(), nT + nT_add)
            time_m = np.array([int(_ / (len(colat) * len(mlt))) for _ in range(len(colat) * len(mlt) * (nT + nT_add))])

        G = constructG(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, nT, np.deg2rad(mlt_m), np.deg2rad(colat_m), time_m, linear)
        iterations += 1

        w = G.dot(res)
        idx = (w<0)
        print(idx)
 
        if np.any(idx):
                            
            Gnew = G[idx,:]
            wnew = w[idx]
   
            print('constructing M')

            NGT = Ninv * Gnew.transpose()
            M = Gnew.dot(NGT)

            print('solving LCP')

            sol = lemkelcp(M,wnew,1000000)
        
            print(sol[2])
            try:        
                res = res + NGT.dot(sol[0])
                print ('lcp adjustment done')
            except:
                print('no lcp solution found')
                proceed = False


        if (iterations<=10) and (proceed == True):
            proceed = True
        else:
            proceed = False 

        print (iterations)
    return res


def solve_all(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, tint, sigma0, sigma_v, data, gigs=2, lcp=True, nworkers=1, linear=True):
    chunk_size = GB_CHUNK * gigs

    time = data['time']
    mlt1 = data['mlt1']
    mcolat1 = data['mcolat1']
    mlt2 = data['mlt2']
    mcolat2 = data['mcolat2']
    el = data['el']
    time_ref = data['time_ref']
    mlt1_ref = data['mlt1_ref']
    mcolat1_ref = data['mcolat1_ref']
    mlt2_ref = data['mlt2_ref']
    mcolat2_ref = data['mcolat2_ref']
    el_ref = data['el_ref']
    rhs = data['rhs']
    IPPh_layer1 = data['IPPh_layer1']
    IPPh_layer2 = data['IPPh_layer2']

    ndays = np.ceil((np.max(time) - np.min(time)) / 86400).astype('int') # number of days in data
    nT = tint * ndays  # number of intervals for all time period      

    nchunks = np.int(len(rhs) / chunk_size) # set chuncks size to fit in memory ~4Gb
    nchunks = 1 if nchunks < 1 else nchunks

    print('start, nbig_l1=%s, mbig_l1=%s, nbig_l2=%s, mbig_l2=%s, nT=%s, ndays=%s, sigma0=%s, sigma_v=%s, number of observations=%s, number of chuncks=%s' % (nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, nT, ndays, sigma0, sigma_v, len(rhs), nchunks))


    # split data into chunks
    time_chunks = np.array_split(time, nchunks)
    mlt1_chunks = np.array_split(mlt1, nchunks)
    mcolat1_chunks = np.array_split(mcolat1, nchunks)
    mlt2_chunks = np.array_split(mlt2, nchunks)
    mcolat2_chunks = np.array_split(mcolat2, nchunks)
    el_chunks = np.array_split(el, nchunks)
    time_ref_chunks = np.array_split(time_ref, nchunks)
    mlt1_ref_chunks = np.array_split(mlt1_ref, nchunks)
    mcolat1_ref_chunks = np.array_split(mcolat1_ref, nchunks)
    mlt2_ref_chunks = np.array_split(mlt2_ref, nchunks)
    mcolat2_ref_chunks = np.array_split(mcolat2_ref, nchunks)
    el_ref_chunks = np.array_split(el_ref, nchunks)
    rhs_chunks = np.array_split(rhs, nchunks)


    if (nworkers > 1):
        res, disp_scale, Ninv = stack_constrain_solve_ns(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, nT, ndays, sigma0, sigma_v, 
                                                        time_chunks, mlt1_chunks, mcolat1_chunks, mlt2_chunks, mcolat2_chunks, el_chunks, 
                                                        time_ref_chunks, mlt1_ref_chunks, mcolat1_ref_chunks, mlt2_ref_chunks, mcolat2_ref_chunks, el_ref_chunks, rhs_chunks, nworkers, linear)


    if (nworkers == 1):
        res, disp_scale, Ninv = stack_constrain_solve_ns_np(nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, IPPh_layer1, IPPh_layer2, nT, ndays, sigma0, sigma_v, 
                                                        time_chunks, mlt1_chunks, mcolat1_chunks, mlt2_chunks, mcolat2_chunks, el_chunks, 
                                                        time_ref_chunks, mlt1_ref_chunks, mcolat1_ref_chunks, mlt2_ref_chunks, mcolat2_ref_chunks, el_ref_chunks, rhs_chunks, linear)



    if lcp:
        res = LCPCorrection(res, Ninv, nbig_layer1, mbig_layer1, nbig_layer2, mbig_layer2, nT, ndays, linear) 


    return res, disp_scale, Ninv


