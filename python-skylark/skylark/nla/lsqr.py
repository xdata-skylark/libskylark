from math import sqrt, log

import numpy as np
from numpy.linalg import norm, lstsq

import scipy as sp
from scipy.sparse.linalg import aslinearoperator

colnorm = lambda X: np.sum(np.abs(X)**2, axis=0)**(1./2)

def lsqr( A, B, tol=1e-14, iter_lim=None ):
    """
    A simple version of LSQR for solving || AX - B ||^2_fro
    """

    A    = aslinearoperator(A)
    m, n = A.shape
    m, k = B.shape
    
    eps  = 32*np.finfo(float).eps;      # slightly larger than eps

    if tol < eps:
        tol = eps
    elif tol >= 1:
        tol = 1-eps

    U    = B.squeeze().copy()
    beta = colnorm(U)
    ibeta = beta.copy()
    ibeta[np.nonzero(beta)] = 1./beta[np.nonzero(beta)] 
    U = np.dot(U, np.diag(ibeta))
    
    #if beta != 0:
    #    u   /= beta
 
    V     = A.rmatvec(U)
    alpha = colnorm(V)
    ialpha = alpha.copy()
    ialpha[np.nonzero(alpha)] = 1./alpha[np.nonzero(alpha)] 
    V = np.dot(V, np.diag(ialpha))
    
    #if alpha != 0:
    #    v    /= alpha

    W     = V.copy()

    X     = np.zeros((n, k))

    phibar = beta
    rhobar = alpha

    nrm_a    = np.zeros(k)
    cnd_a    = np.zeros(k)
    sq_d     = np.zeros(k)
    nrm_r    = beta
    nrm_ar_0 = alpha*beta

    if all(nrm_ar_0 == 0):                     # alpha == 0 || beta == 0
        return X, 0, 0

    nrm_x  = np.zeros(k)
    sq_x   = np.zeros(k)
    z      = np.zeros(k)
    cs2    = -1*np.ones(k)
    sn2    = np.zeros(k)

    max_n_stag = 3
    stag       = 0

    flag = -1
    if iter_lim is None:
        iter_lim = np.max( [20, 2*np.min([m,n])] )

    for itn in xrange(int(iter_lim)):

        U    = A.matvec(V) - np.dot(U, np.diag(alpha))
        beta = colnorm(U)
        U   = np.dot(U, np.diag(1./beta));
        
        # estimate of norm(A)
        nrm_a = np.sqrt(nrm_a**2 + alpha**2 + beta**2)

        V     = A.rmatvec(U) - np.dot(V, np.diag(beta))
        alpha = colnorm(V)
        V = np.dot(V, np.diag(1/alpha))

        rho    =  np.sqrt(rhobar**2+beta**2)
        cs     =  rhobar/rho
        sn     =  beta/rho
        theta  =  sn*alpha
        rhobar = -cs*alpha
        phi    =  cs*phibar
        phibar =  sn*phibar
 
        X     += np.dot(W, np.diag(phi/rho))
        W      = V - np.dot(W, np.diag(theta/rho))

        # estimate of norm(r)
        nrm_r   = phibar

        # estimate of norm(A'*r)
        nrm_ar  = phibar*alpha*np.abs(cs)

        # check convergence
        if all(nrm_ar < tol*nrm_ar_0):
            flag = 0
            break

        if all(nrm_ar < eps*nrm_a*nrm_r):
            flag = 0
            break

        # estimate of cond(A)
        #sq_w    = np.dot(W,W)
        nrm_w   = colnorm(W)
        sq_w    = nrm_w*nrm_w 
        sq_d   += sq_w/(rho**2)
        cnd_a   = nrm_a*np.sqrt(sq_d)

        # check condition number
        if any(cnd_a > 1/eps):
            flag = 1
            break

        # check stagnation
        if any(abs(phi/rho)*nrm_w < eps*nrm_x):
            stag += 1
        else:
            stag  = 0
        if stag >= max_n_stag:
            flag = 1
            break

        # estimate of norm(x)
        delta   =  sn2*rho
        gambar  = -cs2*rho
        rhs     =  phi - delta*z
        zbar    =  rhs*gambar
        nrm_x   =  np.sqrt(sq_x + zbar**2)
        gamma   =  np.sqrt(gambar**2 + theta**2)
        cs2     =  gambar/gamma
        sn2     =  theta/gamma
        z       =  rhs/gamma
        sq_x   +=  z**2

    return X, flag, itn

def lsqr_single_rhs( A, b, tol=1e-14, iter_lim=None ):
    """
    A simple version of LSQR
    """

    A    = aslinearoperator(A)
    m, n = A.shape

    eps  = 32*np.finfo(float).eps;      # slightly larger than eps

    if tol < eps:
        tol = eps
    elif tol >= 1:
        tol = 1-eps

    u    = b.squeeze().copy()
    beta = norm(u)
    if beta != 0:
        u   /= beta

    v     = A.rmatvec(u)
    alpha = norm(v)
    if alpha != 0:
        v    /= alpha

    w     = v.copy()

    x     = np.zeros(n)

    phibar = beta
    rhobar = alpha

    nrm_a    = 0.0
    cnd_a    = 0.0
    sq_d     = 0.0
    nrm_r    = beta
    nrm_ar_0 = alpha*beta

    if nrm_ar_0 == 0:                     # alpha == 0 || beta == 0
        return x, 0, 0

    nrm_x  = 0
    sq_x   = 0
    z      = 0
    cs2    = -1
    sn2    = 0

    max_n_stag = 3
    stag       = 0

    flag = -1
    if iter_lim is None:
        iter_lim = np.max( [20, 2*np.min([m,n])] )

    for itn in xrange(int(iter_lim)):

        u    = A.matvec(v) - alpha*u
        beta = norm(u)
        u   /= beta

        # estimate of norm(A)
        nrm_a = sqrt(nrm_a**2 + alpha**2 + beta**2)

        v     = A.rmatvec(u) - beta*v
        alpha = norm(v)
        v    /= alpha

        rho    =  sqrt(rhobar**2+beta**2)
        cs     =  rhobar/rho
        sn     =  beta/rho
        theta  =  sn*alpha
        rhobar = -cs*alpha
        phi    =  cs*phibar
        phibar =  sn*phibar

        x     += (phi/rho)*w
        w      = v-(theta/rho)*w

        # estimate of norm(r)
        nrm_r   = phibar

        # estimate of norm(A'*r)
        nrm_ar  = phibar*alpha*np.abs(cs)

        # check convergence
        if nrm_ar < tol*nrm_ar_0:
            flag = 0
            break

        if nrm_ar < eps*nrm_a*nrm_r:
            flag = 0
            break

        # estimate of cond(A)
        sq_w    = np.dot(w,w)
        nrm_w   = sqrt(sq_w)
        sq_d   += sq_w/(rho**2)
        cnd_a   = nrm_a*sqrt(sq_d)

        # check condition number
        if cnd_a > 1/eps:
            flag = 1
            break

        # check stagnation
        if abs(phi/rho)*nrm_w < eps*nrm_x:
            stag += 1
        else:
            stag  = 0
        if stag >= max_n_stag:
            flag = 1
            break

        # estimate of norm(x)
        delta   =  sn2*rho
        gambar  = -cs2*rho
        rhs     =  phi - delta*z
        zbar    =  rhs/gambar
        nrm_x   =  sqrt(sq_x + zbar**2)
        gamma   =  sqrt(gambar**2 + theta**2)
        cs2     =  gambar/gamma
        sn2     =  theta /gamma
        z       =  rhs   /gamma
        sq_x   +=  z**2

    return x, flag, itn
       
       
def generate_problem(m, n, k):
    A = np.random.rand(m, n)
    X_opt = np.random.rand(n, k)
    B = np.dot(A, X_opt)
    return A, B, X_opt
  
def _test():

    m = 500
    n = 100
    k = 10
    r = 80
    c = 1e3                            # well-conditioned

    A, B, X_opt = generate_problem( m, n, k )
    
    tol      = 1e-14
    iter_lim = 400 # np.ceil( (log(tol)-log(2.0))/log((c-1.0)/(c+1.0)) )

    X, flag, itn = lsqr(A, B, tol, iter_lim)
    
    relerr       = norm(X-X_opt)/norm(X_opt)
    print relerr, flag
    
    for i in range(0, k):
        x, flag2, itn2 = lsqr_single_rhs(A, B[:, i], tol, iter_lim)
        print i, norm(x - X[:, i])
        
    
    if flag == 0:
        print "LSQR converged in %d iterations." % (itn,)
    else:
        print "LSQR didn't converge in %d iterations." % (itn,)

    if relerr < 1e-10:
        print "LSQR test passed with relerr %G." % (relerr,)
    else:
        print "LSQR test failed with relerr %G." % (relerr,)    

if __name__ == '__main__':
    _test()
    
