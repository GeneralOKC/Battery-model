#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:08:59 2020

@author: hanrach
"""
from functools import partial
from jax.scipy.linalg import solve
import jax.numpy as np
import numpy as onp
from jax.numpy.linalg import norm
from scipy.sparse import csr_matrix, csc_matrix
# from scikits.umfpack import spsolve
from scipy.sparse.linalg import spsolve
import os
import timeit
from jax import device_get
os.chdir("D:\\")





def newton_fast_sparse(fn_fast, jac_fn_fast, U, cs_pe1, gamma_p, Icell, T):
    maxit=500
    tol = 1e-6
    res = 100
    fail = 0
    Uold = U
    count = 0
    start = timeit.default_timer()
    J = jac_fn_fast(U, Uold, cs_pe1, gamma_p, Icell, T).block_until_ready()
    y = fn_fast(U,Uold,cs_pe1, gamma_p,  Icell, T).block_until_ready()
    end = timeit.default_timer();
    jf_time0 = end-start
    # print('det', onp.linalg.det(J))
    start=timeit.default_timer()
    Jsparse=csr_matrix(J)
    
    # print(np.linalg.det(Jsparse))
    end=timeit.default_timer()
    overhead0= end-start
    
    start=timeit.default_timer()
    delta = spsolve(Jsparse,y)
    end = timeit.default_timer()
    solve0 = end-start;
    U = U - delta;
    res0 = norm(y/norm(U,np.inf),np.inf)
    # print(count, res0)

    count = 1
    solve_time = solve0
    overhead = overhead0
    jf_time=jf_time0
    while(count < maxit and res > tol):
                 
        start=timeit.default_timer()
        J = jac_fn_fast(U,Uold, cs_pe1, gamma_p, Icell, T).block_until_ready()
        y = fn_fast(U,Uold, cs_pe1, gamma_p, Icell, T).block_until_ready()
        end=timeit.default_timer()
        jf_time += end-start
        # print('det', onp.linalg.det(J))


        start=timeit.default_timer()
        Jsparse=csr_matrix(J)

        # print(J)
        # zero_rows = onp.where(Jsparse.getnnz(axis=1) == 0)[0]
        # zero_cols = onp.where(Jsparse.getnnz(axis=0) == 0)[0]
        # print("zero rows:", zero_rows, "zero cols:", zero_cols)
        # onp.savetxt('python\\p2d_fast_solver-main\\all_in\\not_nan.txt',  device_get(J), delimiter=' ')

        end = timeit.default_timer()
        overhead += end-start
        
        res = norm(y/norm(U,np.inf),np.inf)
        
        start=timeit.default_timer()
#        delta = onp.linalg.solve(J,y)
        delta = spsolve(Jsparse,y)
        end= timeit.default_timer()
        solve_time += end-start;
        # print(count, res)

        U = U - delta
        count = count + 1


    if fail ==0 and np.any(np.isnan(delta)):
        fail = 1
        print("nan solution")
        
    if fail == 0 and max(abs(np.imag(delta))) > 0:
        fail = 1
        print("solution complex")
    
    if fail == 0 and res > tol:
        fail = 1;
        print('Newton fail: no convergence')
    else:
        fail == 0 
        
    info=(fail, solve_time, overhead, jf_time)
    return U, info




