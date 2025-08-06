#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 23:21:54 2020

@author: hanrach
"""

from jax import jacfwd, jit
from res_fn_fast import ResidualFunctionFast
import jax.numpy as np
from unpack import unpack_fast

def p2d_init_fast(Np, Mp, Ms, delta_t, T, p_electrode_param, electrolyte_sep_param, cathode, electrolyte):
    solver_fast = ResidualFunctionFast(Mp, Np, Ms,delta_t, T, p_electrode_param, electrolyte_sep_param, cathode, electrolyte)
   
    def fn_fast(U, Uold, cs_pe1, gamma_p, Icell, T):
    #    val = np.zeros(Ntot)
        val= np.zeros(solver_fast.Ntot)
#        U, Uold, cs_pe1, cs_ne1, gamma_p, gamma_n
#        U = arg0["U"]; Uold=arg0["Uold"]; cs_pe1=arg0["cs_pe1"]; cs_ne1=arg0["cs_ne1"]; gamma_p=arg0["gamma_p"]; gamma_n=arg0["gamma_n"]       
#        gamma_p = gamma_p*np.ones(Mp)
#        gamma_n = gamma_n*np.ones(Mn)
        uvec_pe, uvec_sep, phie_pe, phie_sep, phis_pe, jvec_pe,eta_pe = unpack_fast(U, Mp, Np, Ms)
    
        
        uvec_old_pe, uvec_old_sep,\
        _, _, \
        _, _,\
        _= unpack_fast(Uold,Mp, Np, Ms)
        
        
        ''' add direct solve for c'''
    
       
        val = solver_fast.res_u_pe(val, uvec_pe, jvec_pe, uvec_old_pe, uvec_sep, phie_pe)
        val = solver_fast.res_u_sep(val, uvec_sep, uvec_old_sep, uvec_pe, Icell, phie_sep)
        # val = solver_fast.res_u_ne(val, uvec_ne, jvec_ne, uvec_old_ne, uvec_sep)
        
        
        val = solver_fast.res_phie_pe(val, uvec_pe, phie_pe, jvec_pe, uvec_sep,phie_sep)
        val = solver_fast.res_phie_sep(val, uvec_sep, phie_sep, phie_pe, Icell)
        # val = solver_fast.res_phie_ne(val, uvec_ne, phie_ne,jvec_ne, uvec_sep, phie_sep)
    
        val = solver_fast.res_phis(val, phis_pe, jvec_pe, Icell)
    
        val = solver_fast.res_j_fast(val, jvec_pe, uvec_pe, eta_pe, cs_pe1, gamma_p)
        val = solver_fast.res_eta_fast(val, eta_pe, phis_pe, phie_pe, jvec_pe, cs_pe1, gamma_p)
        return val
    
    fn_fast=jit(fn_fast)
    jac_fn_fast=jit(jacfwd(fn_fast))
        
    return fn_fast, jac_fn_fast
    
def set_current(path):
    arr = np.loadtxt(path, skiprows=1)
    I_times = arr[:, 1]
    Iapp = arr[:, 2]
    return I_times, Iapp