#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:11:07 2020

@author: hanrach
"""
from settings import R, Tref
from jax import config
import jax.numpy as np
import numpy as onp
from jax.scipy.interpolate import RegularGridInterpolator
config.update("jax_enable_x64", True)


#FUNCTIONS TO RESCALE COEFFICIENTS
def solidConductCoeff(sigma,eps):
    # sig_eff = sigma*(1-eps-epsf)
    sig_eff = sigma
    return sig_eff

def electrolyteConductCoeff_delT(eps,brugg,u,T):
    kap_eff = 2*(eps**brugg)*1e-4*u*(-10.5 + 0.668*1e-3*u + 0.494*1e-6*u**2 + \
                                 (0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2)*T + \
                                 (-6.96*1e-5 + 2.8*1e-8*u)*T**2)*(  (0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2) + 2*(-6.96*1e-5 + 2.8*1e-8*u)*T) 
    return kap_eff

def solidDiffCoeff(Ds,T):
    Ds_eff = Ds #*np.exp( -(EqD/R)*( (1/T) - (1/Tref) ) )
    return Ds_eff

##ELECTROLYTE CONDUCTIVITY COEFFICIENT
def electrolyteConductCoeff_example(kap, eps,brugg,u,T):
    kap_eff = (eps**brugg)*1e-4*u*(-10.5 + 0.668*1e-3*u + 0.494*1e-6*u**2 + \
                                 (0.074 - 1.78*1e-5*u - 8.86*1e-10*u**2)*T + \
                                 (-6.96*1e-5 + 2.8*1e-8*u)*T**2)**2
    return kap_eff

def electrolyteConductCoeff_LFP(kap, eps,brugg,u,T):
    if kap==1:
        u = u / 1000          # mol L⁻¹
        kap = 0.1297 * u**3 - 2.51 * u**1.5 + 3.329 * u
    return kap*eps**brugg

def electrolyteConductCoeff_NMC_622(kap, eps,brugg,u,T):
    return electrolyteConductCoeff_LFP(kap, eps,brugg,u,T)

def electrolyteConductCoeff_NMC_532(kap, eps,brugg,u,T):
    return 1.3*eps**brugg#electrolyteConductCoeff_LFP(eps,brugg,u,T)

def electrolyteConductCoeff_NMC_release(kap, eps,brugg,u,T):
    if kap==1:
        u_ref = 1530
        k_ref = 0.02
        return k_ref*u/u_ref*eps**brugg
    else:
        return kap*eps**brugg
#EC:EMC 3:7
def electrolyteConductCoeff_NMC_811(kap, eps,brugg,u,T):
    if kap==1:
        u = u / 1000
        kap = 0.1297 * u**3 - 2.51 * u**1.5 + 3.329 * u
    return kap*eps**brugg

#EC:DEC 1:1
def electrolyteConductCoeff_EC_DEC_1_1(kap, eps,brugg,u,T):
    if kap==1:
        u = u / 1000
        kap = 0.1147 * u**3 - 2.238 * u**1.5 + 2.915 * u
    return kap*eps**brugg

##ELECTROLYTE DIFFUSION COEFFICIENTS
def electrolyteDiffCoeff_example(D, eps,brugg,u,T):
    if D==1:
        D_eff = (eps**brugg)*1e-4*(10**(-4.43 - 54/(T - 229 - 5*1e-3*u) - 0.22*1e-3*u))
    else:
        D_eff = D*eps**brugg
    return D_eff

def electrolyteDiffCoeff_LFP(D, eps, brugg, u, T):
    if D==1:
        D = 2e-10**eps**brugg
    else:
        D = D*eps**brugg
    return D

def electrolyteDiffCoeff_NMC_622(D, eps,brugg,u,T):
    if D==1:
        u =u / 1000  # приведение к диапазону 0-1.4 моль/л, u-концентрация электролита
        D = eps**brugg*(8.794e-11 * u**2 - 3.972e-10 * u + 4.862e-10)
    else:
        D = D*eps**brugg
    return D

def electrolyteDiffCoeff_NMC_532(D, eps,brugg,u,T):
    if D==1:
        u =u / 1000  # приведение к диапазону 0-1.4 моль/л
        D = eps**brugg*(8.794e-11 * u**2 - 3.972e-10 * u + 4.862e-10)
    else:
        D = D*eps**brugg
    return 5.2e-10#D

def electrolyteDiffCoeff_NMC_release(D, eps,brugg,u,T):
    if D==1:
        D = 16e-13*eps**brugg #/1.8
    else:
        D = D*eps**brugg
    return D

#EC:EMC 3:7
def electrolyteDiffCoeff_NMC_811(D, eps,brugg,u,T):
    if D==1:
        u =u / 1000
        D = 8.794e-11 * u**2 - 3.972e-10 * u + 4.862e-10
        D = eps**brugg*D*activity_EC_EMC_3_7(u)
    else:
        D = D*eps**brugg
    return D

#EC:DEC 1:1
def electrolyteDiffCoeff_EC_DEC_1_1(D, eps,brugg,u,T):
    if D==1:
        u =u / 1000
        D = 7.588e-11 * u**2 - 3.036e-10 * u + 3.654e-10
        D = eps**brugg*D
    else:
        D = D*eps**brugg
    return D

##TRANSFERENCE NUMBERS OF ELECTROLYTE

def transfer_func_norm(trans, u):
    return trans

#EC:EMC 3:7
def transfer_func_EC_EMC_3_7(trans, u):
    if trans==1:
        u = u/1000
        trans = -0.1287*u**3 +  0.4106*u**2 - 0.4717*u + 0.4492
    return trans

#EC:DEC 1:1
def transfer_func_EC_DEC_1_1(trans, u):
    if trans==1:
        u = u/1000
        trans = -0.1291*u**3 +  0.3517*u**2 - 0.4893*u + 0.4287
    return trans

#ACTIVITIES

#EC:EMC 3:7
def activity_EC_EMC_3_7(u):
    u = u/1000
    Y = (0.28687*u**2 - 0.74678*u + 0.44103)/(1-transfer_func_EC_EMC_3_7(1, u*1e3))
    return Y

##OCVs
def open_circ_poten_example(cs,cmax):
    theta = cs/cmax;
    ans = (-4.656 + 88.669*(theta**2) - 401.119*(theta**4) + 342.909*(theta**6) -  462.471*(theta**8) + 433.434*(theta**10))/\
    (-1 + 18.933*(theta**2) - 79.532*(theta**4) + 37.311*(theta**6) - 73.083*(theta**8) + 95.96*(theta**10))    
    return ans

def open_circ_poten_LFP0(cs,cmax):
    arr = onp.loadtxt('python\\p2d_fast_solver-main\\all_in\\OCV-DoD_LFP.csv')
    soc = arr[:,0]
    U = arr[:, 1]
    soc = np.array(soc)
    U = np.array(U)
    print((soc))
    func = RegularGridInterpolator((soc, ), U, method='linear')
    # print(type(cs))
    theta = cs/cmax
    xi = theta[..., None]
    ans = func(xi)
    return ans.squeeze(-1) 

def open_circ_poten_LFP1(cs,cmax):
    soc = cs/cmax
    c1 = -150 * soc
    c2 = -30  * (1 - soc)

    U = (3.4077 - 0.020269 * soc + 0.5  * np.exp(c1) - 0.9  * np.exp(c2))
    return U
def open_circ_poten_NMC_532(cs, cmax):
    soc = cs/cmax
    U = (
    -0.8090 * soc
    + 4.4875
    - 0.0428 * np.tanh(18.5138 * (soc - 0.5542))
    - 17.7326 * np.tanh(15.7890 * (soc - 0.3117))
    + 17.5842 * np.tanh(15.9308 * (soc - 0.3120)))
    return U

def open_circ_poten_NMC_release(cs,cmax):
    arr = onp.loadtxt('python\\p2d_fast_solver-main\\all_in\\OCV-DoD_NMC_3.0V.csv')
    soc = arr[:,0]
    U = arr[:, 1]
    soc = np.array(soc)
    U = np.array(U)
    # func = RegularGridInterpolator((soc, ), U, method='linear')
    # # print(type(cs))
    # theta = cs/cmax
    # xi = theta[..., None]
    # ans = func(xi)
    # return ans.squeeze(-1) 
    AM_dens = 4780 # kg/m3; active material density
    AM_molw = 0.097281
    Lix_ref = 0.5
    Lix_soc0 = 0.10 # [1]; Li content in AM (SoC = 0%)
    Lix_soc1 = 0.95 # [1]; Li content in AM (SoC = 100%)
    c_ssoc0 = AM_dens/AM_molw*Lix_soc0
    c_ssoc1 = AM_dens/AM_molw*Lix_soc1
    theta=(cs-c_ssoc0)/(c_ssoc1-c_ssoc0)
    # theta = cs / cmax

    # 3) линейная 1D‑интерполяция
    return np.interp(theta, soc, U)

def open_circ_poten_NMC_811(cs, cmax):
    soc = cs/cmax
    u_eq = (
        -0.8090 * soc
        + 4.4875
        - 0.0428  * np.tanh(18.5138 * (soc - 0.5542))
        - 17.7326 * np.tanh(15.7890 * (soc - 0.3117))
        + 17.5842 * np.tanh(15.9308 * (soc - 0.3120))
    )
    return u_eq

electrolyteDiffCoeffs = {'example': electrolyteDiffCoeff_example, 'LFP': electrolyteDiffCoeff_LFP, 'NMC_622':electrolyteDiffCoeff_NMC_622, 'NMC_532':electrolyteDiffCoeff_NMC_532, 'NMC_release':electrolyteDiffCoeff_NMC_release, 'NMC_811':electrolyteDiffCoeff_NMC_811, 'LiPF6_EC:EMC_3:7':electrolyteDiffCoeff_NMC_811, 'LiPF6_EC:DEC_1:1': electrolyteDiffCoeff_EC_DEC_1_1}
electrolyteConductCoeffs = {'example': electrolyteConductCoeff_example, 'LFP': electrolyteConductCoeff_LFP, 'NMC_622': electrolyteConductCoeff_NMC_622, 'NMC_532': electrolyteConductCoeff_NMC_532, 'NMC_release': electrolyteConductCoeff_NMC_release, 'NMC_811': electrolyteConductCoeff_NMC_811, 'LiPF6_EC:EMC_3:7': electrolyteConductCoeff_NMC_811, 'LiPF6_EC:DEC_1:1': electrolyteConductCoeff_EC_DEC_1_1}
open_circuits = {'example': open_circ_poten_example, 'LFP': open_circ_poten_LFP1, 'NMC_622': open_circ_poten_LFP1, 'NMC_532': open_circ_poten_NMC_532, 'NMC_release': open_circ_poten_NMC_release, 'NMC_811': open_circ_poten_NMC_811, 'NMC_811_tuned': open_circ_poten_NMC_811}
transference_numbers = {'example': transfer_func_norm, 'LFP': transfer_func_norm, 'NMC_622': transfer_func_norm, 'NMC_532': transfer_func_norm, 'NMC_release':transfer_func_norm, 'NMC_811':transfer_func_EC_EMC_3_7, 'LiPF6_EC:EMC_3:7': transfer_func_EC_EMC_3_7, 'LiPF6_EC:DEC_1:1': transfer_func_EC_DEC_1_1}