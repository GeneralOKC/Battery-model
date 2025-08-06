#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 12:50:15 2020

@author: hanrach
"""

# import coeffs as coeffs
from coeffs import electrolyteDiffCoeffs, electrolyteConductCoeffs, solidConductCoeff, solidDiffCoeff, open_circuits, transference_numbers, activity_EC_EMC_3_7
import jax.numpy as np
from settings import F, R, Tref #gamma, trans, 
from jax import config
config.update("jax_enable_x64", True)
from functools import partial
import jax
import os
import numpy as onp
from jax.scipy.interpolate import RegularGridInterpolator
# from scipy.interpolate import interp1d
os.chdir("D:\\")

class ElectrodeEquation:   
    
    def __init__(self, constants, gridparam, electrode_type, s_constants, s_grid, delta_t, T, cathode, electrolyte):
        self.c_init = constants.c_init
        self.cmax = constants.cmax
        self.electrode_type = electrode_type
        self.sigma = constants.sigma
        self.eps = constants.eps
        self.brugg = constants.brugg
        self.a = constants.a
        self.Rp = constants.Rp
        self.k = constants.k
        self.l = constants.l
        self.Ds = constants.Ds;
        #self.sigma = constants.sigma
        #self.Ds = constants.Ds
        self.kap = s_constants.kap
        self.delta_t=delta_t
        self.sep = s_constants
        self.sep_hx = self.sep.l/s_grid.M
        self.T = T
        self.trans = s_constants.trans
        self.D = s_constants.D
        self.u_initial = s_constants.u_initial
        self.gamma = 2*(1-self.trans)*8.31/F
        self.N = gridparam.N; self.M = gridparam.M
        N = self.N; M = self.M;
        self.hx = self.l/M; self.hy = self.Rp/N


        self.electrolyteDiffCoeff, self.electrolyteConductCoeff, self.OCV = electrolyteDiffCoeffs[electrolyte], electrolyteConductCoeffs[electrolyte], open_circuits[cathode]
        self.transfer_func = transference_numbers[electrolyte]
        self.activity = activity_EC_EMC_3_7
        # self.sigeff  = self.sigma*(1-self.eps - self.epsf)
        sigeff = solidConductCoeff(self.sigma, self.eps)
        
        self.rpts_end = np.arange(0,N)*self.Rp/N
        self.rpts_mid = self.Rp*(np.arange(0,N)-0.5)/N
        
        self.rpts_end = self.Rp*np.linspace(0,N, N+1)/N
        self.rpts_mid = self.Rp*(np.linspace(1,N, N)-0.5)/N
        R = self.Rp*(np.linspace(1,N,N)-(1/2))/N;
#        R = np.arange(0,self.Rp + self.hy, self.hy) + self.hy/2 ;
#        R = R[0:-1]
        r = self.Rp*(np.linspace(0,N, N+1))/N
#        r = np.arange(0,self.Rp + self.hy, self.hy) + self.hy; 
#        self.rpts_end = self.Rp*np.linspace(0,N, N+1)/N
#        self.rpts_mid = self.Rp*(np.linspace(1,N, N)-0.5)/N
#        lambda1 = k*r[0:N]**2/R**2/hy**2;
#        lambda2 = k*r[1:N+1]**2/R**2/hy**2;
        self.lambda1 = delta_t*r[0:N]**2/(R**2*self.hy**2);
        self.lambda2 = delta_t*r[1:N+1]**2/(R**2*self.hy**2);
    
    """ Equations for c"""
    def solid_conc(self,cn,cc,cp, cold, lambda1, lambda2):
        Ds = self.Ds
        ans = (cc-cold)  + Ds*( cc*(lambda2 + lambda1) - lambda2*cp - lambda1*cn)
        return ans
    


    def solid_conc_2(self,cn,cc,cp, cold):
        hy = self.hy
        Ds = self.Ds
        
        lambda1  = self.delta_t*self.rpts_end[0:self.N]**2/self.rpts_mid**2/hy**2;
        lambda2  = self.delta_t*self.rpts_end[1:self.N+1]**2/self.rpts_mid**2/hy**2;
        
        k = self.delta_t;
        N = self.N
        R = self.Rp*(np.linspace(1,N,N)-(1/2))/N;
#        R = np.arange(0,self.Rp + self.hy, self.hy) + self.hy/2 ;
#        R = R[0:-1]
        r = self.Rp*(np.linspace(0,N, N+1))/N
#        r = np.arange(0,self.Rp + self.hy, self.hy) + self.hy; 
#        self.rpts_end = self.Rp*np.linspace(0,N, N+1)/N
#        self.rpts_mid = self.Rp*(np.linspace(1,N, N)-0.5)/N
#        lambda1 = k*r[0:N]**2/R**2/hy**2;
#        lambda2 = k*r[1:N+1]**2/R**2/hy**2;
        lambda1 = k*r[0:N]**2/(R**2*hy**2);
        lambda2 = k*r[1:N+1]**2/(R**2*hy**2);
#(c1 - cold) + c1*(lambda2 + lambda1) - lambda2*c2 - lambda1*c0
        ans = (cc-cold)  + Ds*( cc*(lambda2 + lambda1) - lambda2*cp - lambda1*cn)
        return ans
    
    def bc_neumann_c(self,c0,c1,jvec):
        hy = self.hy
        Deff = solidDiffCoeff(self.Ds, self.T)
        bc = (c1 - c0)/hy + jvec/Deff
#        bc = (c1-c0) + jvec*hy/Deff
        return bc.reshape()
    
    """ Equations for u """
    @partial(jax.jit, static_argnums=(0,))
    def electrolyte_conc(self,un, uc, up, j,uold, phien, phiec, phiep):
        eps = self.eps; brugg = self.brugg; hx = self.hx
        a = self.a
        T = self.T
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        # Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        Deff_r = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_r,self.T);
        Deff_l = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_l,self.T);
        
        kapeff = self.electrolyteConductCoeff(self.kap, eps,brugg, (umid_r+umid_l)/2,self.T)
        # ans = (uc-uold) - (self.delta_t/eps)*( ( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx + a*(1-self.trans)*j ) #old but good
        phie_mid_r = (phiep+phiec)/2
        phie_mid_l = (phiec+phien)/2
        trans_c = self.transfer_func(self.trans, (umid_r+umid_l)/2)
        trans_l = self.transfer_func(self.trans, un)
        trans_r = self.transfer_func(self.trans, up)
        Deff_c = self.electrolyteDiffCoeff(self.D, eps,brugg,(umid_r+umid_l)/2,self.T)
        # ans = (uc-uold) - (self.delta_t/eps)*( ( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx -(-kapeff*(phie_mid_r-phie_mid_l)/hx + Deff_c*F*(umid_r-umid_l)/hx/trans_c)*(trans_r-trans_l)/hx/F + a*(1-trans_c)*j )

        term = kapeff*R*T/F*self.activity(uc)*(1-trans_c)*(np.log(umid_r)-np.log(umid_l))/hx
        ans = (uc-uold) - (self.delta_t/eps)*( ( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx -(-kapeff*(phie_mid_r-phie_mid_l)/hx + term)*(trans_r-trans_l)/hx/F + a*(1-trans_c)*j )
        return ans.reshape()
    
    def bc_zero_neumann(self,u0, u1):
        bc =  u1 - u0
        return bc.reshape()

    def bc_const_dirichlet(self,u0, u1, constant):
        bc =  (u1 + u0)/2 - constant
        return bc.reshape()
    
    # boundary condition for positive electrode
    def bc_u_sep_p(self,u0_pe,u1_pe,\
             u0_sep,u1_sep):
        
        eps_p = self.eps; eps_s = self.sep.eps;
        brugg_p = self.brugg; brugg_s = self.sep.brugg;
        
#        Deff_pe = coeffs.electrolyteDiffCoeff(eps_p,brugg_p,(u0_pe + u1_pe)/2,(T0_pe + T1_pe)/2)
#        Deff_sep = coeffs.electrolyteDiffCoeff(eps_s,brugg_s,(u0_sep + u1_sep)/2,(T0_sep + T1_sep)/2)
        
        Deff_pe = (self.electrolyteDiffCoeff(self.D, eps_p,brugg_p,u0_pe,self.T) + self.electrolyteDiffCoeff(self.D, eps_p,brugg_p,u1_pe,self.T))/2
        Deff_sep =( self.electrolyteDiffCoeff(self.D, eps_s,brugg_s,u0_sep,self.T) + self.electrolyteDiffCoeff(self.D, eps_s,brugg_s,u1_sep,self.T))/2          
        
        bc = -Deff_pe*(u1_pe - u0_pe)/self.hx + Deff_sep*(u1_sep - u0_sep)/self.sep_hx
#        bc = -Deff_pe*(u1_pe - u0_pe)*sep.hx + Deff_sep*(u1_sep - u0_sep)*pe.hx
        return bc.reshape()
    
    def bc_inter_cont(self, u0_pe, u1_pe, u0_sep, u1_sep):
        ans = (u0_pe+u1_pe)/2 - (u0_sep+u1_sep)/2
        # ans = u1_pe - u0_sep
        return ans.reshape()

    """ Electrolyte potential equations: phie """
    
    def electrolyte_poten(self,un, uc, up, phien, phiec, phiep, j):
    
        eps = self.eps; brugg = self.brugg;
        hx = self.hx; 
        a = self.a;
        
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        # Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        
        kapeff_r = self.electrolyteConductCoeff(self.kap, eps,brugg,umid_r,self.T);
        kapeff_l = self.electrolyteConductCoeff(self.kap, eps,brugg,umid_l,self.T);
        D_r = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_r,self.T)
        D_l = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_l,self.T)
        T = self.T
        # ans = a*F*j + (kapeff_r*(phiep - phiec)/hx \
        #             - kapeff_l*(phiec - phien)/hx)/hx \
        # - self.gamma*(kapeff_r*T*(np.log(up) - \
        #                                        np.log(uc))/hx  \
        #     - kapeff_l*T*(np.log(uc) - \
        #                                        np.log(un))/hx )/hx

        # ans = a*F*j + (kapeff_r*(phiep - phiec)/hx- kapeff_l*(phiec - phien)/hx)/hx - ((D_r*F/self.transfer_func(self.trans, umid_r)*(up-uc)/hx)-D_l*F/self.transfer_func(self.trans, umid_l)*(uc-un)/hx)/hx
        trans_c = self.transfer_func(self.trans, uc)
        trans_l = self.transfer_func(self.trans, umid_l)
        trans_r = self.transfer_func(self.trans, umid_r)
        # ans = a*F*j + (kapeff_r*(phiep - phiec)/hx- kapeff_l*(phiec - phien)/hx)/hx - ((D_r*F/trans_r*(up-uc)/hx)-D_l*F/trans_l*(uc-un)/hx)/hx

        term_l = kapeff_l*R*T/F*self.activity(umid_l)*(1-trans_l)*(np.log(uc)-np.log(un))/hx
        term_r = kapeff_r*R*T/F*self.activity(umid_r)*(1-trans_r)*(np.log(up)-np.log(uc))/hx
        ans = a*F*j + (kapeff_r*(phiep - phiec)/hx- kapeff_l*(phiec - phien)/hx)/hx - (term_r-term_l)/hx
        return ans.reshape()

    def bc_zero_dirichlet(self,phie0, phie1):
        ans= (phie0 + phie1)/2
        return ans.reshape()

    def bc_phie_p(self,phie0_p, phie1_p, phie0_s, phie1_s, u0_p, u1_p, u0_s, u1_s):
        T = self.T
        kapeff_p = self.electrolyteConductCoeff(self.kap, self.eps,self.brugg,(u0_p + u1_p)/2,T);
        kapeff_s = self.electrolyteConductCoeff(self.kap, self.sep.eps,self.sep.brugg,(u0_s + u1_s)/2,T);
        
        bc = -kapeff_p*(phie1_p - phie0_p)/self.hx + kapeff_s*(phie1_s - phie0_s)/self.sep_hx
        return bc.reshape()
    
    """ Equations for solid potential phis"""
    def solid_poten(self,phisn, phisc, phisp, j):
        hx = self.hx; a = self.a
        # sigeff = self.sigma*(1-self.eps-self.epsf)
        sigeff = solidConductCoeff(self.sigma, self.eps)
        ans = ( phisn - 2*phisc + phisp) - (a*F*j*hx**2)/sigeff
        return ans.reshape()
    
    def bc_phis(self,phis0, phis1, source):
        # sigeff = self.sigma*(1-self.eps-self.epsf)
        sigeff = solidConductCoeff(self.sigma, self.eps)
        bc = ( phis1 - phis0 ) + self.hx*(source)/sigeff
        return bc.reshape()
    
    def ionic_flux_fast(self,j,u,eta,cs1, gamma_c,cmax):
        T = self.T
        cs = cs1 - gamma_c*j/solidDiffCoeff(self.Ds, T)
        keff = self.k# *np.exp( (-self.Ek/R)*((1/T) - (1/Tref)))
        var = ((0.5*F)/(R*T))*eta
        term2 = (np.exp(var)-np.exp(-var))/2
        ans = j - 2*keff*np.sqrt(u*(cmax - cs)*cs)*term2
    #        ans = j - 2*keff*np.sqrt(u*(cmax - cs)*cs)*np.sinh( (0.5*F/(R*T))*(eta) )
        return ans.reshape()

    # def open_circ_poten_ref_fast_LFP0(self,cs,cmax):
    #     arr = onp.loadtxt('python\\p2d_fast_solver-main\\all_in\\OCV-DoD_LFP.csv')
    #     soc = arr[:,0]
    #     U = arr[:, 1]
    #     soc = np.array(soc)
    #     U = np.array(U)
    #     print((soc))
    #     func = RegularGridInterpolator((soc, ), U, method='linear')
    #     # print(type(cs))
    #     theta = cs/cmax
    #     xi = theta[..., None]
    #     ans = func(xi)
    #     return ans.squeeze(-1) 
    
    # def open_circ_poten_ref_fast_LFP1(self,cs,cmax):
    #     soc = cs/cmax
    #     c1 = -150 * soc
    #     c2 = -30  * (1 - soc)

    #     U = (3.4077 - 0.020269 * soc + 0.5  * np.exp(c1) - 0.9  * np.exp(c2))
    #     return U
    # def open_circ_poten_ref_fast_NMC_532(self, cs, cmax):
    #      soc = cs/cmax
    #      U = (
    #     -0.8090 * soc
    #     + 4.4875
    #     - 0.0428 * np.tanh(18.5138 * (soc - 0.5542))
    #     - 17.7326 * np.tanh(15.7890 * (soc - 0.3117))
    #     + 17.5842 * np.tanh(15.9308 * (soc - 0.3120)))
    #      return U
    
    def over_poten_fast(self,eta, phis,phie, j,cs1, gamma_c, cmax):
        T = self.T
        cs = cs1 - gamma_c*j/solidDiffCoeff(self.Ds, T)
        ans = eta - phis + phie + self.OCV(cs,cmax);
        return ans.reshape()
    
    

