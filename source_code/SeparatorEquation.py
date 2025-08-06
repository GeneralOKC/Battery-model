#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  1 15:19:58 2020

@author: hanrach
"""

from coeffs import electrolyteDiffCoeffs, electrolyteConductCoeffs, solidConductCoeff, solidDiffCoeff, transference_numbers, activity_EC_EMC_3_7
import jax.numpy as np
from jax import grad
from settings import F, R
from jax import config
config.update("jax_enable_x64", True)

class SeparatorEquation:
    def __init__(self, constants, gridparam, p_constants, p_grid, delta_t, T, cathode, electrolyte):
        self.eps = constants.eps
        self.brugg = constants.brugg
        self.l = constants.l
        self.delta_t = delta_t
        self.M = gridparam.M
        M = self.M;
        self.hx = self.l/M; 
#        self.hx = 1/M; self.hy=1/N 
        self.kap = constants.kap
        self.pe = p_constants
        self.pe_hx = self.pe.l/p_grid.M
        self.T = T
        self.trans = constants.trans
        self.D = constants.D
        self.gamma = 2*(1-self.trans)*R/F
        self.electrolyteDiffCoeff, self.electrolyteConductCoeff = electrolyteDiffCoeffs[electrolyte], electrolyteConductCoeffs[electrolyte]
        self.transfer_func = transference_numbers[electrolyte]
        self.activity = activity_EC_EMC_3_7
        
    def electrolyte_conc(self,un, uc, up, uold, phien, phiec, phiep):
        eps = self.eps; brugg = self.brugg; hx = self.hx
        T = self.T
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        phie_mid_r = (phiep+phiec)/2
        phie_mid_l = (phiec+phien)/2
        # Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        Deff_r = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_r,T);
        Deff_l = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_l,T);
 
        # ans = (uc-uold) -  (self.delta_t/eps)*( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx 
        kapeff = self.electrolyteConductCoeff(self.kap, eps,brugg,(umid_r+umid_l)/2,self.T)
        trans_c = self.transfer_func(self.trans, (umid_r+umid_l)/2)
        trans_l = self.transfer_func(self.trans, umid_l)
        trans_r = self.transfer_func(self.trans, umid_r)
        Deff_c = self.electrolyteDiffCoeff(self.D, eps,brugg,(umid_r+umid_l)/2,self.T)
        # ans = (uc-uold) - (self.delta_t/eps)*( ( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx -(-kapeff*(phie_mid_r-phie_mid_l)/hx + Deff_c*F*(umid_r-umid_l)/hx/trans_c)*(trans_r-trans_l)/hx/F)

        term = kapeff*R*T/F*self.activity(uc)*(1-trans_c)*(np.log(umid_r)-np.log(umid_l))/hx
        ans = (uc-uold) - (self.delta_t/eps)*( ( Deff_r*(up - uc)/hx - Deff_l*(uc - un)/hx )/hx -(-kapeff*(phie_mid_r-phie_mid_l)/hx + term)*(trans_r-trans_l)/hx/F)
        return ans.reshape()
    
    # boundary condition for positive electrode
    def bc_u_sep_p(self,u0_pe,u1_pe,\
             u0_sep,u1_sep):
        T = self.T
        eps_p = self.pe.eps; eps_s = self.eps;
        brugg_p = self.pe.brugg; brugg_s = self.brugg;
        Deff_pe = self.electrolyteDiffCoeff(self.D, eps_p,brugg_p,(u0_pe + u1_pe)/2,T)
        Deff_sep = self.electrolyteDiffCoeff(self.D, eps_s,brugg_s,(u0_sep + u1_sep)/2,T)
        bc = -Deff_pe*(u1_pe - u0_pe)/self.pe_hx + Deff_sep*(u1_sep - u0_sep)/self.hx
        return bc.reshape()

    # boundary condition for negative electrode
    def bc_u_sep_n(self,u0_ne,u1_ne,\
                 u0_sep,u1_sep):
        eps_n = self.ne.eps; eps_s = self.eps;
        brugg_n = self.ne.brugg; brugg_s = self.ne.brugg;
        T = self.T
        Deff_ne = self.electrolyteDiffCoeff(self.D, eps_n,brugg_n,(u0_ne + u1_ne)/2,T)
        Deff_sep = self.electrolyteDiffCoeff(self.D, eps_s,brugg_s,(u0_sep + u1_sep)/2,T)
        
        bc = -Deff_sep*(u1_sep - u0_sep)/self.hx + Deff_ne*(u1_ne - u0_ne)/self.ne_hx
        return bc.reshape()
    def bc_zero_neumann(self,u0, u1):
        bc =  u1 - u0
        return bc.reshape()
    
    def bc_neumann_new(self, u0, u1, Iapp):
        eps = self.eps; brugg = self.brugg
        umid = (u1+u0)/2
        trans = self.transfer_func(self.trans, umid)
        bc = (u1-u0)-Iapp*(trans-1)/self.electrolyteDiffCoeff(self.D, eps,brugg, (u1+u0)/2,self.T)/F*self.hx #/40
        # print(bc)
        return bc.reshape()
    
    def electrolyte_poten(self,un, uc, up, phien, phiec, phiep):
    
        eps = self.eps; brugg = self.brugg;
        hx = self.hx; 
        T = self.T
        umid_r = (up+uc)/2; umid_l = (un+uc)/2;
        # Tmid_r = (Tp+Tc)/2; Tmid_l = (Tn+Tc)/2;
        
        kapeff_r = self.electrolyteConductCoeff(self.kap, eps,brugg,umid_r,T);
        kapeff_l = self.electrolyteConductCoeff(self.kap, eps,brugg,umid_l,T);
        D_r = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_r,self.T)
        D_l = self.electrolyteDiffCoeff(self.D, eps,brugg,umid_l,self.T)
        # ans = - ( kapeff_r*(phiep - phiec)/hx - kapeff_l*(phiec - phien)/hx )/hx + self.gamma*( kapeff_r*T*(np.log(up) - np.log(uc))/hx  \
        #     - kapeff_l*T*(np.log(uc) - np.log(un))/hx )/hx
        # ans =  (kapeff_r*(phiep - phiec)/hx- kapeff_l*(phiec - phien)/hx)/hx - ((D_r*F/self.transfer_func(self.trans, umid_r)*(up-uc)/hx)-D_l*F/self.transfer_func(self.trans, umid_l)*(uc-un)/hx)/hx
        trans_c = self.transfer_func(self.trans, uc)
        trans_l = self.transfer_func(self.trans, umid_l)
        trans_r = self.transfer_func(self.trans, umid_r)
        # ans = (kapeff_r*(phiep - phiec)/hx- kapeff_l*(phiec - phien)/hx)/hx - ((D_r*F/trans_r*(up-uc)/hx)-D_l*F/trans_l*(uc-un)/hx)/hx

        term_l = kapeff_l*R*T/F*self.activity(umid_l)*(1-trans_l)*(np.log(uc)-np.log(un))/hx
        term_r = kapeff_r*R*T/F*self.activity(umid_r)*(1-trans_r)*(np.log(up)-np.log(uc))/hx
        ans = (kapeff_r*(phiep - phiec)/hx- kapeff_l*(phiec - phien)/hx)/hx - (term_r-term_l)/hx
        return ans.reshape()
    
    def bc_phie_ps(self,phie0_p, phie1_p, phie0_s, phie1_s, u0_p, u1_p, u0_s, u1_s):
        T = self.T
        kapeff_p = self.electrolyteConductCoeff(self.kap, self.pe.eps,self.pe.brugg,(u0_p + u1_p)/2,T);
        kapeff_s = self.electrolyteConductCoeff(self.kap, self.eps,self.brugg,(u0_s + u1_s)/2,T);
        bc = -kapeff_p*(phie1_p - phie0_p)/self.pe_hx + kapeff_s*(phie1_s - phie0_s)/self.hx
        return bc.reshape()
    
    def bc_phie_sn(self,phie0_n, phie1_n, phie0_s, phie1_s, u0_n, u1_n, u0_s, u1_s):
        T = self.T
        kapeff_n = self.electrolyteConductCoeff(self.kap, self.ne.eps,self.ne.brugg,(u0_n + u1_n)/2,T);
        
        kapeff_s = self.electrolyteConductCoeff(self.kap, self.eps,self.brugg,(u0_s + u1_s)/2,T);
        bc = -kapeff_s*(phie1_s - phie0_s)/self.hx + kapeff_n*(phie1_n - phie0_n)/self.ne_hx
        return bc.reshape()
    
    def bc_phi_new(self, phie0, phie1, u1, u2, Iapp):
        kapeff = self.electrolyteConductCoeff(self.kap, self.eps,self.brugg,(u1 + u2)/2,self.T)
        # ans = (phie1-phie0) + Iapp/kapeff/self.trans*self.hx
        umid = (u1+u2)/2
        trans = self.transfer_func(self.trans, umid)
        # ans = 2*phie1 + Iapp/kapeff/trans*self.hx
        # ans = 2*phie1 + Iapp*(1+trans)/2/trans/kapeff*self.hx
        T = self.T
        eps = self.eps; brugg = self.brugg
        ans = 2*phie1 + (Iapp/kapeff + R*T/F*self.activity(umid)*(1-trans)**2*Iapp/self.electrolyteDiffCoeff(self.D, eps,brugg,umid,self.T)/F/umid)*self.hx
        return ans.reshape()

