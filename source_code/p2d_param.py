#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:07:37 2020

@author: hanrach
"""
from ElectrodeEquation import ElectrodeEquation
from SeparatorEquation import SeparatorEquation
from settings import p_electrodes,p_electrode_grid_param, electrolytes, sep_grid_param
#cavg 25751
#cavg 24568

def get_battery_sections(Np, Mp,Ms, delta_t, Temp, p_electrode_param, electrolyte_sep_param, cathode, electrolyte):
    p_electrode_constants, sep_constants = p_electrode_param, electrolyte_sep_param
    peq = ElectrodeEquation(p_electrode_constants,p_electrode_grid_param(Mp, Np), "positive", \
                                 sep_constants, sep_grid_param(Ms), \
                                 delta_t, Temp, cathode, electrolyte)
    # neq = ElectrodeEquation(n_electrode_constants(),n_electrode_grid_param(Mn, Nn), "negative", \
    #                              sep_constants(), sep_grid_param(Ms), \
    #                              26128, 30555,delta_t, T)
    
    sepq = SeparatorEquation(sep_constants,sep_grid_param(Ms), \
                                  p_electrode_constants,\
                                  p_electrode_grid_param(Mp,Np), delta_t, Temp, cathode, electrolyte)
    # accq = CurrentCollectorEquation(a_cc_constants(),cc_grid_param(Ma),delta_t, Iapp)
    # zccq = CurrentCollectorEquation(z_cc_constants(),cc_grid_param(Mz),delta_t, Iapp)

    Mp = peq.M;  Ms = sepq.M; #Ma = accq.M; Mz = zccq.M
    Np = peq.N;
    
    Ntot_pe = (Np+2)*(Mp) + 3*(Mp + 2) + 2*(Mp)
    # Ntot_ne = (Nn+2)*(Mn) + 3*(Mn + 2) + 2*(Mn)
    
    Ntot_sep =  2*(Ms + 2)
    # Ntot_acc =Ma+ 2
    # Ntot_zcc = Mz+ 2
    Ntot = Ntot_pe +  Ntot_sep #+ Ntot_acc + Ntot_zcc
    return peq, sepq
