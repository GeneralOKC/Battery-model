from dataclasses import dataclass
import numpy as onp
# global trans
global F
global R
# global gamma
global Tref
# global Iapp



F = 96485;
R = 8.314472;
# gamma = 2*(1-trans)*R/F;
Tref = 298.15;
# h = 1

@dataclass
class sep_grid_param_pack:
    M: int;
    
@dataclass
class electrolyte_constants:    
    eps: float; #Porosity
    brugg: float; # Bruggeman's coefficient
    l:float; #Thickness
    trans: float #Transference number
    D: float; #Electrolyte diffusion coefficient
    kap: float #Electrolyte conductivity coefficient
    u_initial: float #Initial electrolyte concentration

def electrolyte_constants_example():
    eps = 0.724
    brugg = 4;
    l = 2.5*1e-5;
#    l = 8*1e-5;
    trans = 0.364
    D = 1e-4
    kap = 1
    u_initial = 1000
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

def electrolyte_constants_LFP():
    eps = 0.45
    brugg = 1.5
    l = 2.5e-5
    trans = 0.36
    D = 2e-10
    kap = 1
    u_initial = 1200
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

def electrolyte_constants_NMC_622():
    l = 1.2e-5
    eps = 0.47
    brugg = 1.5
    trans = 0.2594
    D = 1
    kap = 1
    u_initial = 1000
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

def electrolyte_constants_NMC_532():
    l = 1.2e-5
    eps = 0.4
    brugg = 1.5
    trans = 0.38
    D = 5.2e-10
    kap = 1
    u_initial = 1000
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

def electrolyte_constants_NMC_release():
    l = 10e-6
    eps = 1
    brugg = 1.5
    trans = 0.3
    D = 16e-13
    kap = 1
    u_initial = 1530
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

def electrolyte_constants_EC_EMC_3_7():
    l = 1.2e-5
    eps = 0.47
    brugg = 2.57
    trans = 0.2594
    D = 1#3.22e-16
    kap = 1
    u_initial = 1000
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

#EC:EMC_3:7 LiPF6
#https://doi.org/10.1016/j.electacta.2008.04.023
def electrolyte_constants_EC_EMC_3_7_tuned():
    l = 1.2e-5
    eps = 0.47
    brugg = 1.5
    u_initial = 1000

    trans = 1#0.2594
    D = 1#3.22e-16
    kap = 1
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

#EC:DEC 1:1 LiPF6
#DOI 10.1149/2.0641503jes
def electrolyte_constants_EC_DEC_1_1():
    l = 1.2e-5
    eps = 0.47
    brugg = 1.5
    u_initial = 1000

    trans = 1
    D = 1
    kap = 1
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

#EC:EMC:DEC 1:1:1 LiPF6
def electrolyte_constants_EC_EMC_DEC_1_1_1():
    l = _
    eps = _
    brugg = _
    trans = _
    D = _
    kap = 0.003
    u_initial = 1000
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

#PEO-LITFSI-SiO2-SN
#https://doi.org/10.1002/open.202000107
def electrolyte_constants_PEO_LITFSI():
    l = 1.2e-5
    eps = 0.47
    brugg = 1.5
    u_initial = 1000

    trans = 1
    D = 1
    kap = 8e-3 #1e-1
    return electrolyte_constants(eps, brugg, l, trans, D, kap, u_initial)

def sep_grid_param(M):
    
    return sep_grid_param_pack(M)


@dataclass
class electrode_constants:
    eps: float; # porosity
    brugg:float; # Bruggeman's coefficient
    a: float; # Particle surface area to volume
    Rp: float; # Particle radius
    k:float # Reaction rate
    Ds: float; # Solid-phase diffusivity
    l:float # Thickness
    sigma: float; # Solid-phase conductivity
    cmax: float  #Maximum solid phase concentration
    c_init: float
    
def p_electrode_constants_example():
    # porosity
    eps = 0.385; 
    
    # Bruggeman's coefficient
    brugg = 4;
    
    # Particle surface area to volume
    a= 885000;
    
    # Particle radius
    Rp= 2*1e-6;
    
    # Reaction rate
    k = 2.334*1e-11
    
    # Solid-phase diffusivity
    Ds = 1e-14; 
    
    # Thickness
    l = 8*1e-5;
    
    # Solid-phase conductivity
    sigma = 100;

    #Maximum concentration
    cmax = 51554

    c_init =  25751
    return electrode_constants(eps,brugg,a, Rp, k, Ds, l,sigma, cmax,  c_init)

def p_electrode_constants_LFP():
    # porosity
    eps = 0.426; 
    
    # Bruggeman's coefficient
    brugg = 1.5
    
    # Particle radius
    Rp= 5e-8;

    # Particle surface area to volume
    eps_AM = 0.374 
    a= 3*eps_AM/Rp 
    
    # Reaction rate
    k = 6e-1/F #???
    
    # Solid-phase diffusivity
    Ds = 5.9e-18; 
    
    # Thickness
    l = 8e-5;
    
    # Solid-phase conductivity
    sigma = 0.33795074
    
    #Maximum concentration
    cmax = 22806

    c_init = 86.6628
    return electrode_constants(eps,brugg,a, Rp, k, Ds, l,sigma, cmax, c_init)

def p_electrode_constants_NMC_622():
    eps = 0.335 

    brugg = 1.5

    Rp= 5.22e-6

    a= 3*(1-eps)/Rp #???

    k = 5e-2 #???

    Ds = 4e-15

    l = 7.56e-5
  
    sigma = 0.18
    
    cmax = 63104

    return electrode_constants(eps,brugg,a, Rp, k, Ds, l,sigma, cmax, c_init)

def p_electrode_constants_NMC_532():
    eps = 0.3

    brugg = 1.5

    Rp= 3.5e-6

    eps_AM = 0.445
    a= 3*eps_AM/Rp #???

    Ds = 8e-15 #Ds = 4e-15

    l = 6.7e-5
  
    sigma = 100
    
    cmax = 35380

    i0_ref = 1.9609e-6
    u_ref = 1000.0
    k = i0_ref/(u_ref**0.5*cmax)/F #???
    k = 5e-11

    u_initial = 1000
    c_init = 48.8682#31513
    return electrode_constants(eps,brugg,a, Rp, k, Ds, l,sigma, cmax, c_init)

def p_electrode_constants_NMC_release():
    AM_dens = 4780 # kg/m3; active material density
    AM_molw = 0.097281 # kg/mol; active material molecular weight

    eps = 0.4

    #brugg = 1.53
    brugg = 1.53 - onp.log(1.8)/onp.log(eps)

    Rp= 1e-6

    a= 3*(1-eps)/Rp #???

    Ds = 4e-15 #from NMC_532

    l = 50e-6
  
    sigma = 10
    
    Lix_max = 1.0
    cmax = AM_dens/AM_molw*Lix_max

    i0_cath = 3
    c_el_cath_ref = 1000 
    
    Lix_ref = 0.5 
    c_sref =  AM_dens/AM_molw*Lix_ref
    k =  i0_cath/(c_el_cath_ref**0.5* c_sref**(1-0.5) * (cmax-c_sref)**0.5)/F


    Lix_init = 0.10
    c_init = AM_dens/AM_molw*Lix_init
    return electrode_constants(eps,brugg,a, Rp, k, Ds, l,sigma, cmax, c_init)

#NMC811
#DOI 10.1149/1945-7111/ab9050
def p_electrode_constants_NMC_811():
    eps = 0.335

    brugg = 2.43

    Rp= 5.22e-6

    eps_AM = 0.665
    a= 3*eps_AM/Rp #???

    Ds = 1.48e-15#4e-15

    l = 7.56e-5
  
    sigma = 0.18
    
    cmax = 51765#63104

    k = 3.42e-6/F/2 

    u_initial = 1000
    c_init = 17038
    return electrode_constants(eps,brugg,a, Rp, k, Ds, l,sigma,  cmax, c_init)

#NMC811 tuned
#DOI 10.1149/1945-7111/ab9050
def p_electrode_constants_NMC_811_tuned():
    eps = 0.335

    brugg = 1.5

    Rp= 5.22e-6

    eps_AM = 0.665
    a= 3*eps_AM/Rp #???

    Ds = 4e-15

    l = 7.56e-5
  
    sigma = 0.18
    
    cmax = 63104

    k = 3.42e-6/F 
    c_init = 17038
    return electrode_constants(eps,brugg,a, Rp, k, Ds, l,sigma,  cmax, c_init)

p_electrodes = {'example':p_electrode_constants_example, 'LFP': p_electrode_constants_LFP, 'NMC_622': p_electrode_constants_NMC_622, 'NMC_532': p_electrode_constants_NMC_532, 'NMC_release': p_electrode_constants_NMC_release, 'NMC_811': p_electrode_constants_NMC_811, 'NMC_811_tuned':p_electrode_constants_NMC_811_tuned} 
electrolytes = {'example': electrolyte_constants_example, 'LFP': electrolyte_constants_LFP, 'NMC_622': electrolyte_constants_NMC_622, 'NMC_532': electrolyte_constants_NMC_532, 'NMC_release': electrolyte_constants_NMC_release, 'NMC_811':electrolyte_constants_EC_EMC_3_7, 'LiPF6_EC:EMC_3:7': electrolyte_constants_EC_EMC_3_7_tuned, 'LiPF6_EC:DEC_1:1': electrolyte_constants_EC_DEC_1_1}  
 
# def n_electrode_constants():
#     # porosity
#     eps = 0.485; 
    
#     # Bruggeman's coefficient
#     brugg = 4;
    
#     # Particle surface area to volume
#     a= 723600;
    
#     # Particle radius
#     Rp= 2*1e-6;
    
#     # Thermal conductivity
#     lam = 1.7; 
    
#     # Filler fraction
#     epsf = 0.0326;
    
#     # Density
#     rho = 2500; 
    
#     # Specific heat
#     Cp = 700;
    
#     # Reaction rate
#     k = 5.031*1e-11
    
#     # Solid-phase diffusivity
#     Ds = 3.9*1e-14; 
    
#     # Thickness
#     l = 8.8*1e-5;
# #    l = 8*1e-5;
    
#     # Solid-phase conductivity
#     sigma = 100;
    
#     Ek = 5000;
    
#     ED = 5000;

#     return electrode_constants(eps,brugg,a, Rp,lam, epsf, \
#                                rho,Cp, k, Ds, l,sigma, Ek, ED)


@dataclass
class elec_grid_param_pack:
    M: int; N:int
    
def p_electrode_grid_param(M, N):
#    M = 10; N = 5;
    return elec_grid_param_pack(M, N)

# def n_electrode_grid_param(M, N):
# #    M = 10; N = 5;
#     return elec_grid_param_pack(M, N)

# @dataclass
# class current_collector_constants:    
#     lam: float; rho:float;
#     Cp: float; sigma: float; 
#     l:float
    
# @dataclass
# class cc_grid_param_pack:
#     M: int
    
# def a_cc_constants():
#     lam = 237; rho = 2700;
#     Cp = 897;
#     sigma = 3.55*1e7
#     l = 1.0*1e-5
# #    l = 8*1e-5;
#     return current_collector_constants(lam, rho, Cp, sigma,l)

# def z_cc_constants():
#     lam = 401; rho = 8940;
#     Cp = 385;
#     sigma = 5.96*1e7
#     l = 1.0*1e-5
# #    l = 8*1e-5;
#     return current_collector_constants(lam, rho, Cp, sigma,l)

# def cc_grid_param(M):
    
#     return cc_grid_param_pack(M)
