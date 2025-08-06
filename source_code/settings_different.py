from dataclasses import dataclass

global F
global R
global Tref


F = 96485;
R = 8.314472;
Tref = 298.15;
h = 1


@dataclass
class separator_constants:    
    eps: float;
    brugg: float; 
    l:float; 
    trans: float
    gamma: float
    D: float
    
def sep_constants_LFP():
    eps = 0.45
    brugg = 1.5
    l = 2.5e-5
    trans = 0.36
    gamma = 2*(1-trans)*R/F
    D = 2e-10
    return separator_constants(eps, brugg, l, trans, gamma, D)

def sep_constants_NMC_622():
    l = 1.2e-5
    eps = 0.47
    brugg = 1.5
    trans = 0.2594
    gamma = 2*(1-trans)*R/F
    D = 1
    return separator_constants(eps, brugg, l, trans, gamma, D)

def sep_constants_NMC_532():
    l = 1.2e-5
    eps = 0.4
    brugg = 1.5
    trans = 0.38
    gamma = 2*(1-trans)*R/F
    D = 1
    return separator_constants(eps, brugg, l, trans, gamma, D)

@dataclass
class electrode_constants:
    eps: float; brugg:float;
    a: float; Rp: float;
    epsf: float;
    k:float
    Ds: float; l:float
    sigma: float; Ek: float;
    ED: float;
    trans: float
    gamma: float
    cmax: float
    D: float

def p_electrode_constants_LFP():
    # porosity
    eps = 0.426; 
    
    # Bruggeman's coefficient
    brugg = 1.5
    
    # Particle radius
    Rp= 5e-8;

    # Particle surface area to volume
    a= 3*eps/Rp #???

    # Filler fraction
    epsf = 0.025; #not necessary
    
    # Reaction rate
    k = 5e-2 #???
    
    # Solid-phase diffusivity
    Ds = 5.9e-18; 
    
    # Thickness
    l = 8e-5;
    
    # Solid-phase conductivity
    # sigma = 100;
    sigma = 0.33795074
    
    cmax = 22806
    Ek = 5000
    ED = 5000
    trans = 0.364
    gamma = 2*(1-trans)*R/F;
    D = 2e-10
    return electrode_constants(eps,brugg,a, Rp, epsf, k, Ds, l,sigma,Ek, ED, trans, gamma, cmax, D)

def p_electrode_conctants_NMC_622():
    eps = 0.335 

    brugg = 1.5

    Rp= 5.22e-6

    a= 3*eps/Rp #???

    epsf = 0.025; #not necessary

    k = 5e-2 #???

    Ds = 4e-15

    l = 7.56e-5
  
    sigma = 0.18
    
    cmax = 63104

    Ek = 5000
    ED = 5000

    trans = 0.2594
    gamma = 2*(1-trans)*R/F

    D = 1
    return electrode_constants(eps,brugg,a, Rp, epsf, k, Ds, l,sigma,Ek, ED, trans, gamma, cmax, D)

def p_electrode_conctants_NMC_532():
    eps = 0.3

    brugg = 1.5

    Rp= 3.5e-6

    a= 3*eps/Rp #???

    epsf = 0.025; #not necessary

    k = 5e-2 #???

    Ds = 4e-15

    l = 6.7e-5
  
    sigma = 100
    
    cmax = 35380

    Ek = 5000
    ED = 5000

    trans = 0.38
    gamma = 2*(1-trans)*R/F

    D = 3.22e-16
    return electrode_constants(eps,brugg,a, Rp, epsf, k, Ds, l,sigma,Ek, ED, trans, gamma, cmax, D)

@dataclass
class elec_grid_param_pack:
    M: int; N:int

@dataclass
class sep_grid_param_pack:
    M: int;
        
def p_electrode_grid_param(M, N):
#    M = 10; N = 5;
    return elec_grid_param_pack(M, N)

def sep_grid_param(M):
    
    return sep_grid_param_pack(M)
