import numpy as np
from settings import p_electrodes, electrolytes
from init import set_current
from coeffs import transference_numbers
import shutil
import os
os.chdir('D:\\')

"""Simulation time in seconds"""
Tf = 1000

"""Voltage cutoff in Volts after which the simulation is stopped"""
Ucutoff = 3.0

"""Provide full path to array of current"""
# path_to_current = 'python\\p2d_fast_solver-main\\all_in_release\\2.csv'
# I_times, Iapp = set_current(path_to_current)

"""Or choose value of constant current""" 
I_times = 0
Iapp = -10 #[A/m**2]


"""
List of electrloyte/separator parameters
    eps #Porosity
    brugg # Bruggeman's coefficient
    l #Thickness [m]
    trans #Transference number
    D #Electrolyte diffusion coefficient [m^2/s]
    kap #Electrolyte conductivity coefficient [S/m]
    u_initial #Initial electrolyte concentration [mol/m^3]

List of cathode material parameters
    eps # porosity
    brugg # Bruggeman's coefficient
    a # Particle surface area to volume [1/m]
    Rp # Particle radius [m]
    k # Reaction rate
    Ds # Solid-phase diffusivity [m^2/s]
    l # Thickness [m]
    sigma # Solid-phase conductivity [S/m]
    cmax #Maximum solid phase concentration [mol/m^3]
    c_init #Initial solid phase concentration [mol/m^3]
"""
"""
Choose cathode material and electrolyte+separator parameters
Available electrolytes: 'LiPF6_EC:EMC_3:7', 'LiPF6_EC:DEC_1:1'
Available cathode material: 'NMC_811_tuned'
"""

cathode = 'NMC_811_tuned'
electrolyte = 'LiPF6_EC:EMC_3:7'
p_electrode_param, electrolyte_sep_param = p_electrodes[cathode](), electrolytes[electrolyte]()
electrolyte_sep_param.trans = transference_numbers[electrolyte](1, electrolyte_sep_param.u_initial)
# p_electrode_param.Ds = 1.07e-13
# p_electrode_param.sigma = p_electrode_param.sigma*100

"""Provide path to save data"""

path_to_save = 'python\\p2d_fast_solver-main\\all_in_release\\NMC_811_EC_EMC_activity'
path_to_exec = 'python\\p2d_fast_solver-main\\all_in_release\\main_decoupled.py'
name_file = 'figs_NMC_811_bigger_sigma'
os.makedirs(path_to_save, exist_ok=True)

"""Choose how many time points to be plotted"""

num_plots = 5

"""Execution of program"""
if __name__ == "__main__":
    exec(open(path_to_exec).read(), locals())
    path_to_input = 'python\\p2d_fast_solver-main\\all_in_release\\input_file.py'
    path_to_save_input = 'python\\p2d_fast_solver-main\\all_in_release\\NMC_811_EC_EMC_activity'
    shutil.copy(path_to_input, path_to_save_input)