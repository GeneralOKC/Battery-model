from jax import config
config.update('jax_enable_x64', True)
from unpack import unpack_fast, unpack
from init import p2d_init_fast
from p2d_main_fast_fn import p2d_fast_fn_short
import matplotlib.pyplot as plt
import numpy as np
from settings import p_electrodes, electrolytes
from input_file import Tf, I_times, Iapp, p_electrode_param, electrolyte_sep_param, cathode, electrolyte, name_file, num_plots, path_to_save, Ucutoff
import os
os.chdir('D:\\')
# Number of points in each battery section
Np = 50
# Nn = Np
Mp = Np
Ms = Np
# Mn = Np
# Ma = 60
# Mz = 60
Temp = 297
# Time-step size
delta_t = 1
# Tf = 1000
# Applied current
# Iapp = -5

# cathode = 'NMC_811_tuned'
# electrolyte = 'EC:EMC_3:7'
# path_to_save = 'python\\p2d_fast_solver-main\\all_in_release\\NMC_811_tuned_transfer'
# p_electrode_param, electrolyte_sep_param = p_electrodes[cathode], electrolytes[electrolyte]
# I_times = None

# Iapp = np.zeros(1000)
# Iapp[0:300] = -3
# Iapp[500:700] = -5
# I_times = np.linspace(0, 1000, 1000)

# def set_current(path):
#     arr = np.loadtxt(path, skiprows=1)
#     I_times = arr[:, 1]
#     Iapp = arr[:, 2]
#     return I_times, Iapp

# I_times, Iapp = set_current('python\\p2d_fast_solver-main\\all_in\\2.csv')
# I_times = 0


fn, jac_fn = p2d_init_fast(Np, Mp, Ms, delta_t, Temp, p_electrode_param, electrolyte_sep_param, cathode, electrolyte)
uvec_list, cmat_list, Rec_list, eta_list, voltages, phie_list, phis_list, psi_list, x, x_cs, times, time_stats = p2d_fast_fn_short(Np, Mp, Ms, delta_t, fn, jac_fn, I_times, Iapp, Tf, Temp, p_electrode_param, electrolyte_sep_param, cathode, electrolyte, Ucutoff)
# times = np.array(times)

# print(voltages)
# print(np.shape(U_fast), np.shape(cmat_pe), np.shape(voltages), np.shape(temps),np.shape(time), time)
# uvec_pe, Tvec_pe, phie_pe, phis_pe, \
# uvec_ne, Tvec_ne, phie_ne, phis_ne,\
# uvec_sep, Tvec_sep, phie_sep, Tvec_acc, Tvec_zcc,\
# j_pe, eta_pe, j_ne, eta_ne = unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)

# uvec_pe, uvec_sep, phie_pe, phie_sep, phis_pe, j_pe, eta_pe = unpack_fast(U_fast,Mp, Np, Ms)
# print(uvec_pe, uvec_sep)
# uvec_pe = (uvec_pe[0:-1]+uvec_pe[1:])/2
# uvec_sep = (uvec_sep[0:-1]+uvec_sep[1:])/2
# u1 = np.concatenate([uvec_pe*0.385*8e-5/Mp, uvec_sep*0.724*2.5e-5/Ms])
# u_initial = np.concatenate([np.ones(np.size(uvec_pe))*0.385*1000*8e-5/Mp, np.ones(np.size(uvec_sep))*0.724*1000*2.5e-5/Ms])
# print('conc', np.sum(u1), np.sum(u_initial))
# x_pe = np.linspace(0, 8*1e-5, np.size(uvec_pe))
# x_sep = np.linspace(8*1e-5, 8*1e-5+2.5*1e-5, np.size(uvec_sep))
# x = np.concatenate([x_pe, x_sep])
# print(np.shape(unpack_fast(U_fast,Mp, Np, Mn, Nn, Ms, Ma, Mz)))
# print(uvec_pe)
# u = np.concatenate([uvec_pe, uvec_sep])
# u = np.concatenate([uvec_sep, uvec_pe])
# x = np.linspace(0, 10.5, np.size(u))

# for i in range(0, Tf, 1000):
#     plt.plot(x*10**6, uvec_list[i])
# plt.show()

# for i in range(0 ,Tf, 1000):
#     plt.plot(x_cs*10**6, cmat_list[i])
# plt.show()

# for i in range(0 ,Tf, 1000):
#     plt.plot(x_cs*10**6, j_list[i])
# plt.show()

# for i in range(0 ,Tf, 1000):
#     plt.plot(x_cs*10**6, eta_list[i])
# plt.show()

# plt.plot(x, u1)
# plt.plot(x, u_initial)
# plt.show()
# cmat = np.concatenate([cmat_pe, cmat_ne])
# x1 = np.linspace(0, 1, np.size(cmat))
# plt.plot(x, cmat)
# plt.show()

# plt.plot(times, voltages)
# plt.show()

def multipage(filename, figs=None, dpi=200):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    # all opened figures are saved to PDF 'filename'
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()

def plotter(num_graph, report_path, PLOT_POTENTIALS=False, name = 'report_figs'):
    step = round(times.size/(num_graph-1))
    # time_points_ivp - indeces of time-points for the array of ivp_solver results (concentration results)
    time_points_ivp = np.arange(0, step*(num_graph-1), step)
    time_points_ivp = np.append(time_points_ivp,times.size-1)

    time_legend = []
    for i in time_points_ivp:
        time_legend = time_legend + ["%7.1f min" % (times[i]/60)]

    fig1 = plt.figure()
    plt.plot(times/60, voltages)
    plt.xlabel('t, min')
    plt.ylabel('Ucell, V')
    plt.grid()

    fig2 = plt.figure()
    for l, i in enumerate(time_points_ivp):
        plt.plot(x/1e-6, uvec_list[i]/1000, label=time_legend[l])
    plt.legend()
    plt.xlabel('x, um')
    plt.ylabel('electrolyte concentration, mol/l')
    plt.grid()

    fig3 = plt.figure()
    for l, i in enumerate(time_points_ivp):
        plt.plot(x_cs/1e-6, cmat_list[i]/1000, label=time_legend[l])
    plt.legend()
    plt.xlabel('x, um')
    plt.ylabel('intercalated lithim, mol/l')
    plt.grid()
    # fig4 = plt.figure()
    # for l, i in enumerate(time_points_ivp):
    #     plt.plot(grid_cath/1e-6, dod[i,:]*100, label=time_legend[l])
    # plt.legend()
    # plt.xlabel('x, um')
    # plt.ylabel('Depth of Discharge, %')

    fig5 = plt.figure()
    for l, i in enumerate(time_points_ivp):
        plt.plot(x_cs/1e-6, -Rec_list[i]/1000, label=time_legend[l])
    plt.legend()
    plt.xlabel('x, um')
    plt.ylabel('reaction rate, mA/cm3')
    plt.grid()

    fig6 = plt.figure()
    for l, i in enumerate(time_points_ivp):
        plt.plot(x_cs/1e-6, eta_list[i]*1000, label=time_legend[l])
    plt.legend()
    plt.xlabel('x, um')
    plt.ylabel('Overpotential, mV')
    plt.grid()
    multipage(f'{report_path}\\{name}.pdf')
    plt.close('all')

def export_res_text(path):
    np.savetxt(f'{path}\\grid_full.csv', x, fmt='%8.5e', delimiter=' ')
    np.savetxt(f'{path}\\grid_cathode.csv', x_cs, fmt='%8.5e', delimiter=' ')
    np.savetxt(f'{path}\\time.csv', times, fmt='%10.2f', delimiter=' ')
    np.savetxt(f'{path}\\Ucell.csv', voltages, fmt='%8.3f', delimiter=' ')
    np.savetxt(f'{path}\\conc_electrolyte.csv', uvec_list, fmt='%10.2f', delimiter=' ')
    np.savetxt(f'{path}\\conc_solid.csv',       cmat_list, fmt='%10.2f', delimiter=' ')
    # np.savetxt(f'{path}\\AM_saturation.csv',    dod, fmt='%8.3f', delimiter=' ')
    np.savetxt(f'{path}\\reaction_rate.csv',    -Rec_list, fmt='%12.5e', delimiter=' ')
    np.savetxt(f'{path}\\overpotential.csv',    eta_list, fmt='%10.6f', delimiter=' ')
    np.savetxt(f'{path}\\Psi.csv',              psi_list, fmt='%10.6f', delimiter=' ')
    np.savetxt(f'{path}\\PhiEL.csv',            phie_list, fmt='%10.6f', delimiter=' ')
    np.savetxt(f'{path}\\PhiAM.csv',            phis_list, fmt='%10.6f', delimiter=' ')

# num_plots = 5
# name_file = 'figs_NMC_811_tuned_transfer'
plotter(num_plots, path_to_save, False, name_file)
export_res_text(path_to_save)