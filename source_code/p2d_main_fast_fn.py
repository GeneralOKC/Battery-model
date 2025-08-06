import jax
import jax.numpy as np
from jax import vmap
from jax import config
config.update("jax_enable_x64", True)
from settings import F
import numpy as onp
import coeffs as coeffs
import timeit
from unpack import unpack_vars, unpack_fast
from scipy.sparse import csc_matrix
# from scikits.umfpack import splu
from scipy.sparse.linalg import splu
from p2d_newton_fast import newton_fast_sparse
image_folder = 'images'
video_name = 'video.avi'
from precompute_c import precompute
from p2d_param import get_battery_sections

def p2d_fast_fn_short(Np, Mp, Ms, delta_t, fn_fast, jac_fn, I_times, Iapp, Tf, Temp, p_electrode_param, electrolyte_sep_param, cathode, electrolyte, Ucutoff):
    start0 = timeit.default_timer()
    peq, sepq = get_battery_sections(Np, Mp, Ms, delta_t, Temp, p_electrode_param, electrolyte_sep_param, cathode, electrolyte) #10
    Ap, gamma_p, temp_p = precompute(peq)
    gamma_p_vec  = gamma_p * np.ones(Mp)
    # gamma_n_vec = gamma_n * np.ones(Mn)
    lu_p = splu(csc_matrix(Ap))
    # lu_n = splu(csc_matrix(An))
    OCV = coeffs.open_circuits[cathode]
    @jax.jit
    def cmat_format_p(cmat):
        # Обновлённый синтаксис: вместо jax.ops.index_update используем .at[...].set(...)
        val = cmat.at[0:Mp*(Np+2):Np+2].set(0)
        val = val.at[Np+1:Mp*(Np+2):Np+2].set(0)
        return val

    # @jax.jit
    # def cmat_format_n(cmat):
    #     val = cmat.at[0:Mn*(Nn+2):Nn+2].set(0)
    #     val = val.at[Nn+1:Mn*(Nn+2):Nn+2].set(0)
    #     return val
    
    @jax.jit
    def form_c2_p_jit(temp, j, Temp):
        Deff_vec = vmap(coeffs.solidDiffCoeff)(peq.Ds * np.ones(Mp), Temp*np.ones(Mp))
        fn = lambda j, temp, Deff: -(j * temp / Deff)
        val = vmap(fn, in_axes=(0, None, 0), out_axes=1)(j, temp, Deff_vec)
        return val

    # @jax.jit    
    # def form_c2_n_jit(temp, j, T):
    #     Deff_vec = vmap(coeffs.solidDiffCoeff)(neq.Ds * np.ones(Mn),
    #                                              neq.ED * np.ones(Mn),
    #                                              T*np.ones(Mn))
    #     fn = lambda j, temp, Deff: -(j * temp / Deff)
    #     val = vmap(fn, in_axes=(0, None, 0), out_axes=1)(j, temp, Deff_vec)
    #     return val

    # @jax.jit(static_argnums=(2, 3))
    def Icell(t):
        if onp.isscalar(Iapp):
            return Iapp
        else:
            return onp.interp(t, I_times, Iapp)

    def overpot_anode(c, t):
        k_ec_anode = 6.1e-6
        F = 96485
        R = 8.31
        eta = np.arcsinh(Icell(t)/2/k_ec_anode/F/c**0.5) * R * Temp / 0.5 / F
        return eta

    def combine_c(cII, cI_vec, M, N):
        return np.reshape(cII, [M * (N + 2)], order="F") + cI_vec
    combine_c = jax.jit(combine_c, static_argnums=(2, 3))
    
    U_fast = np.hstack([
        peq.u_initial + np.zeros(Mp + 2),
        peq.u_initial + np.zeros(Ms + 2),
        np.zeros(Mp),
        np.zeros(Mp),
        np.zeros(Mp+2) + OCV(peq.c_init, peq.cmax), #peq.open_circuit_poten_fast(peq.cavg,  peq.cmax),
        np.zeros(Mp+2) + 0,
        np.zeros(Ms+2) + 0,
    ])

    cmat_pe = peq.c_init * np.ones(Mp * (Np + 2))
    # cmat_ne = neq.cavg * np.ones(Mn * (Nn + 2))
    
    lu = {"pe": lu_p}

    steps = Tf / delta_t
    voltages = []
    times = []
    uvec_list = []
    cmat_list = []
    Rec_list = []
    eta_list = []
    phie_list = []
    phis_list = []
    psi_list = []
    end0 = timeit.default_timer()
    
    print("setup time", end0 - start0)

    cmat_rhs_pe = cmat_format_p(cmat_pe)
    # cmat_rhs_ne = cmat_format_n(cmat_ne)
    lu_pe = lu["pe"]
    # lu_ne = lu["ne"]

    cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
    # cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))

    cs_pe1 = (cI_pe_vec[Np : Mp * (Np + 2) : Np + 2] + cI_pe_vec[Np + 1 : Mp * (Np + 2) : Np + 2]) / 2
    # cs_ne1 = (cI_ne_vec[Nn : Mn * (Nn + 2) : Nn + 2] + cI_ne_vec[Nn + 1 : Mn * (Nn + 2) : Nn + 2]) / 2

    start_init = timeit.default_timer()
    Jinit = jac_fn(U_fast, U_fast, cs_pe1, gamma_p_vec, Icell(0), Temp).block_until_ready()
    end_init = timeit.default_timer()

    init_time = end_init - start_init

    solve_time_tot = 0
    jf_tot_time = 0
    overhead_time = 0

    start1 = timeit.default_timer()
    
    for i in range(0, int(steps)):
        cmat_rhs_pe = cmat_format_p(cmat_pe)
        # cmat_rhs_ne = cmat_format_n(cmat_ne)
        lu_pe = lu["pe"]
        # lu_ne = lu["ne"]

        start = timeit.default_timer()
        cI_pe_vec = lu_pe.solve(onp.asarray(cmat_rhs_pe))
        # cI_ne_vec = lu_ne.solve(onp.asarray(cmat_rhs_ne))
        end = timeit.default_timer()
        c_lintime = end - start

        cs_pe1 = (cI_pe_vec[Np : Mp * (Np + 2) : Np + 2] + cI_pe_vec[Np + 1 : Mp * (Np + 2) : Np + 2]) / 2
        # cs_ne1 = (cI_ne_vec[Nn : Mn * (Nn + 2) : Nn + 2] + cI_ne_vec[Nn + 1 : Mn * (Nn + 2) : Nn + 2]) / 2
        U_fast, info = newton_fast_sparse(fn_fast, jac_fn, U_fast, cs_pe1, gamma_p_vec, Icell(i*delta_t), Temp)
        (fail, solve_time, overhead, jf_time) = info
        
        


        overhead_time += overhead
        solve_time_tot += solve_time + c_lintime
        jf_tot_time += jf_time 
        
        phis_pe, phis_sep0, jvec_pe= unpack_vars(U_fast, Mp, Ms)
      
        cII_p = form_c2_p_jit(temp_p, jvec_pe, Temp)
        # cII_n = form_c2_n_jit(temp_n, jvec_ne, T)
        cmat_pe = combine_c(cII_p, cI_pe_vec, Mp, Np)
        cmat_pe1 = (cmat_pe[Np : Mp * (Np + 2) : Np + 2] + cmat_pe[Np + 1 : Mp * (Np + 2) : Np + 2])/2
        # cmat_ne = combine_c(cII_n, cI_ne_vec, Mn, Nn)
    
        # volt = phis_pe[1] - phis_sep0[Ms]
        uvec_pe, uvec_sep, phie_pe, phie_sep, phis_pe, j_pe, eta_pe = unpack_fast(U_fast,Mp, Np, Ms) #stopping
        volt = phis_pe[1] - phie_sep[Ms] - overpot_anode(uvec_sep[Ms], Icell(i*delta_t))

        voltages.append(volt)
        times.append(i*delta_t)
        uvec_pe1 = (uvec_pe[0:-1]+uvec_pe[1:])/2
        uvec_sep1 = (uvec_sep[0:-1]+uvec_sep[1:])/2
        uvec_list.append(np.concatenate([uvec_pe1, uvec_sep1]))
        cmat_list.append(cmat_pe1)
        Rec_list.append(j_pe*F*peq.a)
        eta_list.append(eta_pe)

        phie_sep1 = (phie_sep[0:-1]+phie_sep[1:])/2
        phie_pe1 = (phie_pe[0:-1]+phie_pe[1:])/2
        phie_list.append(np.concatenate([phie_pe1, phie_sep1]))
        phis_pe1 =  (phis_pe[0:-1]+phis_pe[1:])/2
        phis_list.append(phis_pe1)
        psi_list.append(phis_pe1-phie_pe1)
        if fail != 0:
            print('Premature end of run\n') 
            print("timestep:", i)
            break 

        # uvec_pe, uvec_sep, phie_pe, phie_sep, phis_pe, j_pe, eta_pe = unpack_fast(U_fast,Mp, Np, Ms) #stopping
        if np.any(uvec_pe < 1e-3) or np.any(uvec_sep < 1e-3):
            print('zero')
            # return U_fast, cmat_pe, voltages, temps, time_stats
            break
        elif volt < Ucutoff:
            # return U_fast, cmat_pe, voltages, temps, time_stats
            print('voltage < 3, stopped on step', i)
            break
        
    end1 = timeit.default_timer()
    tot_time = (end1 - start1)
    time_stats = (tot_time, solve_time_tot, jf_tot_time, overhead_time, init_time)
    print('total calculation time', tot_time/60)
    
    u_content_initial = np.sum(np.concatenate([peq.eps*peq.u_initial*np.ones(Mp+2)*peq.l/(Mp+2), sepq.eps*peq.u_initial*np.ones(Ms+2)*sepq.l/(Ms+2)]))
    u_content_end = np.sum(np.concatenate([uvec_pe*peq.eps*peq.l/(Mp+2), uvec_sep*sepq.eps*sepq.l/(Ms+2)]))
    print('initial and final electrolyte content', u_content_initial, u_content_end)

    uvec_pe = (uvec_pe[0:-1]+uvec_pe[1:])/2
    uvec_sep = (uvec_sep[0:-1]+uvec_sep[1:])/2
    uvec = np.concatenate([uvec_pe, uvec_sep])
    x_pe = np.linspace(0, peq.l, Mp+1)
    x_sep = np.linspace(peq.l, peq.l+sepq.l, Ms+1)
    x = np.concatenate([x_pe, x_sep])
    x_cs = np.linspace(0, peq.l, Mp)

    uvec_list = np.array(uvec_list)
    cmat_list = np.array(cmat_list)
    Rec_list = np.array(Rec_list)
    eta_list = np.array(eta_list)
    voltages = np.array(voltages)
    phie_list = np.array(phie_list)
    phis_list = np.array(phis_list)
    psi_list = np.array(psi_list)
    times = np.array(times)

    print("Done decoupled simulation\n")
    return uvec_list, cmat_list, Rec_list, eta_list, voltages, phie_list, phis_list, psi_list, x, x_cs, times, time_stats