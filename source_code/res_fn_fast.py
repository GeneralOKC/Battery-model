from functools import partial
import jax
import jax.numpy as np
from jax import vmap
from jax import config
config.update("jax_enable_x64", True)
# from numpy.linalg import norm
# import matplotlib.pylab as plt
from ElectrodeEquation import ElectrodeEquation
from SeparatorEquation import SeparatorEquation
# from CurrentCollectorEquation import CurrentCollectorEquation
# from settings import (
#     p_electrode_constants,
#     p_electrode_grid_param,
#     n_electrode_constants,
#     n_electrode_grid_param,
#     sep_constants,
#     sep_grid_param,
#     a_cc_constants,
#     z_cc_constants,
#     cc_grid_param,
# )
from settings import (
    p_electrode_grid_param,
    sep_grid_param,
    p_electrodes,
    electrolytes
)
class ResidualFunctionFast():
    def __init__(self, Mp, Np, Ms, delta_t, T, p_electrode_param, electrolyte_sep_param, cathode, electrolyte):
        self.Mp = Mp
        self.Np = Np
        self.Ms = Ms
        self.T = T
        p_electrode_constants, sep_constants = p_electrode_param, electrolyte_sep_param
        self.peq = ElectrodeEquation(
            p_electrode_constants,
            p_electrode_grid_param(Mp, Np),
            "positive",
            sep_constants,
            sep_grid_param(Ms),
            delta_t,
            T,
            cathode,
            electrolyte
        )

        self.sepq = SeparatorEquation(
            sep_constants,
            sep_grid_param(Ms),
            p_electrode_constants,
            p_electrode_grid_param(Mp, Np),
            delta_t,
            T,
            cathode,
            electrolyte
        )

        # self.Iapp = Iapp
        self.up0 = 0
        self.usep0 = self.up0 + Mp + 2

        self.jp0 = self.usep0 + Ms + 2

        self.etap0 = self.jp0 + Mp

        self.phisp0 = self.etap0 + Mp

        self.phiep0 = self.phisp0 + Mp + 2
        self.phiesep0 = self.phiep0 + Mp + 2


        Ntot_pe = 3 * (Mp + 2) + 2 * (Mp)
        Ntot_sep = 2 * (Ms + 2)
        Ntot = Ntot_pe + Ntot_sep #+ Ntot_acc + Ntot_zcc
        self.Ntot = Ntot

    @partial(jax.jit, static_argnums=(0,))
    def res_u_pe(self, val, uvec, jvec, uvec_old, uvec_sep, phievec):
        up0 = self.up0
        Mp = self.Mp
        peq = self.peq
        val = val.at[up0].set(peq.bc_zero_neumann(uvec[0], uvec[1]))
        # val =val.at[up0].set(peq.bc_neumann_new(uvec[0], uvec[1]))
        val = val.at[up0 + 1 : up0 + Mp + 1].set(
            vmap(peq.electrolyte_conc)(
                uvec[0:Mp],
                uvec[1:Mp + 1],
                uvec[2:Mp + 2],
                jvec[0:Mp],
                uvec_old[1:Mp + 1],
                phievec[0:Mp],
                phievec[1:Mp + 1],
                phievec[2:Mp + 2],
            )
        )
        val = val.at[up0 + Mp + 1].set(
            peq.bc_u_sep_p(
                uvec[Mp],
                uvec[Mp + 1],
                uvec_sep[0],
                uvec_sep[1],
            )
        )
        return val

    @partial(jax.jit, static_argnums=(0,))
    def res_u_sep(self, val, uvec, uvec_old, uvec_pe, Icell, phievec):
        usep0 = self.usep0
        Ms = self.Ms
        peq = self.peq
        Mp = self.Mp
        sepq = self.sepq
        val = val.at[usep0].set(
            peq.bc_inter_cont(uvec[0], uvec[1], uvec_pe[Mp], uvec_pe[Mp + 1])
        )
        val = val.at[usep0 + 1 : usep0 + 1 + Ms].set(
            vmap(sepq.electrolyte_conc)(
                uvec[0:Ms],
                uvec[1:Ms + 1],
                uvec[2:Ms + 2],
                uvec_old[1:Ms + 1],
                phievec[0:Mp],
                phievec[1:Mp + 1],
                phievec[2:Mp + 2],
            )
        )
        val = val.at[usep0 + Ms + 1].set(
            sepq.bc_neumann_new(uvec[Ms], uvec[Ms + 1], Icell)
        )
        return val

    @partial(jax.jit, static_argnums=(0,))
    def res_phie_pe(self, val, uvec, phievec, jvec, uvec_sep, phievec_sep):
        phiep0 = self.phiep0
        peq = self.peq
        Mp = self.Mp
        val = val.at[phiep0].set(
            peq.bc_zero_neumann(phievec[0], phievec[1])
        )
        val = val.at[phiep0 + 1 : phiep0 + Mp + 1].set(
            vmap(peq.electrolyte_poten)(
                uvec[0:Mp],
                uvec[1:Mp + 1],
                uvec[2:Mp + 2],
                phievec[0:Mp],
                phievec[1:Mp + 1],
                phievec[2:Mp + 2],
                jvec[0:Mp],
            )
        )
        val = val.at[phiep0 + Mp + 1].set(
            peq.bc_phie_p(
                phievec[Mp],
                phievec[Mp + 1],
                phievec_sep[0],
                phievec_sep[1],
                uvec[Mp],
                uvec[Mp + 1],
                uvec_sep[0],
                uvec_sep[1],
            )
        )
        return val

    @partial(jax.jit, static_argnums=(0,))
    def res_phie_sep(self, val, uvec, phievec, phievec_pe, Icell):
        phiesep0 = self.phiesep0
        peq = self.peq
        sepq = self.sepq
        Ms = self.Ms
        # neq = self.neq
        Mp = self.Mp
        val = val.at[phiesep0].set(
            peq.bc_inter_cont(phievec_pe[Mp], phievec_pe[Mp + 1], phievec[0], phievec[1])
        )
        val = val.at[phiesep0 + 1 : phiesep0 + Ms + 1].set(
            vmap(sepq.electrolyte_poten)(
                uvec[0:Ms],
                uvec[1:Ms + 1],
                uvec[2:Ms + 2],
                phievec[0:Ms],
                phievec[1:Ms + 1],
                phievec[2:Ms + 2],
            )
        )
        val = val.at[phiesep0 + Ms + 1].set(
            sepq.bc_phi_new(phievec[Ms], phievec[Ms + 1], uvec[Ms], uvec[Ms+1], Icell)
        )
        return val

    @partial(jax.jit, static_argnums=(0,))
    def res_phis(self, val, phis_pe, jvec_pe, Icell):
        phisp0 = self.phisp0
        peq = self.peq
        Mp = self.Mp
        val = val.at[phisp0].set(
            peq.bc_phis(phis_pe[0], phis_pe[1], Icell)
        )
        val = val.at[phisp0 + 1 : phisp0 + Mp + 1].set(
            vmap(peq.solid_poten)(
                phis_pe[0:Mp],
                phis_pe[1:Mp + 1],
                phis_pe[2:Mp + 2],
                jvec_pe[0:Mp],
            )
        )
        val = val.at[phisp0 + Mp + 1].set(
            peq.bc_phis(phis_pe[Mp], phis_pe[Mp + 1], 0)
        )
        return val

    @partial(jax.jit, static_argnums=(0,))
    def res_j_fast(self, val, jvec_pe, uvec_pe, eta_pe, cs_pe1, gamma_p):
        jp0 = self.jp0
        peq = self.peq
        Mp = self.Mp
        val = val.at[jp0 : jp0 + Mp].set(
            vmap(peq.ionic_flux_fast)(
                jvec_pe,
                uvec_pe[1:Mp + 1],
                eta_pe,
                cs_pe1,
                gamma_p,
                peq.cmax * np.ones(Mp),
            )
        )
        return val

    @partial(jax.jit, static_argnums=(0,))
    def res_eta_fast(self, val, eta_pe, phis_pe, phie_pe, jvec_pe, cs_pe1, gamma_p):
        etap0 = self.etap0
        # etan0 = self.etan0
        Mp = self.Mp
        # Mn = self.Mn
        peq = self.peq
        # neq = self.neq
        val = val.at[etap0 : etap0 + Mp].set(
            vmap(peq.over_poten_fast)(
                eta_pe,
                phis_pe[1:Mp + 1],
                phie_pe[1:Mp + 1],
                jvec_pe,
                cs_pe1,
                gamma_p,
                peq.cmax * np.ones(Mp),
            )
        )
        return val