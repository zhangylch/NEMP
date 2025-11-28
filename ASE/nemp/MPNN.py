import sys
import numpy as np
import jax
import flax
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype,array
import flax.linen as nn
from typing import Sequence, List, Union, Optional
from flax.core import freeze
from jax.lax import fori_loop
from jax import Array
# need to be updated for ase interface
from ase.calculators.nemp import MLP     # need to be updated for ase interface
from ase.calculators.nemp import sph_cal     # need to be updated for ase interface
from ase.calculators.nemp.data_config import ModelConfig




class MPNN(nn.Module):
    """
    An equivariant message passing neural network, optimized for JAX.
    This version uses a standard Python for-loop for the message passing steps,
    which is robust, readable, and efficient for a typical number of iterations.
    """
    config: ModelConfig

    def setup(self):
        # The model's data type is dynamically inferred from the configuration.
        dtype = self.config.initbias_neigh.dtype

        self.sph_cal=sph_cal.SPH_CAL(max_l = self.config.rmaxl-1)

        self.scale = self.param('scale', lambda rng: jnp.array(np.array([1.0, 0.0]*self.config.nspec), dtype=dtype))

        self.spec_coeff = self.param('spec_coeff', nn.initializers.normal(1.0), (self.config.nspec, self.config.nwave, self.config.nwave), dtype)

        self.contract_coeff = self.param('contract_coeff', nn.initializers.normal(1.0), (self.config.MP_loop, self.config.nspec, 3, self.config.nwave, self.config.nwave), dtype)

        self.l_coeff = self.param('l_coeff', nn.initializers.normal(1.0), (self.config.MP_loop, self.config.nspec, self.config.num_cg, self.config.nwave), dtype)

        self.neighcoeffnn = MLP.MLP(num_output = self.config.npaircode, num_blocks = self.config.emb_nl[0], features = self.config.emb_nl[1], layers_per_block = self.config.emb_nl[2], use_bias=True, bias_init_value = jnp.ones(self.config.npaircode), cst=self.config.cst, dtype=dtype)

        self.neighnn = MLP.MLP(num_output = self.config.nradial, num_blocks = self.config.emb_nl[0], features = self.config.emb_nl[1], layers_per_block = self.config.emb_nl[2], use_bias=True, bias_init_value = self.config.initbias_neigh, cst=self.config.cst, dtype=dtype)

        self.rweightnn = MLP.MLP(num_output = self.config.nradial + self.config.nwave, num_blocks = self.config.emb_nl[0], features = self.config.emb_nl[1], layers_per_block = self.config.emb_nl[2], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype)

        self.radialnn = MLP.MLP(num_output = (4*self.config.prmaxl+2) * self.config.nwave, num_blocks = self.config.radial_nl[0], features = self.config.radial_nl[1], layers_per_block = self.config.radial_nl[2], use_linear = self.config.radial_nl[3], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype)

        self.MPNN_list=[MLP.MLP(num_output = (self.config.prmaxl + self.config.rmaxl)*self.config.nwave, num_blocks = self.config.MP_nl[0], features = self.config.MP_nl[1], layers_per_block = self.config.MP_nl[2], use_linear = self.config.MP_nl[3], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.MP_loop)]

        self.ead_list=[MLP.MLP(num_output = 3*self.config.prmaxl*self.config.nwave, num_blocks = self.config.MP_nl[0], features = self.config.MP_nl[1], layers_per_block = self.config.MP_nl[2], use_linear = self.config.MP_nl[3], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.MP_loop)]

        self.outnn=MLP.MLP(num_output = 1, num_blocks = self.config.out_nl[0], features = self.config.out_nl[1], layers_per_block = self.config.out_nl[2], use_linear=self.config.out_nl[3], use_bias=self.config.use_bias, bias_init_value = None, cst=self.config.cst, dtype=dtype)


    def __call__(self, cart, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        dtype = self.config.initbias_neigh.dtype
        assert cart.dtype == dtype, f"Input cart dtype {cart.dtype} must match config dtype {dtype}"

        rmaxl_i, prmaxl_i = self.config.rmaxl, self.config.prmaxl
        pnorb_i = prmaxl_i**2
        nwave_i = self.config.nwave

        prmaxl_f = jnp.array(prmaxl_i, dtype=dtype)
        nwave_f = jnp.array(nwave_i, dtype=dtype)
        cutoff_f = jnp.array(self.config.cutoff, dtype=dtype)
        pn_f = jnp.array(self.config.pn, dtype=dtype)
        dtype_1 = jnp.array(1.0, dtype=dtype)
        dtype_2 = jnp.array(2.0, dtype=dtype)
        dtype_3 = jnp.array(3.0, dtype=dtype)
        eps = jnp.array(1e-8, dtype=dtype)

        numatom = cart.shape[0]
        symm_cell = (disp_cell + disp_cell.T) / dtype_2
        cart = cart + jnp.matmul(cart, symm_cell)
        cell = cell + jnp.matmul(cell, symm_cell)
        shiftimage = jnp.matmul(shiftimage, cell)

        spec_emb = jnp.less(jnp.abs(species[:, None] - self.config.reduce_spec), 0.5).astype(neighlist.dtype)
        spec_indices = jnp.argmax(spec_emb.astype(dtype), axis=1)

        expand_cart = cart[neighlist]
        distvec = expand_cart[1] - expand_cart[0] + shiftimage
        distsq = jnp.sum(jnp.square(distvec), axis=1)
        judge = distsq > eps
        neigh_factor = judge.astype(dtype)
        distances = jnp.sqrt(distsq + eps)
        sph = self.sph_cal(distvec.T / distances)
        sph_norm = jnp.zeros((rmaxl_i, sph.shape[1]), dtype=dtype).at[self.config.index_l].add(jnp.square(sph))
        sph_norm = sph_norm + eps
        sph = sph / jnp.sqrt(sph_norm[self.config.index_l]) * jnp.sqrt(dtype_2 * self.config.index_l[:, None] + dtype_1)

        norm_dist = distances / cutoff_f
        dist_pow = jnp.power(norm_dist, pn_f)
        poly_env = dtype_1 - dist_pow * ((pn_f + dtype_1) * (pn_f + dtype_2) / dtype_2 - pn_f * (pn_f + dtype_2) * norm_dist + pn_f * (pn_f + dtype_1) / dtype_2 * norm_dist * norm_dist)
        cut_func = poly_env * poly_env * neigh_factor

        ave_neigh = jnp.zeros(numatom, dtype=dtype).at[neighlist[0]].add(cut_func)
        ave_neigh = ave_neigh[:, None] + eps

        expand_spec = species[neighlist].T
        spec_emb = jnp.less(jnp.sum(jnp.abs(expand_spec[:, None] - self.config.com_spec), axis=-1), 0.5).astype(neighlist.dtype)
        expand_indices = jnp.argmax(spec_emb.astype(dtype), axis=1)
        pair_spec = self.neighcoeffnn(self.config.com_spec)

        emb_coeff = self.neighnn(pair_spec)[expand_indices]
        init_ead = self.rweightnn(pair_spec)[expand_indices]
        smooth_ead = init_ead * cut_func[:, None]
        radial_func = jnp.sinc(norm_dist[:, None] * emb_coeff) * cut_func[:, None] * dtype_2
        radial_func = jnp.concatenate((smooth_ead[:, nwave_i:], radial_func), axis=1)

        wradial = self.radialnn(radial_func).reshape(-1, 4*prmaxl_i+2, nwave_i)
        ead = jnp.concatenate((smooth_ead[:, :nwave_i], wradial[:, -1]), axis=1)
        density = jnp.zeros((numatom, nwave_i), dtype=dtype).at[neighlist[0]].add(wradial[:, -2])

        pindex_l = self.config.index_l[:pnorb_i]
        worbital = jnp.einsum("ijk, ji -> ijk", wradial[:, pindex_l], sph[:pnorb_i])
        center_orbital = jnp.zeros((numatom, pnorb_i, nwave_i), dtype=dtype).at[neighlist[0]].add(worbital)
        center_orbital = jnp.einsum("ikm, ijk ->ijm", (self.spec_coeff / jnp.sqrt(nwave_f))[spec_indices], center_orbital / ave_neigh[:, None])

        # --- Message Passing Loop using a standard Python for-loop ---
        radial = wradial[:, prmaxl_i:-2].reshape(-1, 3, prmaxl_i, nwave_i)

        for iter_loop in range(self.config.MP_loop):
            mp_model = self.MPNN_list[iter_loop]

            norm_corb = center_orbital * (self.config.ens_cg[:pnorb_i, None] / jnp.sqrt(prmaxl_f))
            ead1 = jnp.einsum("ji, ijk, ijk -> ik", sph[:pnorb_i], radial[:, 0, pindex_l], norm_corb[neighlist[0]])
            ead2 = jnp.einsum("ji, ijk, ijk -> ik", sph[:pnorb_i], radial[:, 1, pindex_l], norm_corb[neighlist[1]])
            norm_ead = (ead1 + ead2) / jnp.sqrt(dtype_2)
            ead = jnp.concatenate((ead, norm_ead), axis=1)

            orbital = jnp.einsum("ijk, ji -> ijk", radial[:, 2, pindex_l], sph[:pnorb_i])
            sum_orb = jnp.zeros_like(center_orbital).at[neighlist[0]].add(orbital)
            density1 = jnp.sum(sum_orb * norm_corb, axis=1)
            density = jnp.concatenate((density, density1), axis=1)

            orb_coeff = mp_model(ead).reshape(-1, self.config.prmaxl + self.config.rmaxl, self.config.nwave)
            contract_coeff_iter = (self.contract_coeff / jnp.sqrt(nwave_f))[iter_loop, spec_indices]
            l_coeff_iter = self.l_coeff[iter_loop, spec_indices]

            center_orbital = self.sum_interaction(
                prmaxl_i = prmaxl_i,
                nwave_i = nwave_i,
                center_orbital=center_orbital,
                contract_coeff=contract_coeff_iter,
                l_coeff=l_coeff_iter,
                orb_coeff=orb_coeff,
                neighlist=neighlist,
                ave_neigh=ave_neigh,
                pindex_l=pindex_l,
                sph=sph,
                dtype_2=dtype_2
            )

            if self.config.use_norm:
                norm_factor = jnp.sqrt(jnp.sum(jnp.square(center_orbital)) / (jnp.sum(center_factor) * pnorb_i * nwave_f))
                center_orbital = center_orbital / norm_factor

            radial = self.ead_list[iter_loop](ead).reshape(-1, 3, self.config.prmaxl, self.config.nwave)
        # --- End of Message Passing Loop ---

        norm_corb = center_orbital * (self.config.ens_cg[:pnorb_i, None] / jnp.sqrt(prmaxl_f * dtype_3))
        orbital = jnp.einsum("iljk, ji -> ijk", radial[:, :, pindex_l], sph[:pnorb_i])
        sum_orb = jnp.zeros_like(center_orbital).at[neighlist[0]].add(orbital)
        density1 = jnp.sum(sum_orb * norm_corb, axis=1)
        density = jnp.concatenate((density, density1), axis=1)

        scale = self.scale.reshape(-1, 2)[spec_indices]
        atomic_ene = self.outnn(density / ave_neigh).reshape(-1)
        atomic_ene = atomic_ene*scale[:, 0] + scale[:, 1]

        return jnp.sum(atomic_ene * center_factor) * jnp.array(self.config.std, dtype=dtype)


    def sum_interaction(self, prmaxl_i, nwave_i, center_orbital, contract_coeff, l_coeff, orb_coeff, neighlist, ave_neigh, pindex_l, sph, dtype_2):
        corbital = jnp.einsum("ijk, ikm -> ijm", center_orbital, contract_coeff[:, 0])
        iter_orb = jnp.zeros_like(center_orbital).at[neighlist[0]].add(corbital[neighlist[1]] * orb_coeff[:, pindex_l])

        orb_coeff = orb_coeff[:, prmaxl_i:]
        worbital = jnp.einsum("ijk, ji ->ijk", orb_coeff[:, self.config.index_l], sph)
        init_orb = jnp.zeros_like(worbital, shape=(center_orbital.shape[0], *worbital.shape[1:])).at[neighlist[0]].add(worbital)

        inter_orbital = jnp.einsum("ikj, ikj, k -> ikj", init_orb[:, self.config.index_i1], iter_orb[:, self.config.index_i2], self.config.ens_cg)

        mp_orbital = jnp.zeros((center_orbital.shape[0], self.config.index_add.shape[0], nwave_i), dtype=center_orbital.dtype).at[:, self.config.index_den].add(inter_orbital)

        iter_orb = jnp.zeros_like(center_orbital).at[:, self.config.index_add].add(mp_orbital * l_coeff[:, self.config.index_squ])
        norm = ave_neigh * ave_neigh * jnp.sqrt(self.config.count_l[pindex_l])
        iter_orb = jnp.einsum("ij, ijk, ikm -> ijm", jnp.reciprocal(norm), iter_orb, contract_coeff[:, 1])

        center_orbital = jnp.einsum("ijk, ikm -> ijm", center_orbital, contract_coeff[:, 2])

        center_orbital = (center_orbital + iter_orb) / jnp.sqrt(dtype_2)

        return center_orbital
