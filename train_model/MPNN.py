import sys
import numpy as np
import jax
import flax
import jax.numpy as jnp
import jax.random as jrm
from jax.numpy import dtype,array
import flax.linen as nn
from jax.ops import segment_sum
from typing import Sequence, List, Union, Optional
from flax.core import freeze
from jax.lax import fori_loop
from jax import Array
#  need to be updated for JAX_MD
from low_level import sph_cal
from src.data_config import ModelConfig
from low_level import MLP  # Refers to the final, robust MLP.py




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

        self.l_coeff = self.param('l_coeff', nn.initializers.normal(1.0), (self.config.MP_loop, self.config.num_cg, self.config.nspec, self.config.nwave), dtype)

        self.neighcoeffnn = MLP.MLP(num_output = self.config.npaircode, num_blocks = self.config.emb_nl[0], features = self.config.emb_nl[1], layers_per_block = self.config.emb_nl[2], use_bias=True, bias_init_value = jnp.ones(self.config.npaircode), cst=self.config.cst, dtype=dtype)

        self.neighnn = MLP.MLP(num_output = self.config.nradial, num_blocks = self.config.emb_nl[0], features = self.config.emb_nl[1], layers_per_block = self.config.emb_nl[2], use_bias=True, bias_init_value = self.config.initbias_neigh, cst=self.config.cst, dtype=dtype)

        self.rweightnn = MLP.MLP(num_output = self.config.nradial + self.config.nwave, num_blocks = self.config.emb_nl[0], features = self.config.emb_nl[1], layers_per_block = self.config.emb_nl[2], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype)

        self.radialnn = MLP.MLP(num_output = (self.config.prmaxl+2) * self.config.nwave, num_blocks = self.config.radial_nl[0], features = self.config.radial_nl[1], layers_per_block = self.config.radial_nl[2], use_linear = self.config.radial_nl[3], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype)

        self.MPNN_list=[MLP.MLP(num_output = (self.config.prmaxl + self.config.rmaxl)*self.config.nwave, num_blocks = self.config.MP_nl[0], features = self.config.MP_nl[1], layers_per_block = self.config.MP_nl[2], use_linear = self.config.MP_nl[3], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.MP_loop)]

        self.ead_list=[MLP.MLP(num_output = 3*self.config.prmaxl*self.config.nwave, num_blocks = self.config.MP_nl[0], features = self.config.MP_nl[1], layers_per_block = self.config.MP_nl[2], use_linear = self.config.MP_nl[3], use_bias=False, bias_init_value = None, cst=self.config.cst, dtype=dtype) for iMP_loop in range(self.config.MP_loop+1)]

        self.outnn=MLP.MLP(num_output = 1, num_blocks = self.config.out_nl[0], features = self.config.out_nl[1], layers_per_block = self.config.out_nl[2], use_linear=self.config.out_nl[3], use_bias=self.config.use_bias, bias_init_value = None, cst=self.config.cst, dtype=dtype)


    def __call__(self, cart, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
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

        nnode = cart.shape[0]
        ngraph = cell.shape[0]
        symm_cell = (disp_cell + disp_cell.transpose(0, 2, 1)) / dtype_2
        cell = cell + jnp.einsum("ijk, ikm -> ijm", cell, symm_cell)
        symm_cell = symm_cell[celllist]
        cart = cart + jnp.einsum("ij, ijk -> ik", cart, symm_cell)
        indexlist = celllist[neighlist[0]]
        expand_cell = cell[indexlist]
        shiftimage = jnp.einsum("ji, ijk -> ik", shiftimage, expand_cell)

        spec_emb = jnp.less(jnp.abs(species[:, None] - self.config.reduce_spec), 0.5).astype(neighlist.dtype)
        spec_indices = jnp.argmax(spec_emb.astype(dtype), axis=1)

        expand_cart = cart[neighlist]
        distvec = expand_cart[1] - expand_cart[0] + shiftimage
        distsq = jnp.sum(jnp.square(distvec), axis=1)
        judge = distsq > eps
        neigh_factor = judge.astype(dtype)
        distances = jnp.sqrt(distsq + eps)
        sph = self.sph_cal(distvec.T / distances)
        sph_norm = segment_sum(jnp.square(sph), self.config.index_l, num_segments=rmaxl_i, indices_are_sorted=True)
        sph_norm = sph_norm + eps
        sph = sph / jnp.sqrt(sph_norm[self.config.index_l]) * jnp.sqrt(dtype_2 * self.config.index_l[:, None] + dtype_1)

        norm_dist = distances / cutoff_f
        dist_pow = jnp.power(norm_dist, pn_f)
        poly_env = dtype_1 - dist_pow * ((pn_f + dtype_1) * (pn_f + dtype_2) / dtype_2 - pn_f * (pn_f + dtype_2) * norm_dist + pn_f * (pn_f + dtype_1) / dtype_2 * norm_dist * norm_dist)
        cut_func = poly_env * poly_env * neigh_factor

        ave_neigh = segment_sum(cut_func, neighlist[0], num_segments=nnode, indices_are_sorted=True)
        ave_neigh = ave_neigh[:, None] + eps

        cn_indices = spec_indices[neighlist]
        pair_spec = self.neighcoeffnn(self.config.com_spec)

        emb_coeff = self.neighnn(pair_spec).reshape(self.config.nspec, self.config.nspec, -1)[cn_indices[0], cn_indices[1]]
        init_ead = self.rweightnn(pair_spec).reshape(self.config.nspec, self.config.nspec, -1)[cn_indices[0], cn_indices[1]]
        smooth_ead = init_ead * cut_func[:, None]
        radial_func = jnp.sinc(norm_dist[:, None] * emb_coeff) * cut_func[:, None]
        radial_func = jnp.concatenate((smooth_ead[:, nwave_i:], radial_func), axis=1)

        wradial = self.radialnn(radial_func).reshape(-1, prmaxl_i+2, nwave_i)
        ead = jnp.concatenate((smooth_ead[:, :nwave_i], wradial[:, -1]), axis=1)
        density = segment_sum(wradial[:, -2], neighlist[0], num_segments=nnode, indices_are_sorted=True)

        pindex_l = self.config.index_l[:pnorb_i]
        worbital = jnp.einsum("ijk, ji -> ijk", wradial[:, pindex_l], sph[:pnorb_i])
        center_orbital = segment_sum(worbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)
        center_orbital = jnp.einsum("ikm, ijk ->ijm", (self.spec_coeff / jnp.sqrt(nwave_f))[spec_indices], center_orbital / ave_neigh[:, None])

        # --- Message Passing Loop using a standard Python for-loop ---
        radial = self.ead_list[-1](ead).reshape(-1, 3, prmaxl_i, nwave_i)

        for iter_loop in range(self.config.MP_loop):


            norm_corb = center_orbital * (self.config.ens_cg[:pnorb_i, None] / jnp.sqrt(prmaxl_f))
            add_orb = radial[:, 0, pindex_l] * norm_corb[neighlist[0]] + radial[:, 1, pindex_l] * norm_corb[neighlist[1]]
            norm_ead = jnp.einsum("ji, ijk -> ik", sph[:pnorb_i], add_orb) / jnp.sqrt(dtype_2)
            ead = jnp.concatenate((ead, norm_ead), axis=1)

            orbital = jnp.einsum("ijk, ji -> ijk", radial[:, 2, pindex_l], sph[:pnorb_i])
            sum_orb = segment_sum(orbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)
            density1 = jnp.sum(sum_orb * norm_corb, axis=1)
            density = jnp.concatenate((density, density1), axis=1)

            orb_coeff = self.MPNN_list[iter_loop](ead).reshape(-1, prmaxl_i+rmaxl_i, self.config.nwave)
            contract_coeff_iter = (self.contract_coeff / jnp.sqrt(nwave_f))[iter_loop, spec_indices]
            l_coeff_iter = self.l_coeff[iter_loop]

            center_orbital = self.sum_interaction(
                nnode = nnode,
                prmaxl_i = prmaxl_i,
                nwave_i = nwave_i,
                center_orbital=center_orbital,
                contract_coeff=contract_coeff_iter,
                l_coeff=l_coeff_iter[:, spec_indices],
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

            radial = self.ead_list[iter_loop](ead).reshape(-1, 3, prmaxl_i, nwave_i)
        # --- End of Message Passing Loop ---

        norm_corb = center_orbital * (self.config.ens_cg[:pnorb_i, None] / jnp.sqrt(prmaxl_f * dtype_3))
        orbital = jnp.einsum("iljk, ji -> ijk", radial[:, :, pindex_l], sph[:pnorb_i])
        sum_orb = segment_sum(orbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)
        density1 = jnp.sum(sum_orb * norm_corb, axis=1)
        density = jnp.concatenate((density, density1), axis=1)

        scale = self.scale.reshape(-1, 2)[spec_indices]
        atomic_ene = self.outnn(density / ave_neigh).reshape(-1)
        atomic_ene = (atomic_ene*scale[:, 0] + scale[:, 1]) * center_factor
        graph_ene = segment_sum(atomic_ene, celllist, num_segments=ngraph, indices_are_sorted=True) * jnp.array(self.config.std, dtype=dtype)

        return jnp.sum(graph_ene), graph_ene 


    def sum_interaction(self, nnode, prmaxl_i, nwave_i, center_orbital, contract_coeff, l_coeff, orb_coeff, neighlist, ave_neigh, pindex_l, sph, dtype_2):
        corbital = jnp.einsum("ijk, ikm -> ijm", center_orbital, contract_coeff[:, 0])
        iter_orb = segment_sum(corbital[neighlist[1]] * orb_coeff[:, pindex_l], neighlist[0], num_segments=nnode, indices_are_sorted=True)

        worbital = jnp.einsum("ijk, ji ->ijk", orb_coeff[:, prmaxl_i+self.config.index_l], sph)
        init_orb = segment_sum(worbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)

        inter_orbital = jnp.einsum("ikj, ikj, k -> kij", init_orb[:, self.config.index_i1], iter_orb[:, self.config.index_i2], self.config.ens_cg)

        mp_orbital = segment_sum(inter_orbital, self.config.index_den, num_segments=self.config.index_add.shape[0], indices_are_sorted=True)

        iter_orb = segment_sum(mp_orbital*l_coeff[self.config.index_squ], self.config.index_add, num_segments=prmaxl_i * prmaxl_i)
        norm = ave_neigh * ave_neigh * jnp.sqrt(self.config.count_l[pindex_l])
        iter_orb = jnp.einsum("ij, jik, ikm -> ijm", jnp.reciprocal(norm), iter_orb, contract_coeff[:, 1])

        center_orbital = jnp.einsum("ijk, ikm -> ijm", center_orbital, contract_coeff[:, 2])

        center_orbital = (center_orbital + iter_orb) / jnp.sqrt(dtype_2)

        return center_orbital
