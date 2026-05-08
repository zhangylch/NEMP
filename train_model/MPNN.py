import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from jax.ops import segment_sum
from collections.abc import Mapping
from low_level import cueq_tp
from src.data_config import ModelConfig
from low_level import MLP


def _as_rngs(rngs):
    if isinstance(rngs, nnx.Rngs):
        return rngs
    if isinstance(rngs, Mapping):
        rngs = rngs.get("params", rngs.get("default"))
    return nnx.Rngs(rngs)


class MPNNCore(nnx.Module):
    """
    NNX implementation of the equivariant message passing neural network.
    """

    def __init__(self, config: ModelConfig, *, rngs: nnx.Rngs):
        self.config = config
        dtype = config.initbias_neigh.dtype

        self.scale = nnx.Param(
            jnp.array(np.array([1.0, 0.0] * config.nspec), dtype=dtype)
        )
        self.spec_coeff = nnx.Param(
            nnx.initializers.normal(1.0)(
                rngs.params(), (config.nspec, config.nwave, config.nwave), dtype
            )
        )
        self.contract_coeff = nnx.Param(
            nnx.initializers.normal(1.0)(
                rngs.params(),
                (config.MP_loop, config.nspec, 2, config.nwave, config.nwave),
                dtype,
            )
        )
        self.tp_layers = nnx.List([
            cueq_tp.RadialMixedTP(
                config.nspec,
                config.nwave,
                config.rmaxl,
                config.prmaxl,
                dtype,
                rngs=rngs,
            )
            for _ in range(config.MP_loop)
        ])

        com_spec_features = config.com_spec.shape[-1]
        self.neighcoeffnn = MLP.MLP(
            in_features=com_spec_features,
            num_output=config.npaircode,
            num_blocks=config.emb_nl[0],
            features=config.emb_nl[1],
            layers_per_block=config.emb_nl[2],
            use_bias=True,
            bias_init_value=jnp.ones(config.npaircode),
            cst=config.cst,
            dtype=dtype,
            rngs=rngs,
        )
        self.neighnn = MLP.MLP(
            in_features=config.npaircode,
            num_output=config.nradial,
            num_blocks=config.emb_nl[0],
            features=config.emb_nl[1],
            layers_per_block=config.emb_nl[2],
            use_bias=True,
            bias_init_value=config.initbias_neigh,
            cst=config.cst,
            dtype=dtype,
            rngs=rngs,
        )
        self.rweightnn = MLP.MLP(
            in_features=config.npaircode,
            num_output=config.nradial + config.nwave,
            num_blocks=config.emb_nl[0],
            features=config.emb_nl[1],
            layers_per_block=config.emb_nl[2],
            use_bias=False,
            bias_init_value=None,
            cst=config.cst,
            dtype=dtype,
            rngs=rngs,
        )
        self.radialnn = MLP.MLP(
            in_features=2 * config.nradial,
            num_output=(config.prmaxl + 2) * config.nwave,
            num_blocks=config.radial_nl[0],
            features=config.radial_nl[1],
            layers_per_block=config.radial_nl[2],
            use_linear=config.radial_nl[3],
            use_bias=False,
            bias_init_value=None,
            cst=config.cst,
            dtype=dtype,
            rngs=rngs,
        )
        self.MPNN_list = nnx.List([
            MLP.MLP(
                in_features=(3 + iMP_loop) * config.nwave,
                num_output=(config.prmaxl + config.rmaxl) * config.nwave,
                num_blocks=config.MP_nl[0],
                features=config.MP_nl[1],
                layers_per_block=config.MP_nl[2],
                use_linear=config.MP_nl[3],
                use_bias=False,
                bias_init_value=None,
                cst=config.cst,
                dtype=dtype,
                rngs=rngs,
            )
            for iMP_loop in range(config.MP_loop)
        ])
        self.ead_list = nnx.List([
            MLP.MLP(
                in_features=(3 + iMP_loop) * config.nwave,
                num_output=3 * config.prmaxl * config.nwave,
                num_blocks=config.MP_nl[0],
                features=config.MP_nl[1],
                layers_per_block=config.MP_nl[2],
                use_linear=config.MP_nl[3],
                use_bias=False,
                bias_init_value=None,
                cst=config.cst,
                dtype=dtype,
                rngs=rngs,
            )
            for iMP_loop in range(config.MP_loop)
        ])
        self.ead_list.append(
            MLP.MLP(
                in_features=2 * config.nwave,
                num_output=3 * config.prmaxl * config.nwave,
                num_blocks=config.MP_nl[0],
                features=config.MP_nl[1],
                layers_per_block=config.MP_nl[2],
                use_linear=config.MP_nl[3],
                use_bias=False,
                bias_init_value=None,
                cst=config.cst,
                dtype=dtype,
                rngs=rngs,
            )
        )
        self.outnn = MLP.MLP(
            in_features=(config.MP_loop + 2) * config.nwave,
            num_output=1,
            num_blocks=config.out_nl[0],
            features=config.out_nl[1],
            layers_per_block=config.out_nl[2],
            use_linear=config.out_nl[3],
            use_bias=config.use_bias,
            bias_init_value=None,
            cst=config.cst,
            dtype=dtype,
            rngs=rngs,
        )

    def __call__(
        self,
        cart,
        cell,
        disp_cell,
        neighlist,
        celllist,
        shiftimage,
        center_factor,
        species,
    ):
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
        eps = jnp.array(1e-6, dtype=dtype)

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
        sph = cueq_tp.normalized_spherical_harmonics(
            rmaxl_i,
            distvec / distances[:, None],
            self.config.index_l,
            eps,
        )

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

        wradial = self.radialnn(radial_func).reshape(-1, prmaxl_i + 2, nwave_i)
        ead = jnp.concatenate((smooth_ead[:, :nwave_i], wradial[:, -1]), axis=1)
        density = segment_sum(wradial[:, -2], neighlist[0], num_segments=nnode, indices_are_sorted=True)

        pindex_l = self.config.index_l[:pnorb_i]
        density_norm = jnp.reciprocal(jnp.sqrt((dtype_2 * pindex_l.astype(dtype) + dtype_1) * prmaxl_f))
        worbital = jnp.einsum("ijk, ij -> ijk", wradial[:, pindex_l], sph[:, :pnorb_i])
        center_orbital = segment_sum(worbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)
        center_orbital = jnp.einsum("ikm, ijk ->ijm", (self.spec_coeff / jnp.sqrt(nwave_f))[spec_indices], center_orbital / ave_neigh[:, None])

        radial = self.ead_list[-1](ead).reshape(-1, 3, prmaxl_i, nwave_i)

        for iter_loop in range(self.config.MP_loop):
            norm_corb = center_orbital * density_norm[:, None]
            add_orb = radial[:, 0, pindex_l] * norm_corb[neighlist[0]] + radial[:, 1, pindex_l] * norm_corb[neighlist[1]]
            norm_ead = jnp.einsum("ij, ijk -> ik", sph[:, :pnorb_i], add_orb) / jnp.sqrt(dtype_2)
            ead = jnp.concatenate((ead, norm_ead), axis=1)

            orbital = jnp.einsum("ijk, ij -> ijk", radial[:, 2, pindex_l], sph[:, :pnorb_i])
            sum_orb = segment_sum(orbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)
            density1 = jnp.sum(sum_orb * norm_corb, axis=1)
            density = jnp.concatenate((density, density1), axis=1)

            orb_coeff = self.MPNN_list[iter_loop](ead).reshape(-1, prmaxl_i + rmaxl_i, self.config.nwave)
            contract_coeff_iter = (self.contract_coeff / jnp.sqrt(nwave_f))[iter_loop, spec_indices]

            center_orbital = self.sum_interaction(
                nnode=nnode,
                prmaxl_i=prmaxl_i,
                center_orbital=center_orbital,
                contract_coeff=contract_coeff_iter,
                tp_layer=self.tp_layers[iter_loop],
                spec_indices=spec_indices,
                orb_coeff=orb_coeff,
                neighlist=neighlist,
                ave_neigh=ave_neigh,
                pindex_l=pindex_l,
                sph=sph,
                dtype_2=dtype_2,
            )

            radial = self.ead_list[iter_loop](ead).reshape(-1, 3, prmaxl_i, nwave_i)
            if self.config.use_norm:
                norm_factor = jnp.einsum("ijk, ijk -> i", center_orbital, center_orbital) * jnp.reciprocal(prmaxl_f * nwave_f)
                center_orbital = center_orbital * jnp.reciprocal(jnp.sqrt(norm_factor + eps))[:, None, None]

        norm_corb = center_orbital * (density_norm[:, None] / jnp.sqrt(prmaxl_f * dtype_3))
        orbital = jnp.einsum("iljk, ij -> ijk", radial[:, :, pindex_l], sph[:, :pnorb_i])
        sum_orb = segment_sum(orbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)
        density1 = jnp.sum(sum_orb * norm_corb, axis=1)
        density = jnp.concatenate((density, density1), axis=1)

        scale = self.scale[...].reshape(-1, 2)[spec_indices]
        atomic_ene = self.outnn(density / ave_neigh).reshape(-1)
        atomic_ene = (atomic_ene * scale[:, 0] + scale[:, 1]) * center_factor
        graph_ene = segment_sum(atomic_ene, celllist, num_segments=ngraph, indices_are_sorted=True) * jnp.array(self.config.std, dtype=dtype)

        return jnp.sum(graph_ene), graph_ene

    def sum_interaction(self, nnode, prmaxl_i, center_orbital, contract_coeff, tp_layer, spec_indices, orb_coeff, neighlist, ave_neigh, pindex_l, sph, dtype_2):
        iter_orb = segment_sum(center_orbital[neighlist[1]] * orb_coeff[:, pindex_l], neighlist[0], num_segments=nnode, indices_are_sorted=True)

        worbital = jnp.einsum("ijk, ij ->ijk", orb_coeff[:, prmaxl_i + self.config.index_l], sph)
        init_orb = segment_sum(worbital, neighlist[0], num_segments=nnode, indices_are_sorted=True)

        iter_orb = tp_layer(
            init_orb,
            iter_orb,
            spec_indices,
            self.config.initbias_neigh.dtype,
        )
        norm = ave_neigh * ave_neigh
        iter_orb = jnp.einsum("ij, ijk, ijkn -> ijn", jnp.reciprocal(norm), iter_orb, contract_coeff[:, :, 0])

        center_orbital = jnp.einsum("ijk, ikm -> ijm", center_orbital, contract_coeff[:, :, 1])
        center_orbital = (center_orbital + iter_orb) / jnp.sqrt(dtype_2)

        return center_orbital


class MPNN:
    """
    Compatibility wrapper exposing the previous Linen-style init/apply API.
    Internally this runs the NNX MPNNCore via split/merge.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.graphdef = self._make_graphdef()

    def _build(self, rngs):
        return MPNNCore(self.config, rngs=rngs)

    def _make_graphdef(self):
        graphdef, _ = nnx.split(self._build(nnx.Rngs(0)), nnx.Param)
        return graphdef

    def init(self, rngs, *args, **kwargs):
        graphdef, params = nnx.split(self._build(_as_rngs(rngs)), nnx.Param)
        self.graphdef = graphdef
        return params

    def _state_from_params(self, params):
        if isinstance(params, Mapping) and "params" in params and len(params) == 1:
            params = params["params"]
        if isinstance(params, nnx.State):
            return params

        state = nnx.state(self._build(nnx.Rngs(0)), nnx.Param)
        nnx.replace_by_pure_dict(state, params)
        return state

    def apply(self, params, *args, **kwargs):
        model = nnx.merge(self.graphdef, self._state_from_params(params))
        return model(*args, **kwargs)
