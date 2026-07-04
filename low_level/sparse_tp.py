import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.ops import segment_sum

from low_level import cg_cal
from low_level import sph_cal


def normalize_tp_method(tp_method):
    method = tp_method.lower()
    if method == "custom":
        return "custom"
    if method in ("uniform_1d", "uniform1d", "uniform-1d"):
        return "uniform_1d"
    raise ValueError(
        f"Unsupported tp_method {tp_method!r}; expected 'custom' or 'uniform_1d'."
    )


def normalize_tp_mode(tp_mode):
    mode = tp_mode.lower()
    if mode in ("full", "full_mixing", "full-mixing"):
        return "full"
    if mode in ("channelwise", "channel-wise", "channel"):
        return "channelwise"
    raise ValueError(
        f"Unsupported tp_mode {tp_mode!r}; expected 'full' or 'channelwise'."
    )


def orbital_index_l(max_l):
    index_l = jnp.arange(max_l * max_l)
    for l in range(max_l):
        index_l = index_l.at[l * l : (l + 1) * (l + 1)].set(l)
    return index_l


def normalized_spherical_harmonics(max_l, vectors, index_l, eps):
    sph = sph_cal.SPH_CAL(max_l=max_l - 1)(vectors.T)
    sph_norm = segment_sum(jnp.square(sph), index_l, num_segments=max_l) + eps
    sph = sph / jnp.sqrt(sph_norm[index_l]) * jnp.sqrt((2 * index_l + 1)[:, None])
    return sph.T


def sparse_cg_paths(rmaxl, prmaxl):
    path_specs = []
    count_l = [0] * prmaxl
    for out_l in range(prmaxl):
        for init_l in range(rmaxl):
            low = abs(init_l - out_l)
            high = min(prmaxl, init_l + out_l + 1)
            for iter_l in range(low, high):
                if (init_l + iter_l + out_l) % 2 != 0:
                    continue
                path_specs.append((init_l, iter_l, out_l))
                count_l[out_l] += 1

    paths = []
    for weight_idx, (init_l, iter_l, out_l) in enumerate(path_specs):
        coefficients = cg_cal.clebsch_gordan(init_l, iter_l, out_l)
        coefficients = coefficients / np.sqrt(count_l[out_l])
        mi, mj, mk = np.nonzero(np.abs(coefficients) > 1e-12)
        paths.append(
            (
                weight_idx,
                init_l,
                iter_l,
                out_l,
                tuple(int(value) for value in mi),
                tuple(int(value) for value in mj),
                tuple(int(value) for value in mk),
                tuple(float(coefficients[i, j, k]) for i, j, k in zip(mi, mj, mk)),
            )
        )
    return tuple(paths), tuple(count_l)


def flatten_cg_paths(paths):
    path_indices = []
    init_indices = []
    iter_indices = []
    out_indices = []
    coefficients = []
    for weight_idx, init_l, iter_l, out_l, mi, mj, mk, path_coefficients in paths:
        path_indices.extend([weight_idx] * len(path_coefficients))
        init_indices.extend(init_l * init_l + value for value in mi)
        iter_indices.extend(iter_l * iter_l + value for value in mj)
        out_indices.extend(out_l * out_l + value for value in mk)
        coefficients.extend(path_coefficients)
    return (
        tuple(path_indices),
        tuple(init_indices),
        tuple(iter_indices),
        tuple(out_indices),
        tuple(coefficients),
    )


def _readonly_array(values, dtype):
    array = np.asarray(values, dtype=dtype)
    array.setflags(write=False)
    return array


class _StaticFlatTerms:
    __slots__ = ("path_idx", "init_idx", "iter_idx", "out_idx", "cg_coefficients")

    def __init__(self, flat_terms, dtype):
        path_idx, init_idx, iter_idx, out_idx, coefficients = flat_terms
        self.path_idx = _readonly_array(path_idx, np.int32)
        self.init_idx = _readonly_array(init_idx, np.int32)
        self.iter_idx = _readonly_array(iter_idx, np.int32)
        self.out_idx = _readonly_array(out_idx, np.int32)
        self.cg_coefficients = _readonly_array(coefficients, np.dtype(dtype))


class RadialMixedTP(nnx.Module):
    def __init__(
        self,
        nspec,
        nwave,
        rmaxl,
        prmaxl,
        dtype,
        tp_mode="full",
        *,
        rngs,
    ):
        tp_mode = normalize_tp_mode(tp_mode)
        channelwise = tp_mode == "channelwise"
        paths, count_l = sparse_cg_paths(rmaxl, prmaxl)
        flat_terms = flatten_cg_paths(paths)
        num_weight_paths = len(paths)

        self.nwave = nnx.static(nwave)
        self.rmaxl = nnx.static(rmaxl)
        self.prmaxl = nnx.static(prmaxl)
        self.tp_mode = nnx.static(tp_mode)
        self.channelwise = nnx.static(channelwise)
        self.init_channel_first = nnx.static(False)
        self.flat_terms = nnx.static(_StaticFlatTerms(flat_terms, dtype))
        self.count_l = nnx.static(count_l)
        self.num_paths = nnx.static(num_weight_paths)
        self.weight_norm = nnx.static(1.0 if channelwise else 1.0 / np.sqrt(nwave))
        if channelwise:
            weight_shape = (nspec, num_weight_paths, nwave)
        else:
            weight_shape = (nspec, num_weight_paths, nwave, nwave)
        self.weight_dim = nnx.static(int(np.prod(weight_shape[1:])))
        self.weights = nnx.Param(
            nnx.initializers.normal(1.0)(
                rngs.params(),
                weight_shape,
                dtype,
            )
        )
        if channelwise:
            eye = jnp.eye(nwave, dtype=dtype)
            self.init_mix = nnx.Param(
                jnp.tile(eye[None, None, :, :], (nspec, rmaxl, 1, 1))
            )
            self.iter_mix = nnx.Param(
                jnp.tile(eye[None, None, :, :], (nspec, prmaxl, 1, 1))
            )

    def _apply_channel_mix(self, orb, mix, max_l):
        chunks = []
        for l_value in range(max_l):
            segment = orb[:, l_value * l_value : (l_value + 1) * (l_value + 1), :]
            chunks.append(jnp.einsum("nmu,nuv->nmv", segment, mix[:, l_value]))
        return jnp.concatenate(chunks, axis=1)

    def _call_sparse(self, init_orb, iter_orb, spec_indices):
        num_nodes = init_orb.shape[0]
        dtype = init_orb.dtype
        weights = self.weights[spec_indices] * self.weight_norm
        output = jnp.zeros(
            (num_nodes, self.prmaxl * self.prmaxl, self.nwave),
            dtype=dtype,
        )
        terms = self.flat_terms
        init_terms = init_orb[:, terms.init_idx, :]
        iter_terms = iter_orb[:, terms.iter_idx, :]
        if self.channelwise:
            path_output = jnp.einsum(
                "ntu,ntu,ntu,t->ntu",
                weights[:, terms.path_idx],
                init_terms,
                iter_terms,
                terms.cg_coefficients,
            )
        else:
            path_output = jnp.einsum(
                "ntuv,ntu,ntv,t->ntv",
                weights[:, terms.path_idx],
                init_terms,
                iter_terms,
                terms.cg_coefficients,
            )
        node_indices = jnp.arange(num_nodes)[:, None]
        output = output.at[node_indices, terms.out_idx[None, :], :].add(path_output)
        return output

    def __call__(self, init_orb, iter_orb, spec_indices, dtype):
        if self.channelwise:
            init_orb = self._apply_channel_mix(
                init_orb,
                self.init_mix[spec_indices],
                self.rmaxl,
            )
            iter_orb = self._apply_channel_mix(
                iter_orb,
                self.iter_mix[spec_indices],
                self.prmaxl,
            )
        return self._call_sparse(init_orb, iter_orb, spec_indices)
