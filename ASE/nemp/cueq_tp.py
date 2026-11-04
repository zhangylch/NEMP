import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


LAYOUT = cue.IrrepsLayout.ir_mul
UNIFORM_LAYOUT = cue.IrrepsLayout.mul_ir


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
    vector_rep = cuex.RepArray(
        cue.Irreps("O3", "1o"),
        vectors,
        LAYOUT,
    )
    return cuex.spherical_harmonics(
        list(range(max_l)),
        vector_rep,
        normalize=False,
    ).array


def parity_irreps(max_l, mul):
    terms = [f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_l)]
    return cue.Irreps("O3", " + ".join(terms))


def _scalar_path_infos(rmaxl, prmaxl):
    descriptor = cue.descriptors.fully_connected_tensor_product(
        parity_irreps(rmaxl, 1),
        parity_irreps(prmaxl, 1),
        parity_irreps(prmaxl, 1),
    )
    stp = descriptor.polynomial.operations[0][1]
    stp = stp.normalize_paths_for_operand(1)

    count_l = [0] * prmaxl
    infos = []
    for weight_idx, path in enumerate(stp.paths):
        init_l = int(path.indices[1])
        iter_l = int(path.indices[2])
        out_l = int(path.indices[3])
        coefficients = np.asarray(path.coefficients)
        count_l[out_l] += 1
        infos.append((weight_idx, init_l, iter_l, out_l, coefficients))
    return tuple(infos), tuple(count_l)


def _sparse_paths_from_infos(path_infos):
    paths = []
    for weight_idx, init_l, iter_l, out_l, coefficients in path_infos:
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
    return tuple(paths)


def sparse_cg_paths(rmaxl, prmaxl):
    path_infos, count_l = _scalar_path_infos(rmaxl, prmaxl)
    return _sparse_paths_from_infos(path_infos), count_l


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


def _uniform_1d_polynomial(nwave, rmaxl, prmaxl, path_infos):
    init_irreps = parity_irreps(rmaxl, nwave)
    iter_irreps = parity_irreps(prmaxl, nwave)

    stp = cue.SegmentedTensorProduct.from_subscripts("uv,ui,jv,kv+ijk")
    for l_value in range(rmaxl):
        stp.add_segment(1, (nwave, 2 * l_value + 1))
    for l_value in range(prmaxl):
        stp.add_segment(2, (2 * l_value + 1, nwave))
        stp.add_segment(3, (2 * l_value + 1, nwave))

    for _, init_l, iter_l, out_l, coefficients in path_infos:
        stp.add_path(
            None,
            init_l,
            iter_l,
            out_l,
            c=coefficients,
            dims={"u": nwave, "v": nwave},
        )

    stp = stp.flatten_modes("u")
    polynomial = cue.SegmentedPolynomial(
        stp.operands[:3],
        (stp.operands[3],),
        [(cue.Operation((0, 1, 2, 3)), stp)],
    )
    descriptor = cue.EquivariantPolynomial(
        [
            cue.IrrepsAndLayout(cue.Irreps("O3", f"{stp.operands[0].size}x0e"), LAYOUT),
            cue.IrrepsAndLayout(init_irreps, UNIFORM_LAYOUT),
            cue.IrrepsAndLayout(iter_irreps, LAYOUT),
        ],
        [cue.IrrepsAndLayout(iter_irreps, LAYOUT)],
        polynomial,
    )
    ir_dict_polynomial = (
        descriptor.split_operand_by_irrep(2)
        .split_operand_by_irrep(1)
        .split_operand_by_irrep(-1)
        .polynomial
    )
    return init_irreps, iter_irreps, descriptor, ir_dict_polynomial


class RadialMixedTP(nnx.Module):
    def __init__(
        self,
        nspec,
        nwave,
        rmaxl,
        prmaxl,
        dtype,
        tp_method="custom",
        tp_mode="full",
        *,
        rngs,
    ):
        tp_method = normalize_tp_method(tp_method)
        tp_mode = normalize_tp_mode(tp_mode)
        uniform_1d = tp_method == "uniform_1d"
        channelwise = tp_mode == "channelwise"
        if uniform_1d and channelwise:
            raise ValueError("tp_method='uniform_1d' currently supports tp_mode='full' only.")

        path_infos, count_l = _scalar_path_infos(rmaxl, prmaxl)
        paths = _sparse_paths_from_infos(path_infos)
        flat_terms = flatten_cg_paths(paths)
        num_weight_paths = len(paths)

        init_irreps = None
        iter_irreps = None
        descriptor = None
        ir_dict_polynomial = None
        weight_operand = None
        init_descriptors = None
        iter_descriptors = None
        output_ir_descriptors = None
        if uniform_1d:
            init_irreps, iter_irreps, descriptor, ir_dict_polynomial = _uniform_1d_polynomial(
                nwave,
                rmaxl,
                prmaxl,
                path_infos,
            )
            num_init = len(init_irreps)
            weight_operand = ir_dict_polynomial.inputs[0]
            init_descriptors = tuple(ir_dict_polynomial.inputs[1 : 1 + num_init])
            iter_descriptors = tuple(ir_dict_polynomial.inputs[1 + num_init :])
            output_ir_descriptors = tuple(zip(iter_irreps, ir_dict_polynomial.outputs))

        self.nwave = nnx.static(nwave)
        self.rmaxl = nnx.static(rmaxl)
        self.prmaxl = nnx.static(prmaxl)
        self.tp_method = nnx.static(tp_method)
        self.tp_mode = nnx.static(tp_mode)
        self.uniform_1d = nnx.static(uniform_1d)
        self.channelwise = nnx.static(channelwise)
        self.init_channel_first = nnx.static(uniform_1d)
        self.flat_terms = nnx.static(_StaticFlatTerms(flat_terms, dtype))
        self.count_l = nnx.static(count_l)
        self.init_irreps = nnx.static(init_irreps)
        self.iter_irreps = nnx.static(iter_irreps)
        self.descriptor = nnx.static(descriptor)
        self.ir_dict_polynomial = nnx.static(ir_dict_polynomial)
        self.weight_operand = nnx.static(weight_operand)
        self.init_descriptors = nnx.static(init_descriptors)
        self.iter_descriptors = nnx.static(iter_descriptors)
        self.output_ir_descriptors = nnx.static(output_ir_descriptors)
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

    def _to_ir_dict(self, orb, max_l, irreps, layout, descriptors):
        num_nodes = orb.shape[0]
        result = {}
        for l_value, ((_, ir), desc) in enumerate(zip(irreps, descriptors)):
            if layout == UNIFORM_LAYOUT:
                segment = orb[:, :, l_value * l_value : (l_value + 1) * (l_value + 1)]
            else:
                segment = orb[:, l_value * l_value : (l_value + 1) * (l_value + 1), :]
            result[ir] = segment.reshape(
                (num_nodes, desc.num_segments) + desc.segment_shape
            )
        return result

    def _from_ir_dict(self, values, irreps):
        chunks = []
        for l_value, (_, ir) in enumerate(irreps):
            segment = values[ir]
            num_nodes = segment.shape[0]
            chunks.append(segment.reshape(num_nodes, 2 * l_value + 1, self.nwave))
        return jnp.concatenate(chunks, axis=1)

    def _call_uniform_1d(self, init_orb, iter_orb, spec_indices, dtype):
        num_nodes = init_orb.shape[0]
        polynomial = self.ir_dict_polynomial
        weights = (self.weights[spec_indices] * self.weight_norm).reshape(
            (num_nodes, self.weight_operand.num_segments) + self.weight_operand.segment_shape
        )
        init_dict = self._to_ir_dict(
            init_orb,
            self.rmaxl,
            self.init_irreps,
            UNIFORM_LAYOUT,
            self.init_descriptors,
        )
        iter_dict = self._to_ir_dict(
            iter_orb,
            self.prmaxl,
            self.iter_irreps,
            LAYOUT,
            self.iter_descriptors,
        )
        out_template = {
            ir: jax.ShapeDtypeStruct(
                (num_nodes, desc.num_segments) + desc.segment_shape,
                jnp.dtype(dtype),
            )
            for (_, ir), desc in self.output_ir_descriptors
        }
        output = cuex.ir_dict.segmented_polynomial_uniform_1d(
            polynomial,
            [weights, init_dict, iter_dict],
            out_template,
            math_dtype=jnp.dtype(dtype).name,
        )
        return self._from_ir_dict(output, self.iter_irreps)

    def _call_custom(self, init_orb, iter_orb, spec_indices):
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
        if self.uniform_1d:
            return self._call_uniform_1d(init_orb, iter_orb, spec_indices, dtype)

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
        return self._call_custom(init_orb, iter_orb, spec_indices)
