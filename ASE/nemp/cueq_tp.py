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
    if method in ("native", "naive"):
        return "naive"
    if method in ("uniform_1d", "uniform1d", "uniform-1d"):
        return "uniform_1d"
    raise ValueError(
        f"Unsupported tp_method {tp_method!r}; expected 'custom', 'native', or 'uniform_1d'."
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


def _to_static_tuple(array):
    values = np.asarray(array).tolist()
    if isinstance(values, list):
        return tuple(_to_static_tuple(value) for value in values)
    return float(values)


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
    sph = cuex.spherical_harmonics(
        list(range(max_l)),
        vector_rep,
        normalize=False,
    ).array
    return sph


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
        if channelwise and uniform_1d:
            raise ValueError("tp_mode='channelwise' currently supports tp_method='custom' or 'native'.")
        init_layout = UNIFORM_LAYOUT if uniform_1d else LAYOUT
        iter_layout = LAYOUT

        def parity_irreps(max_l, mul):
            terms = [f"{mul}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_l)]
            return cue.Irreps("O3", " + ".join(terms))

        init_irreps = parity_irreps(rmaxl, nwave)
        iter_irreps = parity_irreps(prmaxl, nwave)
        scalar_descriptor = cue.descriptors.fully_connected_tensor_product(
            parity_irreps(rmaxl, 1),
            parity_irreps(prmaxl, 1),
            parity_irreps(prmaxl, 1),
        )
        scalar_stp = scalar_descriptor.polynomial.operations[0][1]

        # Operand 3 is the STP output; i,j,k are the CG coefficient axes.
        # Full mode gives each CG path an nwave x nwave channel matrix.
        # Channelwise mode pre-mixes both inputs, then applies per-channel path weights.
        if channelwise:
            stp = cue.SegmentedTensorProduct.from_subscripts("u,iu,ju,ku+ijk")
            for l_value in range(rmaxl):
                stp.add_segment(1, (2 * l_value + 1, nwave))
            for l_value in range(prmaxl):
                stp.add_segment(2, (2 * l_value + 1, nwave))
                stp.add_segment(3, (2 * l_value + 1, nwave))
        elif uniform_1d:
            stp = cue.SegmentedTensorProduct.from_subscripts("uv,ui,jv,kv+ijk")
            for l_value in range(rmaxl):
                stp.add_segment(1, (nwave, 2 * l_value + 1))
            for l_value in range(prmaxl):
                stp.add_segment(2, (2 * l_value + 1, nwave))
                stp.add_segment(3, (2 * l_value + 1, nwave))
        else:
            stp = cue.SegmentedTensorProduct.from_subscripts("uv,iu,jv,kv+ijk")
            for l_value in range(rmaxl):
                stp.add_segment(1, (2 * l_value + 1, nwave))
            for l_value in range(prmaxl):
                stp.add_segment(2, (2 * l_value + 1, nwave))
                stp.add_segment(3, (2 * l_value + 1, nwave))
        for path in scalar_stp.paths:
            stp.add_path(
                None,
                path.indices[1],
                path.indices[2],
                path.indices[3],
                c=path.coefficients,
                dims={"u": nwave} if channelwise else {"u": nwave, "v": nwave},
            )
        stp = stp.normalize_paths_for_operand(1)
        num_weight_paths = stp.num_paths
        custom_paths = ()
        if tp_method == "custom":
            custom_paths = tuple(
                (
                    int(path.indices[0]),
                    int(path.indices[1]),
                    int(path.indices[2]),
                    int(path.indices[3]),
                    _to_static_tuple(path.coefficients),
                )
                for path in stp.paths
            )
        polynomial_stp = stp
        if uniform_1d:
            polynomial_stp = stp.flatten_modes("u")

        polynomial = cue.SegmentedPolynomial(
            polynomial_stp.operands[:3],
            (polynomial_stp.operands[3],),
            [(cue.Operation((0, 1, 2, 3)), polynomial_stp)],
        )
        descriptor = cue.EquivariantPolynomial(
            [
                cue.IrrepsAndLayout(cue.Irreps("O3", f"{polynomial_stp.operands[0].size}x0e"), LAYOUT),
                cue.IrrepsAndLayout(init_irreps, init_layout),
                cue.IrrepsAndLayout(iter_irreps, iter_layout),
            ],
            [cue.IrrepsAndLayout(iter_irreps, iter_layout)],
            polynomial,
        )
        ir_dict_polynomial = None
        if uniform_1d:
            ir_dict_polynomial = (
                descriptor.split_operand_by_irrep(2)
                .split_operand_by_irrep(1)
                .split_operand_by_irrep(-1)
                .polynomial
            )

        self.nwave = nnx.static(nwave)
        self.rmaxl = nnx.static(rmaxl)
        self.prmaxl = nnx.static(prmaxl)
        self.init_irreps = nnx.static(init_irreps)
        self.iter_irreps = nnx.static(iter_irreps)
        self.descriptor = nnx.static(descriptor)
        self.ir_dict_polynomial = nnx.static(ir_dict_polynomial)
        self.custom_paths = nnx.static(custom_paths)
        self.tp_method = nnx.static(tp_method)
        self.tp_mode = nnx.static(tp_mode)
        self.uniform_1d = nnx.static(uniform_1d)
        self.channelwise = nnx.static(channelwise)
        self.num_paths = nnx.static(num_weight_paths)
        self.weight_dim = nnx.static(descriptor.inputs[0].dim)
        if channelwise:
            weight_shape = (nspec, num_weight_paths, nwave)
        else:
            weight_shape = (nspec, num_weight_paths, nwave, nwave)
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

    def _to_rep_tensor(self, orb, max_l, layout):
        num_nodes = orb.shape[0]
        if layout == LAYOUT:
            return orb.reshape(num_nodes, max_l * max_l * self.nwave)
        chunks = []
        for l_value in range(max_l):
            segment = orb[:, l_value * l_value : (l_value + 1) * (l_value + 1), :]
            chunks.append(jnp.swapaxes(segment, 1, 2).reshape(num_nodes, -1))
        return jnp.concatenate(chunks, axis=1)

    def _from_rep_tensor(self, flat, num_nodes, layout):
        if layout == LAYOUT:
            return flat.reshape(num_nodes, self.prmaxl * self.prmaxl, self.nwave)
        chunks = []
        offset = 0
        for l_value in range(self.prmaxl):
            width = self.nwave * (2 * l_value + 1)
            segment = flat[:, offset : offset + width].reshape(
                num_nodes, self.nwave, 2 * l_value + 1
            )
            chunks.append(jnp.swapaxes(segment, 1, 2))
            offset += width
        return jnp.concatenate(chunks, axis=1)

    def _to_ir_dict(self, orb, max_l, irreps, layout, descriptors):
        num_nodes = orb.shape[0]
        result = {}
        for l_value, ((_, ir), desc) in enumerate(zip(irreps, descriptors)):
            segment = orb[:, l_value * l_value : (l_value + 1) * (l_value + 1), :]
            if layout == UNIFORM_LAYOUT:
                segment = jnp.swapaxes(segment, 1, 2)
            result[ir] = segment.reshape(
                (num_nodes, desc.num_segments) + desc.segment_shape
            )
        return result

    def _from_ir_dict(self, values, irreps, layout):
        chunks = []
        for l_value, (_, ir) in enumerate(irreps):
            segment = values[ir]
            num_nodes = segment.shape[0]
            if layout == LAYOUT:
                segment = segment.reshape(num_nodes, 2 * l_value + 1, self.nwave)
            else:
                segment = segment.reshape(num_nodes, self.nwave, 2 * l_value + 1)
                segment = jnp.swapaxes(segment, -2, -1)
            chunks.append(segment)
        return jnp.concatenate(chunks, axis=1)

    def _apply_channel_mix(self, orb, mix, max_l):
        chunks = []
        for l_value in range(max_l):
            segment = orb[:, l_value * l_value : (l_value + 1) * (l_value + 1), :]
            chunks.append(jnp.einsum("nmu,nuv->nmv", segment, mix[:, l_value]))
        return jnp.concatenate(chunks, axis=1)

    def _call_uniform_1d(self, init_orb, iter_orb, spec_indices, dtype):
        num_nodes = init_orb.shape[0]
        polynomial = self.ir_dict_polynomial
        weight_operand = polynomial.inputs[0]
        weights = self.weights[spec_indices].reshape(
            (num_nodes, weight_operand.num_segments) + weight_operand.segment_shape
        )
        num_init = len(self.init_irreps)
        init_descriptors = polynomial.inputs[1 : 1 + num_init]
        iter_descriptors = polynomial.inputs[1 + num_init :]
        init_dict = self._to_ir_dict(
            init_orb,
            self.rmaxl,
            self.init_irreps,
            self.descriptor.inputs[1].layout,
            init_descriptors,
        )
        iter_dict = self._to_ir_dict(
            iter_orb,
            self.prmaxl,
            self.iter_irreps,
            self.descriptor.inputs[2].layout,
            iter_descriptors,
        )
        out_template = {
            ir: jax.ShapeDtypeStruct(
                (num_nodes, desc.num_segments) + desc.segment_shape,
                jnp.dtype(dtype),
            )
            for (_, ir), desc in zip(self.iter_irreps, polynomial.outputs)
        }
        output = cuex.ir_dict.segmented_polynomial_uniform_1d(
            polynomial,
            [weights, init_dict, iter_dict],
            out_template,
            math_dtype=jnp.dtype(dtype).name,
        )
        return self._from_ir_dict(
            output,
            self.iter_irreps,
            self.descriptor.outputs[0].layout,
        )

    def _call_custom(self, init_orb, iter_orb, spec_indices):
        num_nodes = init_orb.shape[0]
        dtype = init_orb.dtype
        weights = self.weights[spec_indices]
        output = jnp.zeros(
            (num_nodes, self.prmaxl * self.prmaxl, self.nwave),
            dtype=dtype,
        )
        for weight_idx, init_l, iter_l, out_l, coefficients in self.custom_paths:
            init_segment = init_orb[
                :, init_l * init_l : (init_l + 1) * (init_l + 1), :
            ]
            iter_segment = iter_orb[
                :, iter_l * iter_l : (iter_l + 1) * (iter_l + 1), :
            ]
            cg = jnp.asarray(coefficients, dtype=dtype)
            if self.channelwise:
                path_output = jnp.einsum(
                    "nu,niu,nju,ijk->nku",
                    weights[:, weight_idx],
                    init_segment,
                    iter_segment,
                    cg,
                )
            else:
                path_output = jnp.einsum(
                    "nuv,niu,njv,ijk->nkv",
                    weights[:, weight_idx],
                    init_segment,
                    iter_segment,
                    cg,
                )
            output = output.at[
                :, out_l * out_l : (out_l + 1) * (out_l + 1), :
            ].add(path_output)
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

        if self.tp_method == "custom":
            return self._call_custom(init_orb, iter_orb, spec_indices)

        if self.uniform_1d:
            return self._call_uniform_1d(init_orb, iter_orb, spec_indices, dtype)

        num_nodes = init_orb.shape[0]
        weights = self.weights[spec_indices].reshape(num_nodes, self.weight_dim)

        weight_rep = cuex.RepArray(self.descriptor.inputs[0], weights, LAYOUT)
        init_rep = cuex.RepArray(
            self.init_irreps,
            self._to_rep_tensor(init_orb, self.rmaxl, self.descriptor.inputs[1].layout),
            self.descriptor.inputs[1].layout,
        )
        iter_rep = cuex.RepArray(
            self.iter_irreps,
            self._to_rep_tensor(iter_orb, self.prmaxl, self.descriptor.inputs[2].layout),
            self.descriptor.inputs[2].layout,
        )

        output = cuex.equivariant_polynomial(
            self.descriptor,
            [weight_rep, init_rep, iter_rep],
            method=self.tp_method,
            math_dtype=jnp.dtype(dtype).name,
        )
        return self._from_rep_tensor(output.array, num_nodes, self.descriptor.outputs[0].layout)
