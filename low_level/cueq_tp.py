import cuequivariance as cue
import cuequivariance_jax as cuex
import jax.numpy as jnp
from flax import nnx


LAYOUT = cue.IrrepsLayout.ir_mul
UNIFORM_LAYOUT = cue.IrrepsLayout.mul_ir


def normalize_tp_method(tp_method):
    method = tp_method.lower()
    if method in ("native", "naive"):
        return "naive"
    if method in ("uniform_1d", "uniform1d", "uniform-1d"):
        return "uniform_1d"
    raise ValueError(
        f"Unsupported tp_method {tp_method!r}; expected 'native' or 'uniform_1d'."
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
    sph = cuex.spherical_harmonics(
        list(range(max_l)),
        vector_rep,
        normalize=False,
    ).array
    return sph


class RadialMixedTP(nnx.Module):
    def __init__(self, nspec, nwave, rmaxl, prmaxl, dtype, tp_method="native", *, rngs):
        tp_method = normalize_tp_method(tp_method)
        uniform_1d = tp_method == "uniform_1d"
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
        # u,v mix input channels and output reuses v.
        # uniform_1d flattens the init-channel mode. Keep coefficient modes
        # first on the iter/output operands so cueq can flatten CG modes too.
        if uniform_1d:
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
                dims={"u": nwave, "v": nwave},
            )
        stp = stp.normalize_paths_for_operand(1)
        num_weight_paths = stp.num_paths
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

        self.nwave = nnx.static(nwave)
        self.rmaxl = nnx.static(rmaxl)
        self.prmaxl = nnx.static(prmaxl)
        self.init_irreps = nnx.static(init_irreps)
        self.iter_irreps = nnx.static(iter_irreps)
        self.descriptor = nnx.static(descriptor)
        self.tp_method = nnx.static(tp_method)
        self.uniform_1d = nnx.static(uniform_1d)
        self.num_paths = nnx.static(num_weight_paths)
        self.weight_dim = nnx.static(descriptor.inputs[0].dim)
        self.weights = nnx.Param(
            nnx.initializers.normal(1.0)(
                rngs.params(),
                (nspec, num_weight_paths, nwave, nwave),
                dtype,
            )
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

    def __call__(self, init_orb, iter_orb, spec_indices, dtype):
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
