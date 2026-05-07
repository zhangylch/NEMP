import cuequivariance as cue
import cuequivariance_jax as cuex
import jax.numpy as jnp
from flax import nnx
from jax.ops import segment_sum


LAYOUT = cue.IrrepsLayout.ir_mul


def parity_irreps(max_l, nwave):
    terms = [f"{nwave}x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_l)]
    return cue.Irreps("O3", " + ".join(terms))


def tensor_product_descriptor(rmaxl, prmaxl, nwave, normalize=True):
    descriptor = cue.descriptors.fully_connected_tensor_product(
        parity_irreps(rmaxl, nwave),
        parity_irreps(prmaxl, nwave),
        parity_irreps(prmaxl, nwave),
    )
    if normalize:
        operation, stp = descriptor.polynomial.operations[0]
        polynomial = cue.SegmentedPolynomial(
            descriptor.polynomial.inputs,
            descriptor.polynomial.outputs,
            [(operation, stp.normalize_paths_for_operand(1))],
        )
        descriptor = cue.EquivariantPolynomial(
            descriptor.inputs,
            descriptor.outputs,
            polynomial,
        )
    return descriptor


def tensor_product_path_metadata(rmaxl, prmaxl, nwave):
    poly = tensor_product_descriptor(rmaxl, prmaxl, nwave)
    stp = poly.polynomial.operations[0][1]
    count_l = jnp.zeros(prmaxl)
    for path in stp.paths:
        count_l = count_l.at[path.indices[3]].add(1)
    return stp.num_paths, count_l


def orbital_index_l(max_l):
    index_l = jnp.arange(max_l * max_l)
    for l in range(max_l):
        index_l = index_l.at[l * l : (l + 1) * (l + 1)].set(l)
    return index_l


def density_cg(index_l):
    return jnp.reciprocal(jnp.sqrt(2.0 * index_l + 1.0))


def spherical_harmonics(max_l, vectors):
    vector_rep = cuex.RepArray(
        cue.Irreps("O3", "1o"),
        vectors[:, [1, 2, 0]],
        LAYOUT,
    )
    harmonics = cuex.spherical_harmonics(
        list(range(max_l)),
        vector_rep,
        normalize=False,
    )
    return harmonics.array.T


def normalized_spherical_harmonics(max_l, vectors, index_l, eps):
    sph = spherical_harmonics(max_l, vectors)
    sph_norm = segment_sum(
        jnp.square(sph),
        index_l,
        num_segments=max_l,
        indices_are_sorted=True,
    )
    sph_norm = sph_norm + eps
    l_value = index_l.astype(sph.dtype)
    return sph / jnp.sqrt(sph_norm[index_l]) * jnp.sqrt(2.0 * l_value[:, None] + 1.0)


def _diagonal_channel_weights(l_coeff, nwave):
    l_coeff = jnp.moveaxis(l_coeff, 1, 0)
    num_nodes, num_paths, _ = l_coeff.shape
    weights = jnp.zeros((num_nodes, num_paths, nwave, nwave, nwave), dtype=l_coeff.dtype)
    channel = jnp.arange(nwave)
    weights = weights.at[:, :, channel, channel, channel].set(l_coeff)
    return weights.reshape(num_nodes, num_paths * nwave * nwave * nwave)


class RadialMixedTP(nnx.Module):
    def __init__(self, nspec, nwave, rmaxl, prmaxl, dtype, *, rngs):
        self.nwave = nnx.static(nwave)
        self.rmaxl = nnx.static(rmaxl)
        self.prmaxl = nnx.static(prmaxl)
        self.init_irreps = nnx.static(parity_irreps(rmaxl, nwave))
        self.iter_irreps = nnx.static(parity_irreps(prmaxl, nwave))
        self.descriptor = nnx.static(tensor_product_descriptor(rmaxl, prmaxl, nwave))

        stp = self.descriptor.polynomial.operations[0][1]
        self.num_paths = nnx.static(stp.num_paths)
        self.weights = nnx.Param(
            nnx.initializers.normal(1.0)(
                rngs.params(),
                (stp.num_paths, nspec, nwave),
                dtype,
            )
        )

    def __call__(self, init_orb, iter_orb, spec_indices, dtype):
        num_nodes = init_orb.shape[0]
        weights = _diagonal_channel_weights(self.weights[...][:, spec_indices], self.nwave)

        weight_rep = cuex.RepArray(self.descriptor.inputs[0], weights, LAYOUT)
        init_rep = cuex.RepArray(
            self.init_irreps,
            init_orb.reshape(num_nodes, self.rmaxl * self.rmaxl * self.nwave),
            LAYOUT,
        )
        iter_rep = cuex.RepArray(
            self.iter_irreps,
            iter_orb.reshape(num_nodes, self.prmaxl * self.prmaxl * self.nwave),
            LAYOUT,
        )

        output = cuex.equivariant_polynomial(
            self.descriptor,
            [weight_rep, init_rep, iter_rep],
            method="naive",
            math_dtype=jnp.dtype(dtype).name,
        )
        return output.array.reshape(num_nodes, self.prmaxl * self.prmaxl, self.nwave)
