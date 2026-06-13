import cuequivariance as cue
from flax import nnx
import jax
import jax.numpy as jnp

from low_level import cueq_tp


def _old_even_rule_counts(rmaxl, prmaxl):
    count_l = [0] * prmaxl
    for lf in range(prmaxl):
        for li1 in range(rmaxl):
            low = abs(li1 - lf)
            up = min(prmaxl, li1 + lf + 1)
            for li2 in range(low, up):
                if (li1 + li2 + lf) % 2 == 0:
                    count_l[lf] += 1
    return count_l


def _so3_path_counts(rmaxl, prmaxl):
    def irreps(max_l):
        return cue.Irreps("SO3", " + ".join(f"1x{l}" for l in range(max_l)))

    poly = cue.descriptors.fully_connected_tensor_product(
        irreps(rmaxl),
        irreps(prmaxl),
        irreps(prmaxl),
    )
    stp = poly.polynomial.operations[0][1]
    count_l = [0] * prmaxl
    for path in stp.paths:
        count_l[path.indices[3]] += 1
    return count_l


def _o3_natural_parity_path_counts(rmaxl, prmaxl):
    def irreps(max_l):
        return cue.Irreps(
            "O3",
            " + ".join(f"1x{l}{'e' if l % 2 == 0 else 'o'}" for l in range(max_l)),
        )

    poly = cue.descriptors.fully_connected_tensor_product(
        irreps(rmaxl),
        irreps(prmaxl),
        irreps(prmaxl),
    )
    stp = poly.polynomial.operations[0][1]
    count_l = [0] * prmaxl
    for path in stp.paths:
        count_l[path.indices[3]] += 1
    return count_l


def test_o3_natural_parity_matches_previous_even_path_rule():
    rmaxl, prmaxl = 4, 4

    o3_count_l = _o3_natural_parity_path_counts(rmaxl, prmaxl)
    old_count_l = _old_even_rule_counts(rmaxl, prmaxl)
    so3_count_l = _so3_path_counts(rmaxl, prmaxl)

    assert o3_count_l == old_count_l
    assert so3_count_l != old_count_l


def test_radial_mixed_tp_uses_two_input_channel_weight_axes():
    nspec, nwave = 3, 4
    tp = cueq_tp.RadialMixedTP(
        nspec=nspec,
        nwave=nwave,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )

    assert tp.weights.shape == (nspec, tp.num_paths, nwave, nwave)
    assert tp.weight_dim == tp.num_paths * nwave * nwave


def test_radial_mixed_tp_accepts_configured_methods():
    tp_custom = cueq_tp.RadialMixedTP(
        nspec=1,
        nwave=2,
        rmaxl=2,
        prmaxl=2,
        dtype=jnp.float32,
        tp_method="custom",
        rngs=nnx.Rngs(0),
    )
    tp_native = cueq_tp.RadialMixedTP(
        nspec=1,
        nwave=2,
        rmaxl=2,
        prmaxl=2,
        dtype=jnp.float32,
        tp_method="native",
        rngs=nnx.Rngs(0),
    )
    tp_uniform = cueq_tp.RadialMixedTP(
        nspec=1,
        nwave=2,
        rmaxl=2,
        prmaxl=2,
        dtype=jnp.float32,
        tp_method="uniform_1D",
        rngs=nnx.Rngs(0),
    )

    assert tp_custom.tp_method == "custom"
    assert tp_native.tp_method == "naive"
    assert tp_uniform.tp_method == "uniform_1d"


def test_radial_mixed_tp_custom_forward():
    tp = cueq_tp.RadialMixedTP(
        nspec=2,
        nwave=4,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_method="custom",
        rngs=nnx.Rngs(0),
    )

    init_orb = jax.random.normal(jax.random.key(1), (5, 9, 4), dtype=jnp.float32)
    iter_orb = jax.random.normal(jax.random.key(2), (5, 9, 4), dtype=jnp.float32)
    spec_indices = jnp.array([0, 1, 0, 1, 0])
    out = tp(init_orb, iter_orb, spec_indices, jnp.float32)

    assert out.shape == (5, 9, 4)
    assert out.dtype == jnp.float32


def test_radial_mixed_tp_custom_matches_native_forward():
    tp_custom = cueq_tp.RadialMixedTP(
        nspec=2,
        nwave=3,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_method="custom",
        rngs=nnx.Rngs(0),
    )
    tp_native = cueq_tp.RadialMixedTP(
        nspec=2,
        nwave=3,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_method="native",
        rngs=nnx.Rngs(0),
    )

    init_orb = jax.random.normal(jax.random.key(1), (4, 9, 3), dtype=jnp.float32)
    iter_orb = jax.random.normal(jax.random.key(2), (4, 9, 3), dtype=jnp.float32)
    spec_indices = jnp.array([0, 1, 0, 1])
    out_custom = tp_custom(init_orb, iter_orb, spec_indices, jnp.float32)
    out_native = tp_native(init_orb, iter_orb, spec_indices, jnp.float32)

    assert jnp.allclose(out_custom, out_native, atol=2e-5, rtol=2e-5)


def test_radial_mixed_tp_uniform_1d_forward():
    tp = cueq_tp.RadialMixedTP(
        nspec=2,
        nwave=4,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_method="uniform_1D",
        rngs=nnx.Rngs(0),
    )

    init_orb = jax.random.normal(jax.random.key(1), (5, 9, 4), dtype=jnp.float32)
    iter_orb = jax.random.normal(jax.random.key(2), (5, 9, 4), dtype=jnp.float32)
    spec_indices = jnp.array([0, 1, 0, 1, 0])
    out = tp(init_orb, iter_orb, spec_indices, jnp.float32)

    assert out.shape == (5, 9, 4)
    assert out.dtype == jnp.float32


def test_cueq_spherical_harmonics_uses_native_layout():
    vectors = jnp.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 2.0, 3.0],
        ],
        dtype=jnp.float32,
    )
    vectors = vectors / jnp.linalg.norm(vectors, axis=1, keepdims=True)

    index_l = cueq_tp.orbital_index_l(3)
    eps = jnp.array(1e-8, dtype=vectors.dtype)
    sph = cueq_tp.normalized_spherical_harmonics(
        3,
        vectors,
        index_l,
        eps,
    )

    expected = jnp.array(
        [
            [1.0, 1.7320508, 0.0, 0.0, 0.0, 0.0, -1.118034, 0.0, -1.9364917],
            [1.0, 0.0, 1.7320508, 0.0, 0.0, 0.0, 2.236068, 0.0, 0.0],
            [1.0, 0.0, 0.0, 1.7320508, 0.0, 0.0, -1.118034, 0.0, 1.9364917],
            [1.0, 0.46291003, 0.92582005, 1.38873, 0.82992494, 0.55328333, -0.15971905, 1.6598499, 1.1065665],
        ],
        dtype=jnp.float32,
    )

    assert sph.shape == (vectors.shape[0], 9)
    assert jnp.allclose(sph, expected, atol=3e-6, rtol=3e-6)
