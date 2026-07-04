import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from low_level import sparse_tp


def _old_even_rule_counts(rmaxl, prmaxl):
    count_l = [0] * prmaxl
    for out_l in range(prmaxl):
        for init_l in range(rmaxl):
            low = abs(init_l - out_l)
            high = min(prmaxl, init_l + out_l + 1)
            for iter_l in range(low, high):
                if (init_l + iter_l + out_l) % 2 == 0:
                    count_l[out_l] += 1
    return count_l


def test_sparse_paths_match_previous_even_path_rule():
    rmaxl, prmaxl = 4, 4
    paths, count_l = sparse_tp.sparse_cg_paths(rmaxl, prmaxl)

    assert list(count_l) == _old_even_rule_counts(rmaxl, prmaxl)
    assert len(paths) == sum(count_l)


def test_sparse_paths_store_one_dimensional_cg_terms():
    paths, _ = sparse_tp.sparse_cg_paths(3, 3)
    path = next(path for path in paths if path[4])
    mi, mj, mk, coefficients = path[4], path[5], path[6], path[7]

    assert isinstance(mi, tuple)
    assert isinstance(mj, tuple)
    assert isinstance(mk, tuple)
    assert isinstance(coefficients, tuple)
    assert len(mi) == len(mj) == len(mk) == len(coefficients)
    assert all(isinstance(value, int) for value in mi + mj + mk)
    assert all(isinstance(value, float) for value in coefficients)


def test_radial_mixed_tp_full_uses_two_channel_weight_axes():
    nspec, nwave = 3, 4
    tp = sparse_tp.RadialMixedTP(
        nspec=nspec,
        nwave=nwave,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_mode="full",
        rngs=nnx.Rngs(0),
    )

    assert tp.tp_mode == "full"
    assert not tp.init_channel_first
    assert tp.weights.shape == (nspec, tp.num_paths, nwave, nwave)
    assert np.isclose(tp.weight_norm, 1.0 / np.sqrt(nwave))
    assert tp.weight_dim == tp.num_paths * nwave * nwave


def test_radial_mixed_tp_channelwise_uses_single_channel_weight_axis():
    nspec, nwave = 3, 4
    tp = sparse_tp.RadialMixedTP(
        nspec=nspec,
        nwave=nwave,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_mode="channelwise",
        rngs=nnx.Rngs(0),
    )

    assert tp.tp_mode == "channelwise"
    assert tp.weights.shape == (nspec, tp.num_paths, nwave)
    assert tp.weight_norm == 1.0
    assert tp.init_mix.shape == (nspec, 3, nwave, nwave)
    assert tp.iter_mix.shape == (nspec, 3, nwave, nwave)
    assert tp.weight_dim == tp.num_paths * nwave


def test_radial_mixed_tp_rejects_native_backend():
    with pytest.raises(ValueError):
        sparse_tp.normalize_tp_method("native")


def test_radial_mixed_tp_rejects_channelwise_uniform_1d():
    try:
        from low_level import cueq_tp
    except (ImportError, OSError) as exc:
        pytest.skip(f"cuequivariance JAX runtime is unavailable: {exc}")

    with pytest.raises(ValueError):
        cueq_tp.RadialMixedTP(
            nspec=1,
            nwave=2,
            rmaxl=2,
            prmaxl=2,
            dtype=jnp.float32,
            tp_method="uniform_1d",
            tp_mode="channelwise",
            rngs=nnx.Rngs(0),
        )


def test_radial_mixed_tp_full_forward():
    tp = sparse_tp.RadialMixedTP(
        nspec=2,
        nwave=4,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_mode="full",
        rngs=nnx.Rngs(0),
    )

    init_orb = jax.random.normal(jax.random.key(1), (5, 9, 4), dtype=jnp.float32)
    iter_orb = jax.random.normal(jax.random.key(2), (5, 9, 4), dtype=jnp.float32)
    spec_indices = jnp.array([0, 1, 0, 1, 0])
    out = tp(init_orb, iter_orb, spec_indices, jnp.float32)

    assert out.shape == (5, 9, 4)
    assert out.dtype == jnp.float32


def test_radial_mixed_tp_channelwise_forward():
    tp = sparse_tp.RadialMixedTP(
        nspec=2,
        nwave=4,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_mode="channelwise",
        rngs=nnx.Rngs(0),
    )

    init_orb = jax.random.normal(jax.random.key(1), (5, 9, 4), dtype=jnp.float32)
    iter_orb = jax.random.normal(jax.random.key(2), (5, 9, 4), dtype=jnp.float32)
    spec_indices = jnp.array([0, 1, 0, 1, 0])
    out = tp(init_orb, iter_orb, spec_indices, jnp.float32)

    assert out.shape == (5, 9, 4)
    assert out.dtype == jnp.float32


def test_radial_mixed_tp_uniform_1d_matches_custom_forward():
    try:
        from low_level import cueq_tp
    except (ImportError, OSError) as exc:
        pytest.skip(f"cuequivariance JAX runtime is unavailable: {exc}")
    if not any(device.platform == "gpu" for device in jax.local_devices()):
        pytest.skip("uniform_1d requires a GPU-backed cuequivariance_ops_jax runtime")

    tp_custom = sparse_tp.RadialMixedTP(
        nspec=2,
        nwave=3,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        rngs=nnx.Rngs(0),
    )
    tp_uniform = cueq_tp.RadialMixedTP(
        nspec=2,
        nwave=3,
        rmaxl=3,
        prmaxl=3,
        dtype=jnp.float32,
        tp_method="uniform_1d",
        rngs=nnx.Rngs(0),
    )

    init_orb = jax.random.normal(jax.random.key(1), (4, 9, 3), dtype=jnp.float32)
    iter_orb = jax.random.normal(jax.random.key(2), (4, 9, 3), dtype=jnp.float32)
    spec_indices = jnp.array([0, 1, 0, 1])
    out_custom = tp_custom(init_orb, iter_orb, spec_indices, jnp.float32)
    out_uniform = tp_uniform(jnp.swapaxes(init_orb, 1, 2), iter_orb, spec_indices, jnp.float32)

    assert tp_uniform.init_channel_first
    assert jnp.allclose(out_custom, out_uniform, atol=2e-5, rtol=2e-5)


def test_local_spherical_harmonics_layout():
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

    index_l = sparse_tp.orbital_index_l(3)
    eps = jnp.array(1e-8, dtype=vectors.dtype)
    sph = sparse_tp.normalized_spherical_harmonics(
        3,
        vectors,
        index_l,
        eps,
    )

    assert sph.shape == (vectors.shape[0], 9)
    for l_value in range(3):
        block = sph[:, l_value * l_value : (l_value + 1) * (l_value + 1)]
        expected = jnp.full((vectors.shape[0],), 2 * l_value + 1, dtype=sph.dtype)
        assert jnp.allclose(jnp.sum(jnp.square(block), axis=1), expected, atol=1e-5)
