from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from inference_model.MPNN import MPNN as InferenceMPNN
from train_model.MPNN import MPNN
from src.data_config import checkpoint_params
from src.jax_sharding import device_put_pmap_replicated
from src.save_checkpoint import restore_checkpoint, save_checkpoint
from test_nnx_mpnn_smoke import _minimal_config


def _graph_data():
    cart = jnp.array([[0.1, 0.2, 0.3], [1.1, 0.4, -0.2]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32)[None] * 3.0
    return (
        cart, cell, jnp.zeros_like(cell), jnp.array([[0, 1], [1, 0]]),
        jnp.zeros(2, dtype=jnp.int32),
        jnp.array([[-1.0, 1.0], [0.0, 0.0], [0.0, 0.0]], dtype=jnp.float32),
        jnp.ones(2, dtype=jnp.float32), jnp.ones(2, dtype=jnp.int32),
    )


def _assert_same_state(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(got, want)


def test_training_and_inference_share_parameters_energy_and_derivatives():
    config = _minimal_config()
    train_graph, params = nnx.split(MPNN(config, rngs=nnx.Rngs(7)), nnx.Param)
    inference_graph, inference_params = nnx.split(
        InferenceMPNN(config, rngs=nnx.Rngs(7)), nnx.Param
    )
    _assert_same_state(params, inference_params)
    cart, cell, disp, neigh, cells, shifts, factors, species = _graph_data()

    def training_energy(x, strain):
        return nnx.merge(train_graph, params)(
            x, cell, strain[None], neigh, cells, shifts, factors, species
        )[0]

    def inference_energy(x, strain):
        return nnx.merge(inference_graph, params)(
            x, cell[0], strain, neigh, shifts.T, factors, species
        )

    train_value, train_grads = jax.value_and_grad(training_energy, (0, 1))(cart, disp[0])
    infer_value, infer_grads = jax.value_and_grad(inference_energy, (0, 1))(cart, disp[0])
    # Existing training/inference eps values differ (1e-6 versus 1e-8).
    np.testing.assert_allclose(infer_value, train_value, rtol=2e-4, atol=2e-5)
    for got, want in zip(infer_grads, train_grads):
        np.testing.assert_allclose(got, want, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("format", ["state", "plain", "wrapped", "string_indices"])
def test_checkpoint_parameter_formats_preserve_template(format):
    config = _minimal_config()
    _, expected = nnx.split(MPNN(config, rngs=nnx.Rngs(7)), nnx.Param)
    graph, template = nnx.split(MPNN(config, rngs=nnx.Rngs(42)), nnx.Param)
    before = jax.tree.map(lambda value: np.array(value, copy=True), template)
    payload = expected if format == "state" else nnx.to_pure_dict(expected)
    if format == "wrapped":
        payload = {"params": payload}
    elif format == "string_indices":
        def stringify_keys(value):
            if isinstance(value, Mapping):
                return {str(key): stringify_keys(item) for key, item in value.items()}
            return value
        payload = stringify_keys(payload)

    restored = checkpoint_params(payload, template)
    _assert_same_state(restored, expected)
    _assert_same_state(template, before)
    assert jnp.isfinite(nnx.merge(graph, restored)(*_graph_data())[0])


def test_pickle_checkpoint_restores_native_parameters(tmp_path):
    config = _minimal_config()
    graph, params = nnx.split(MPNN(config, rngs=nnx.Rngs(7)), nnx.Param)
    ema = jax.tree.map(lambda value: value * 0.9, params)
    save_checkpoint(3, params, ema, (), config, str(tmp_path))
    restored = restore_checkpoint(str(tmp_path), jax.local_devices())
    assert restored is not None
    step, loaded_params, loaded_ema, _, _ = restored
    assert step == 3
    _assert_same_state(checkpoint_params(loaded_params, params), params)
    _assert_same_state(checkpoint_params(loaded_ema, params), ema)
    np.testing.assert_array_equal(
        nnx.merge(graph, checkpoint_params(loaded_ema, params))(*_graph_data())[0],
        nnx.merge(graph, ema)(*_graph_data())[0],
    )


def test_jitted_force_loss_parameter_gradient_and_cpu_pmap():
    graph, params = nnx.split(MPNN(_minimal_config(), rngs=nnx.Rngs(7)), nnx.Param)
    data = _graph_data()

    def energy(state, positions):
        return nnx.merge(graph, state)(positions, *data[1:])[0]

    def loss(state):
        force = -jax.grad(energy, argnums=1)(state, data[0])
        return jnp.sum(jnp.square(force - 0.1))

    value, grads = jax.jit(jax.value_and_grad(loss))(params)
    assert jnp.isfinite(value)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree.leaves(grads))
    assert any(bool(jnp.any(leaf != 0)) for leaf in jax.tree.leaves(grads))

    devices = [jax.devices("cpu")[0]]
    replicated = device_put_pmap_replicated(params, devices)
    mapped_energy = jax.pmap(lambda state: energy(state, data[0]), devices=devices)(replicated)
    np.testing.assert_allclose(mapped_energy[0], energy(params, data[0]), rtol=1e-5)
