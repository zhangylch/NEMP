import jax
import jax.numpy as jnp
import optax

import train_model.MPNN as MPNN
from low_level import sparse_tp
from src.data_config import ModelConfig


def _minimal_config():
    dtype = jnp.float32
    return ModelConfig(
        nspec=1,
        emb_nl=[0, 4, 1, False],
        MP_nl=[0, 4, 1, True],
        radial_nl=[0, 4, 1, True],
        out_nl=[0, 4, 1, True],
        reduce_spec=jnp.array([1], dtype=dtype),
        com_spec=jnp.array([[1.0, 1.0]], dtype=dtype),
        index_l=jnp.array([0, 1, 1, 1]),
        initbias_neigh=jnp.array([0.5, 1.5], dtype=dtype),
        use_norm=False,
        use_bias=False,
        cutoff=4.0,
        cst=1.0,
        std=1.0,
        nwave=2,
        npaircode=2,
        nradial=2,
        rmaxl=2,
        prmaxl=1,
        MP_loop=1,
        pn=2,
    )


def test_nnx_mpnn_init_apply_and_grad():
    config = _minimal_config()
    model = MPNN.MPNN(config)

    cart = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=jnp.float32)
    cell = jnp.eye(3, dtype=jnp.float32)[None, :, :] * 8.0
    disp_cell = jnp.zeros_like(cell)
    neighlist = jnp.array([[0, 1], [1, 0]])
    celllist = jnp.array([0, 0])
    shiftimage = jnp.zeros((3, 2), dtype=jnp.float32)
    center_factor = jnp.ones((2,), dtype=jnp.float32)
    species = jnp.array([1, 1])

    params = model.init(
        {"params": jax.random.PRNGKey(0)},
        cart,
        cell,
        disp_cell,
        neighlist,
        celllist,
        shiftimage,
        center_factor,
        species,
    )
    total_energy, graph_energy = model.apply(
        params,
        cart,
        cell,
        disp_cell,
        neighlist,
        celllist,
        shiftimage,
        center_factor,
        species,
    )

    assert total_energy.shape == ()
    assert graph_energy.shape == (1,)
    assert jnp.isfinite(total_energy)

    grad_cart = jax.grad(lambda x: model.apply(
        params,
        x,
        cell,
        disp_cell,
        neighlist,
        celllist,
        shiftimage,
        center_factor,
        species,
    )[0])(cart)
    assert grad_cart.shape == cart.shape

    def loss_fn(model_params):
        return model.apply(
            model_params,
            cart,
            cell,
            disp_cell,
            neighlist,
            celllist,
            shiftimage,
            center_factor,
            species,
        )[0]

    _, grads = jax.value_and_grad(loss_fn)(params)
    optimizer = optax.sgd(1e-3)
    opt_state = optimizer.init(params)
    updates, _ = optimizer.update(grads, opt_state, params)
    updated_params = optax.apply_updates(params, updates)
    assert type(updated_params) is type(params)


def test_nnx_mpnn_nonperiodic_rotation_invariant_energy_and_force():
    dtype = jnp.float32
    config = ModelConfig(
        nspec=1,
        emb_nl=[0, 4, 1, False],
        MP_nl=[0, 4, 1, True],
        radial_nl=[0, 4, 1, True],
        out_nl=[0, 4, 1, True],
        reduce_spec=jnp.array([1], dtype=dtype),
        com_spec=jnp.array([[1.0, 1.0]], dtype=dtype),
        index_l=sparse_tp.orbital_index_l(3),
        initbias_neigh=jnp.array([0.5, 1.5], dtype=dtype),
        use_norm=False,
        use_bias=False,
        cutoff=6.0,
        cst=1.0,
        std=1.0,
        nwave=2,
        npaircode=2,
        nradial=2,
        rmaxl=3,
        prmaxl=3,
        MP_loop=1,
        pn=2,
    )
    model = MPNN.MPNN(config)

    cart = jnp.array(
        [
            [-0.4, 0.1, 0.0],
            [0.8, -0.2, 0.3],
            [0.2, 0.9, -0.5],
        ],
        dtype=dtype,
    )
    cell = jnp.eye(3, dtype=dtype)[None, :, :] * 100.0
    disp_cell = jnp.zeros_like(cell)
    neighlist = jnp.array(
        [
            [0, 0, 1, 1, 2, 2],
            [1, 2, 0, 2, 0, 1],
        ],
        dtype=jnp.int32,
    )
    celllist = jnp.zeros((cart.shape[0],), dtype=jnp.int32)
    shiftimage = jnp.zeros((3, neighlist.shape[1]), dtype=dtype)
    center_factor = jnp.ones((cart.shape[0],), dtype=dtype)
    species = jnp.ones((cart.shape[0],), dtype=jnp.int32)

    angle = jnp.array(0.73, dtype=dtype)
    axis = jnp.array([0.2, -0.3, 0.7], dtype=dtype)
    axis = axis / jnp.linalg.norm(axis)
    kx = jnp.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=dtype,
    )
    rot = jnp.eye(3, dtype=dtype) + jnp.sin(angle) * kx + (1.0 - jnp.cos(angle)) * (kx @ kx)
    rotated_cart = cart @ rot.T

    params = model.init(
        {"params": jax.random.PRNGKey(1)},
        cart,
        cell,
        disp_cell,
        neighlist,
        celllist,
        shiftimage,
        center_factor,
        species,
    )

    def energy_fn(x):
        return model.apply(
            params,
            x,
            cell,
            disp_cell,
            neighlist,
            celllist,
            shiftimage,
            center_factor,
            species,
        )[0]

    energy = energy_fn(cart)
    rotated_energy = energy_fn(rotated_cart)
    grad_cart = jax.grad(energy_fn)(cart)
    grad_rotated = jax.grad(energy_fn)(rotated_cart)

    assert jnp.allclose(energy, rotated_energy, atol=5e-4, rtol=5e-4)
    assert jnp.allclose(grad_rotated, grad_cart @ rot.T, atol=1e-3, rtol=1e-3)
