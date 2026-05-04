import jax
import jax.numpy as jnp
import optax

import train_model.MPNN as MPNN
from src.data_config import ModelConfig


def _minimal_config():
    dtype = jnp.float32
    return ModelConfig(
        nspec=1,
        num_cg=1,
        emb_nl=[0, 4, 1, False],
        MP_nl=[0, 4, 1, True],
        radial_nl=[0, 4, 1, True],
        out_nl=[0, 4, 1, True],
        reduce_spec=jnp.array([1], dtype=dtype),
        com_spec=jnp.array([[1.0, 1.0]], dtype=dtype),
        count_l=jnp.array([1.0], dtype=dtype),
        index_l=jnp.array([0, 1, 1, 1]),
        ens_cg=jnp.array([1.0], dtype=dtype),
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
