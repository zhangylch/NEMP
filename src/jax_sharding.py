import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def leading_axis_sharding(devices):
    mesh = Mesh(np.array(devices), ("x",))
    return NamedSharding(mesh, P("x"))


def device_put_sharded(shards, devices):
    """Replacement for deprecated jax.device_put_sharded."""
    sharding = leading_axis_sharding(devices)
    return jax.tree.map(
        lambda *xs: jax.device_put(np.stack(xs), sharding),
        *shards,
    )


def device_put_replicated(x, devices):
    """Replacement for deprecated jax.device_put_replicated."""
    sharding = leading_axis_sharding(devices)
    return jax.tree.map(
        lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding),
        x,
    )
