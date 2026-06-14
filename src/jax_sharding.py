import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P


def leading_axis_sharding(devices):
    mesh = Mesh(np.array(devices), ("x",))
    return NamedSharding(mesh, P("x"))


def get_jax_devices(expected_local_size=None, log=False):
    devices = jax.local_devices()
    if log:
        device_info = [
            f"{device.id}:{device.platform}:{getattr(device, 'device_kind', 'unknown')}"
            for device in devices
        ]
        print(f"JAX local devices ({len(devices)}): {device_info}", flush=True)
        if not any(device.platform == "gpu" for device in devices):
            print("WARNING: JAX did not find a GPU; execution will run on CPU.", flush=True)
    if expected_local_size is not None and len(devices) != expected_local_size:
        raise RuntimeError(
            "JAX local device count does not match config.local_size: "
            f"{len(devices)} vs {expected_local_size}. Check CUDA_VISIBLE_DEVICES, "
            "Slurm GPU allocation, and the installed JAX CUDA runtime."
        )
    return devices


def device_put_leading_axis_sharded(x, sharding):
    """Put a pytree whose leading axis is already the device axis."""
    return jax.tree.map(lambda y: jax.device_put(y, sharding), x)


def device_put_pmap_replicated(x, devices):
    """Replicate a pytree along the leading axis for pmap inputs."""
    sharding = leading_axis_sharding(devices)
    return jax.tree.map(
        lambda y: jax.device_put(jnp.stack([y] * len(devices)), sharding),
        x,
    )
