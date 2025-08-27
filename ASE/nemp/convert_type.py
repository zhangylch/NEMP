from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as np

def convert_dtype(obj, jnp_dtype="float32"):

    if "32" in jnp_dtype:
        int_dtype = np.int32
        float_dtype = np.float32
        jnp_float = jnp.float32
        jnp_int = jnp.int32
    else:
        int_dtype = np.int64
        float_dtype = np.float64
        jnp_float = jnp.float64
        jnp_int = jnp.int64

    def _convert(x):
        # JAX arrays
        if isinstance(x, jnp.ndarray):
            if jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(jnp_float)
            elif jnp.issubdtype(x.dtype, jnp.integer):
                return x.astype(jnp_int)
            else:
                return x

        # NumPy arrays
        elif isinstance(x, np.ndarray):
            if np.issubdtype(x.dtype, np.floating):
                return x.astype(np.dtype(float_dtype))
            elif np.issubdtype(x.dtype, np.integer):
                return x.astype(np.dtype(int_dtype))
            else:
                return x

        else:
            return x

    return jax.tree.map(_convert, obj)

