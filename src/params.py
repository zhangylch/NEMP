from src.read_json import load_config
from src.gpu_sel import gpu_sel


full_config = load_config("config.json")
gpu_sel(full_config.local_size)

import jax
import jax.numpy as jnp
from low_level import sparse_tp

# jax.config.update("jax_debug_nans", True)

if full_config.jnp_dtype == "float64":
    jax.config.update("jax_enable_x64", True)

if full_config.jnp_dtype == "float32":
    jax.config.update("jax_default_matmul_precision", "highest")

rmaxl = full_config.max_l + 1
prmaxl = full_config.pmax_l + 1
if full_config.pmax_l > full_config.max_l + 0.5:
    raise RuntimeError("Invalid setting of pmax_l. pmax_l must be less than or equal to rmax_l.")

key = jax.random.PRNGKey(full_config.seed)
key = jax.random.split(key, 2)

index_l = sparse_tp.orbital_index_l(rmaxl)

initbias_neigh = jax.random.uniform(key[0], shape=(full_config.nradial,)) * 12 + 0.01
