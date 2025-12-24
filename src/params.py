import time
import numpy as np
import math
import sys

from src.read_json import load_config
from src.gpu_sel import gpu_sel
# 示例：读取配置文件
full_config = load_config("config.json")
gpu_sel(full_config.local_size)

import jax 
jax.config.update("jax_debug_nans", True)
import jax.numpy as jnp
from low_level import sph_cal, cg_cal
import jax.numpy as jnp
from src.data_config import ModelConfig

if full_config.jnp_dtype=='float64':
    jax.config.update("jax_enable_x64", True)

if full_config.jnp_dtype=='float32':
    jax.config.update("jax_default_matmul_precision", "highest")

rmaxl = full_config.max_l + 1
prmaxl = full_config.pmax_l + 1

def contract_sph(rmaxl, prmaxl):

    index_i1 = []
    index_i2 = []
    cg_array = []
    index_den = []
    index_squ = []
    index_add = []
    count_l = jnp.zeros(prmaxl)
    initbias_cg = []
    num_coeff = 0
    num_den = 0
    for lf in range(prmaxl):
        for li1 in range(rmaxl):
            low = abs(li1 - lf)
            up = min(prmaxl, li1+lf+1)
            for li2 in range(low, up):
                if np.mod(li1+li2+lf, 2) < 0.5:       
                    cg = cg_cal.clebsch_gordan(li1, li2, lf)
                    count_l = count_l.at[lf].add(1)        
                    for mf in range(0, 2*lf+1):
                        dim3 = lf * lf + mf
                        index_add.append(dim3)
                        index_squ.append(num_coeff)

                        for mi1 in range(0, 2*li1+1):
                            for mi2 in range(0, 2*li2+1):
                                dim2 = li2 * li2 +  mi2 
                                dim1 = li1 * li1 +  mi1
                                if np.abs(cg[mi1, mi2, mf]) > 1e-3:
                                    index_i1.append(dim1)
                                    index_i2.append(dim2)
                                    cg_array.append(cg[mi1, mi2, mf])
                                    index_den.append(num_den)
                                    print(lf, mf, li1, mi1, li2, mi2, cg[mi1, mi2, mf])
                        num_den += 1
                    num_coeff += 1
         
    index_i1 = jnp.asarray(index_i1)
    index_i2 = jnp.asarray(index_i2)
    ens_cg = jnp.asarray(cg_array)
    index_add = jnp.asarray(index_add)
    index_den = jnp.asarray(index_den)
    index_squ = jnp.asarray(index_squ)

    index_l = jnp.arange(rmaxl*rmaxl)
    for l in range(rmaxl):
        index_l = index_l.at[l*l:(l+1)*(l+1)].set(l)

    return index_i1, index_i2, ens_cg, index_add, index_den, index_squ, initbias_cg, index_l, count_l, num_coeff



key = jax.random.PRNGKey(full_config.seed)
key = jax.random.split(key, 2)

index_i1, index_i2, ens_cg, index_add, index_den, index_squ, initbias_cg, index_l, count_l, num_cg= \
    contract_sph(rmaxl, prmaxl)

initbias_neigh = jax.random.uniform(key[0], shape=(full_config.nradial,)) * 12 + 0.01

sph_pes=sph_cal.SPH_CAL(max_l = rmaxl-1)

print("comnbine number of l: ", num_cg)

