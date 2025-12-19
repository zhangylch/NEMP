#! /usr/bin/env python3

import sys
import numpy as np
import inference_model.MPNN as MPNN
import dataloader_eval.dataloader as dataloader
import dataloader_eval.cudaloader as cudaloader
import jax
from jax import vmap, jit
import jax.numpy as jnp
from src.save_checkpoint import save_checkpoint, restore_checkpoint
from src.data_config import ModelConfig
from src.read_json import load_config

# 示例：读取配置文件
full_config = load_config("full_config.json")
if full_config.jnp_dtype=='float64':
    jax.config.update("jax_enable_x64", True)

if full_config.jnp_dtype=='float32':
    jax.config.update("jax_default_matmul_precision", "highest")

data_load = dataloader.Dataloader(full_config.maxneigh_per_node, full_config.batchsize, initpot=full_config.initpot, ncyc=full_config.ncyc, cutoff=full_config.cutoff, datafolder=full_config.datafolder, ene_shift=full_config.ene_shift, force_table=full_config.force_table, cross_val=full_config.cross_val, jnp_dtype=full_config.jnp_dtype, key=full_config.data_seed, ntrain=full_config.ntrain, eval_mode=True)
# generate random data for initialization

#ntrain = data_load.ntrain
numatoms = data_load.numatoms[:full_config.ntrain]
ntrain = jnp.sum(numatoms)
nforce = np.sum(numatoms) * 3

nprop = 1
prop_length = full_config.ntrain
if full_config.force_table:
    nprop = 2
    prop_length = jnp.array(np.array([ntrain, nforce]))


data_load = cudaloader.CudaDataLoader(data_load, queue_size=full_config.queue_size)


devices = jax.local_devices()
restored = restore_checkpoint(
    full_config.ckpath, 
    devices
)

if restored is not None:
    start_step, params, ema_params, opt_state, model_config = restored

#==============================Equi MPNN==============================================================

config = ModelConfig(**model_config)
model = MPNN.MPNN(config)



if full_config.force_table:
    vmap_model = vmap(jax.value_and_grad(model.apply, argnums=1), in_axes=(None, 0, 0, 0, 0, 0, 0, 0))
else:
    def get_energy(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        return model.apply(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species),
    vmap_model = vmap(get_energy, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))

def make_loss(pes_model, nprop):

    def get_loss(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop):

        nnprop = pes_model(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        if full_config.force_table:
            abpot, abforce = abprop
            nnpot, nnforce = nnprop
            #loss1 = jnp.sum(jnp.abs((abpot - nnpot) / jnp.sum(center_factor, axis=1))) 
            loss1 = jnp.sum(jnp.abs((abpot - nnpot)))
            loss2 = jnp.sum(jnp.abs(abforce - nnforce))
            ploss = jnp.stack([loss1, loss2])
        else:
            abpot, = abprop
            nnpot, = nnprop
            jax.debug.print("{x} {y}", x= abpot, y=nnpot)
            ploss = jnp.sum(jnp.abs((abpot - nnpot) / jnp.sum(center_factor, axis=1)))
        
        return ploss


    return get_loss
 
value_fn = make_loss(vmap_model, nprop)       

def val_loop(nstep):
    def get_loss(params, coor, cell, neighlist, shiftimage, center_factor, species, abprop, ploss_out):
        def body(i, carry):
            params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_fn = carry
            inabprop = (iabprop[i] for iabprop in abprop)
            ploss = value_fn(params, coor[i], cell[i], disp_cell[i], neighlist[i], shiftimage[i], center_factor[i], species[i], inabprop)
            ploss_fn = ploss_fn + ploss
            return params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_fn

        disp_cell = jnp.zeros_like(cell)
        params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_out = \
        jax.lax.fori_loop(0, nstep, body, (params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, ploss_out))
        return ploss_out
    return jax.jit(get_loss)


val_ens = val_loop(full_config.ncyc)
ploss_val = jnp.zeros(nprop)        
for data in data_load:
    coor, cell, neighlist, shiftimage, center_factor, species, abprop = data
    ploss_val = val_ens(params, coor, cell, neighlist, shiftimage, center_factor, species, abprop, ploss_val)
    
ploss_val = ploss_val / prop_length

print(ploss_val)


