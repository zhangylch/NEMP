#! /usr/bin/env python3

import sys
import numpy as np
from src.read_json import load_config
from src.gpu_sel import gpu_sel

full_config = load_config("full_config.json")
gpu_sel(full_config.local_size)

import train_model.MPNN as MPNN
import dataloader.dataloader as dataloader
import dataloader.cudaloader as cudaloader
import jax
import jax.numpy as jnp
from src.save_checkpoint import restore_checkpoint
from src.data_config import ModelConfig, checkpoint_tp_config
from src.jax_sharding import device_put_pmap_replicated, get_jax_devices

# Configure JAX precision.
if full_config.jnp_dtype=='float64':
    jax.config.update("jax_enable_x64", True)

if full_config.jnp_dtype=='float32':
    jax.config.update("jax_default_matmul_precision", "highest")

data_load = dataloader.Dataloader(full_config.maxneigh_per_node, full_config.batchsize, local_size=full_config.local_size, initpot=full_config.initpot, ncyc=full_config.ncyc, cutoff=full_config.cutoff, datafolder=full_config.datafolder, ene_shift=full_config.ene_shift, force_table=full_config.force_table, stress_table=full_config.stress_table, cross_val=full_config.cross_val, jnp_dtype=full_config.jnp_dtype, seed=full_config.data_seed, Fshuffle=False, ntrain=full_config.ntrain, eval_mode=True, node_cap=full_config.node_cap, edge_cap=full_config.edge_cap)
# generate random data for initialization

#ntrain = data_load.ntrain
numatoms = data_load.numatoms[:full_config.ntrain]
ntrain = np.sum(numatoms)
nforce = np.sum(numatoms) * 3

nprop = 1
prop_length = full_config.ntrain
if full_config.stress_table:
    nprop = 3
    prop_length = jnp.array(np.array([ntrain, nforce, full_config.ntrain*9]))
elif full_config.force_table:
    nprop = 2
    prop_length = jnp.array(np.array([ntrain, nforce]))

data_load = cudaloader.CudaDataLoader(data_load, queue_size=full_config.queue_size)


devices = get_jax_devices(full_config.local_size, log=True)
restored = restore_checkpoint(
    full_config.ckpath, 
    devices
)

if restored is not None:
    start_step, params, ema_params, opt_state, model_config = restored

#==============================Equi MPNN==============================================================
model_config = checkpoint_tp_config(model_config, full_config)
config = ModelConfig(**model_config)

model = MPNN.MPNN(config)

if full_config.stress_table:
    def pes_model(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        (_, ene), (force, stress) = jax.value_and_grad(model.apply, argnums=[1, 3], has_aux=True)(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        volume = jnp.sum(cell[:, 0] * jnp.cross(cell[:, 1], cell[:, 2]), axis=-1)
        return ene, force, stress/volume[:, None, None]*jnp.array(full_config.stress_sign)
elif full_config.force_table:
    def pes_model(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        (_, ene), force = jax.value_and_grad(model.apply, argnums=1, has_aux=True)(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        return ene, force
else:
    def pes_model(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species):
        _, ene = model.apply(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        return ene,

def make_loss(pes_model, nprop):

    def get_loss(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop):

        nnprop = pes_model(params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species)
        ploss = jnp.zeros(nprop)
        for i, iprop in enumerate(abprop):
            ploss = ploss.at[i].set(jnp.sum(jnp.abs(nnprop[i] - iprop)))
        
        return ploss


    return get_loss
 
value_fn = make_loss(pes_model, nprop)

def val_loop(nstep):
    def get_loss(params, ploss_out, data):
        def body(i, carry):
            params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_fn = carry
            inabprop = (iabprop[i] for iabprop in abprop)
            ploss = value_fn(params, coor[i], cell[i], disp_cell[i], neighlist[i], celllist[i], shiftimage[i], center_factor[i], species[i], inabprop)
            ploss_fn = ploss_fn + ploss
            return params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_fn

        coor, cell, neighlist, celllist, shiftimage, center_factor, species, numatoms, abprop = data
        disp_cell = jnp.zeros_like(cell)
        params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_out = \
        jax.lax.fori_loop(0, nstep, body, (params, coor, cell, disp_cell, neighlist, celllist, shiftimage, center_factor, species, abprop, ploss_out))
        return ploss_out
    return get_loss


val_ens = jax.pmap(val_loop(full_config.ncyc), axis_name="eval_GPUs")
ploss_val = device_put_pmap_replicated(jnp.zeros((nprop,)), devices)
for data in data_load:
    ploss_val = val_ens(ema_params, ploss_val, data)
    print(ploss_val, flush=True)

ploss_val = jnp.sum(ploss_val, axis=0) / prop_length
print(ploss_val)

