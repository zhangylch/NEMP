#! /usr/bin/env python3

import sys
import math
import numpy as np
import model.MPNN as MPNN
from src.params import *
import dataloader.dataloader as dataloader
import dataloader.cudaloader as cudaloader
import src.print_info as print_info
import optax
from src.print_params import print_params
from jax import vmap, jit
from optax import tree_utils as otu
from src.data_config import ModelConfig
from dataclasses import replace, asdict
import json
import orbax.checkpoint as oc
from typing import Optional, Any



# train function
def train(params, ema_params, config, optim, ckpt, ckpt_cpu, ckpt_restore, ckpt_state, ckpt_state_cpu, schedule_fn, value_and_grad_fn, value_fn, data_load, warm_lr, slr, elr, warm_epoch, Epoch, ncyc, ntrain, nval, nprop):

    def train_loop(nstep):

        def optimize_epoch(params, opt_state, ema_params, scale, loss_out, weight, data):

            def body(i, carry):
                params, opt_state, ema_params, scale, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_fn = carry
                inabprop = (iabprop[i] for iabprop in abprop)
                loss, grads = value_and_grad_fn(params, coor[i], cell[i], disp_cell[i], neighlist[i], shiftimage[i], center_factor[i], species[i], inabprop, weight)
                grads = jax.lax.pmean(grads, axis_name="train_GPUs")
                updates, opt_state = optim.update(grads, opt_state, params)
                updates = otu.tree_scalar_mul(scale, updates)
                params = optax.apply_updates(params, updates)
                ema_params = optax.incremental_update(params, ema_params, 0.001)
                loss_fn += loss
                return params, opt_state, ema_params, scale, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_fn
            
            coor, cell, neighlist, shiftimage, center_factor, species, abprop = data
            disp_cell = jnp.zeros_like(cell)
            tmp_params = params
            tmp_state = opt_state
            tmp_ema = ema_params 
            params, opt_state, ema_params, scale, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_out = \
            jax.lax.fori_loop(0, nstep, body, (params, opt_state, ema_params, scale, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_out))
            params, opt_state, ema_params = jax.lax.cond(jnp.isnan(loss_out), lambda _: (tmp_params, tmp_state, tmp_ema), lambda _: (params, opt_state, ema_params), None)
            return params, opt_state, ema_params, loss_out

        return optimize_epoch
        #return optimize_epoch

    
    def val_loop(nstep):
        def get_loss(params, scale, loss_out, ploss_out, weight, data):
            def body(i, carry):
                params, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_fn, ploss_fn = carry
                inabprop = (iabprop[i] for iabprop in abprop)
                loss, ploss = value_fn(params, coor[i], cell[i], disp_cell[i], neighlist[i], shiftimage[i], center_factor[i], species[i], inabprop, weight)
                loss_fn = loss_fn + loss
                ploss_fn = ploss_fn + ploss
                return params, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_fn, ploss_fn

            coor, cell, neighlist, shiftimage, center_factor, species, abprop = data
            disp_cell = jnp.zeros_like(cell)
            params, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_out, ploss_out = \
            jax.lax.fori_loop(0, nstep, body, (params, weight, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, loss_out, ploss_out))
            return loss_out, ploss_out
        return get_loss

    devices = jax.local_devices()
    train_ens = jax.pmap(train_loop(ncyc), axis_name="train_GPUs")
    val_ens = jax.pmap(val_loop(ncyc), axis_name="val_GPUs")

    opt_state = optim.init(params)
    lr_state = schedule_fn.init(params)
    
    params = jax.device_put_replicated(params, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
 
    ema_params = jax.device_put_replicated(ema_params, devices)
    
    print_err = print_info.Print_Info(ferr)

    best_loss = jnp.sum(jnp.array([1e20]))

   
    scale = jax.device_put_replicated(warm_lr / slr, devices)
    max_scale =  slr / warm_lr
    weight = jax.device_put_replicated(init_weight, devices)
    save_step = 0
    for iepoch in range(Epoch): 

        loss_train = jnp.zeros(full_config.local_size)
        for data in data_load:
            params, opt_state, ema_params, loss_train = train_ens(params, opt_state, ema_params, scale, loss_train, weight, data)
        out_train = jnp.sqrt(jnp.sum(loss_train) / ntrain)

        loss_val = jnp.zeros(full_config.local_size)
        ploss_val = jnp.zeros((full_config.local_size, nprop))
        for data in data_load:
            loss_val, ploss_val = val_ens(ema_params, scale, loss_val, ploss_val, weight, data)
        out_val = jnp.sqrt(jnp.sum(loss_val) / nval)
        ploss_out = jnp.sqrt(jnp.sum(ploss_val, axis=0) / nval)



# print and save information
        lr = slr * scale[0]
        print_err(iepoch, lr, out_train, out_val, ploss_out)

        _, lr_state = schedule_fn.update(updates=params, state=lr_state, value=out_val)
        scale = jnp.ones(full_config.local_size) * lr_state.scale * (1.0 / max_scale + min(((iepoch+1) / warm_epoch), 1) * (1.0 - 1.0 / max_scale))

        if iepoch > warm_epoch:
            weight = final_weight + (lr - elr) / (slr - elr) * (init_weight-final_weight)
            weight = jax.device_put_replicated(weight, devices)

        if out_val > 1e1 * best_loss or jnp.isnan(jnp.sum(loss_val)):

            step = ckpt.latest_step()
            restored = ckpt_restore.restore(step)

            params = restored["params"]
            params = jax.device_put_replicated(params, devices)
            ema_params = params
         

        if out_val < best_loss:
            best_loss = out_val
             
            save_step = save_step + 1
            aveparams = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), params)
            ckpt_state["params"] = aveparams
            ckpt_state_cpu["params"] = jax.device_put(aveparams, jax.devices('cpu')[0])
            ckpt.save(save_step, args=oc.args.StandardSave(ckpt_state))
            ckpt_cpu.save(save_step, args=oc.args.StandardSave(ckpt_state_cpu))

            save_step = save_step + 1
            aveparams = jax.tree_util.tree_map(lambda x: jnp.mean(x, axis=0), ema_params)
            ckpt_state["params"] = aveparams
            ckpt_state_cpu["params"] = jax.device_put(aveparams, jax.devices('cpu')[0])
            ckpt.save(save_step, args=oc.args.StandardSave(ckpt_state))
            ckpt_cpu.save(save_step, args=oc.args.StandardSave(ckpt_state_cpu))
            print(f"Step {save_step}: Saved checkpoint")

            

        if lr < elr+1e-10: 
            ckpt.close()
            break
        sys.stdout.flush()

    

key = jax.random.split(key[-1], 2)

data_load = dataloader.Dataloader(full_config.maxneigh, full_config.batchsize, local_size=full_config.local_size, initpot=full_config.initpot, ncyc=full_config.ncyc, cutoff=full_config.cutoff, datafolder=full_config.datafolder, ene_shift=full_config.ene_shift, force_table=full_config.force_table, stress_table=full_config.stress_table, cross_val=full_config.cross_val, jnp_dtype=full_config.jnp_dtype, key=full_config.seed, Fshuffle=full_config.Fshuffle, ntrain=full_config.ntrain)

full_config = replace(full_config, initpot=data_load.initpot)
with open("full_config.json", "w") as f:
    json.dump(asdict(full_config), f, indent=4) 
# get some system information
ntrain = full_config.ntrain
nval = data_load.nval
ncom_spec = data_load.ncom_spec
nspec = data_load.nspec
reduce_spec = data_load.reduce_spec
com_spec = data_load.com_spec
force_std = data_load.std

nprop = 1
if full_config.stress_table:
    nprop = 3
elif full_config.force_table:
    nprop = 2

final_weight = jnp.array(full_config.final_weight[:nprop])
init_weight = jnp.array(full_config.init_weight[:nprop])

data_load = cudaloader.CudaDataLoader(data_load, queue_size=full_config.queue_size)
for data in data_load:
    pass

for data in data_load:
    pass

get_gpu0_data_op = lambda sharded_array: sharded_array[0]
data_on_gpu0_pytree = jax.tree.map(get_gpu0_data_op, data)
coor, cell, neighlist, shiftimage, center_factor, species, abprop = data_on_gpu0_pytree

initdata = (coor[0, 0], cell[0, 0], jnp.zeros((3,3)), neighlist[0, 0], shiftimage[0, 0], center_factor[0, 0], species[0, 0])

#=================================================Equi MPNN===================================================================
config = ModelConfig(ncom_spec=ncom_spec, nspec=nspec, num_cg=num_cg, emb_nl=full_config.emb_nl, MP_nl=full_config.MP_nl, radial_nl=full_config.radial_nl, out_nl=full_config.out_nl, reduce_spec=reduce_spec, com_spec=com_spec, count_l=count_l, index_l=index_l, index_i1=index_i1, index_i2=index_i2, ens_cg=ens_cg, index_add=index_add, index_den=index_den, index_squ=index_squ, initbias_neigh=initbias_neigh, cutoff=full_config.cutoff, nradial=full_config.nradial, nwave=full_config.nwave, rmaxl=rmaxl, prmaxl=prmaxl, MP_loop=full_config.MP_loop, pn=full_config.pn, npaircode=full_config.npaircode, use_norm=full_config.use_norm, use_bias=full_config.use_bias, std=force_std, cst=1.67462)

model = MPNN.MPNN(config)

params_rng = {"params": key[1]}

params = model.init(params_rng, *initdata)

print("NN structure for pes")
print_params(params)

#ceta = jnp.pi/5
#rotate = jnp.array([[1, 0, 0], [0, jnp.cos(ceta), jnp.sin(ceta)], [0, -jnp.sin(ceta), jnp.cos(ceta)]])


if full_config.stress_table:
    def get_force_stress(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        ene, (force, stress) = jax.value_and_grad(model.apply, argnums=[1, 3])(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        volume = jnp.dot(cell[0], jnp.cross(cell[1], cell[2]))
        return ene, force, stress/volume*jnp.array(full_config.stress_sign)
    vmap_model = jax.jit(vmap(get_force_stress, in_axes=(None, 0, 0, 0, 0, 0, 0, 0)))
elif full_config.force_table:
    vmap_model = jax.jit(vmap(jax.value_and_grad(model.apply, argnums=1), in_axes=(None, 0, 0, 0, 0, 0, 0, 0)))
else:
    def get_energy(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species):
        return model.apply(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species),
    vmap_model = jax.jit(vmap(get_energy, in_axes=(None, 0, 0, 0, 0, 0, 0, 0)))

def make_gradient(pes_model):

    def wf_loss(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, weight):

        nnprop = pes_model(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        #coor = jnp.matmul(coor, rotate) 
        #nnprop1 = pes_model(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        #jax.debug.print("target {x} {y}", x=nnprop1[0], y=nnprop[0])
        if full_config.stress_table:
            abpot, abforce, abstress = abprop
            nnpot, nnforce, nnstress = nnprop
            loss = weight[0] * jnp.sum(jnp.square((abpot - nnpot) / jnp.sum(center_factor, axis=1))) \
                 + weight[1] * jnp.sum(jnp.square(abforce - nnforce) / (3 * jnp.sum(center_factor, axis=1)[:, None, None])) \
                 + weight[2] * jnp.sum(jnp.square(abstress - nnstress) / jnp.array(9.0)) 
        elif full_config.force_table:
            abpot, abforce = abprop
            nnpot, nnforce = nnprop
            loss = weight[0] * jnp.sum(jnp.square((abpot - nnpot) / jnp.sum(center_factor, axis=1))) \
                 + weight[1] * jnp.sum(jnp.square(abforce - nnforce) / (3 * jnp.sum(center_factor, axis=1)[:, None, None]))
        else:
            abpot, = abprop
            nnpot, = nnprop
            loss = jnp.sum(jnp.square((abpot - nnpot) / jnp.sum(center_factor, axis=1))) * weight[0]
        
        return loss


    return jax.value_and_grad(wf_loss)
        
value_and_grad_fn = make_gradient(vmap_model)

def make_loss(pes_model, nprop):

    def get_loss(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species, abprop, weight):

        nnprop = pes_model(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species)
        if full_config.stress_table:
            abpot, abforce, abstress = abprop
            nnpot, nnforce, nnstress = nnprop
            loss1 = jnp.sum(jnp.square((abpot - nnpot) / jnp.sum(center_factor, axis=1))) 
            loss2 = jnp.sum(jnp.square(abforce - nnforce) / (3 * jnp.sum(center_factor, axis=1)[:, None, None])) 
            loss3 = jnp.sum(jnp.square(abstress - nnstress) / jnp.array(9.0)) 
            ploss = jnp.stack([loss1, loss2, loss3])
            loss = loss1*weight[0] + loss2*weight[1] + loss3*weight[2]
        elif full_config.force_table:
            abpot, abforce = abprop
            nnpot, nnforce = nnprop
            loss1 = jnp.sum(jnp.square((abpot - nnpot) / jnp.sum(center_factor, axis=1))) 
            loss2 = jnp.sum(jnp.square(abforce - nnforce) / (3 * jnp.sum(center_factor, axis=1)[:, None, None]))
            ploss = jnp.stack([loss1, loss2])
            loss = loss1*weight[0] + loss2*weight[1]
        else:
            abpot, = abprop
            nnpot, = nnprop
            ploss = jnp.sum(jnp.square((abpot - nnpot) / jnp.sum(center_factor, axis=1)))
            loss = ploss * weight[0]
        return loss, ploss


    return get_loss
 
value_fn = make_loss(vmap_model, nprop)       


schedule_fn = optax.contrib.reduce_on_plateau(factor=full_config.decay_factor, patience=full_config.patience_step, cooldown=full_config.cooldown, min_scale=full_config.elr/full_config.slr)

#optim = optax.amsgrad(learning_rate=slr)
optim = optax.chain(optax.add_decayed_weights(full_config.weight_decay), optax.clip_by_global_norm(full_config.clip_norm), optax.amsgrad(learning_rate=full_config.slr))

ferr=open("nn.err","w")
ferr.write("Hybrid Equivariant MPNN package based on three-body descriptors \n")
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))

def to_cpu(x):
    if isinstance(x, (jax.Array, jnp.ndarray)):
        return jax.device_put(x, jax.devices("cpu")[0])
    return x
                                    
cpu_config = jax.tree_util.tree_map(to_cpu, asdict(config))


ckpt_state = {"params":params, "config": asdict(config)}
ckpt_state_cpu = {"params":jax.device_put(params, jax.devices('cpu')[0]), "config": cpu_config}


options = oc.CheckpointManagerOptions(max_to_keep=2, create=True)
ckpt_restore = oc.CheckpointManager(full_config.ckpath, options=options)

ema_params = params
if full_config.restart:
    step = ckpt_restore.latest_step()
    restored = ckpt_restore.restore(step-1)
    params = restored["params"]
    restored = ckpt_restore.restore(step)
    ema_params = restored["params"]
    ckpt_state = restored

ckpt = oc.CheckpointManager(
  oc.test_utils.erase_and_create_empty(full_config.ckpath),
  options=options,
)

ckpt_cpu = oc.CheckpointManager(
  oc.test_utils.erase_and_create_empty(full_config.ckpath_cpu),
  options=options,
)

ckpt.save(0, args=oc.args.StandardSave(ckpt_state))
ckpt_cpu.save(0, args=oc.args.StandardSave(ckpt_state_cpu))


train(params, ema_params, config, optim, ckpt, ckpt_cpu, ckpt_restore, ckpt_state, ckpt_state_cpu, schedule_fn, value_and_grad_fn, value_fn, data_load, full_config.warm_lr, full_config.slr, full_config.elr, full_config.warm_epoch, full_config.Epoch, full_config.ncyc, full_config.ntrain, nval, nprop)
         
ferr.write(time.strftime("%Y-%m-%d-%H_%M_%S \n", time.localtime()))
ferr.close()
print("Normal termination")
