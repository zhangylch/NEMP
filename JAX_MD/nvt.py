from src.gpu_sel import gpu_sel
gpu_sel(1)

import jax
import jax.numpy as jnp
from jax_md import simulate, space, partition, quantity
from jax import random, jit
from flax import traverse_util
import numpy as np
import threading
import queue
import time
from functools import partial
import orbax.checkpoint as oc
from src.data_config import ModelConfig
from src.read_json import load_config
from ASE.nemp.convert_type import convert_dtype
import inference_model.MPNN as MPNN
import JAX_MD.build_neigh as build_neigh
import fortran.jax_0_7_1.getneigh as getneigh

from ase import Atoms
from ase.io import extxyz
#import faulthandler
#faulthandler.enable()


def stop_grad(variables):
    flat_vars = traverse_util.flatten_dict(variables)
    new_vars = {k: jax.lax.stop_gradient(v) for k, v in flat_vars.items()}
    return traverse_util.unflatten_dict(new_vars)


#UNIT DEFINATION the default mass is amu, so if you use eV as your output energy unitand angstrom as the unit of your coordinates, then the unit of time is around 10.18fs


#give the parameters
skin = 1.0
temperature = 0.025 # in the unit of energy of your ML 
maxneigh = 7500 * 8
skin_neigh = 7500 * 8
steps_per_block = 10  # step for update the neighlist
total_blocks = 500000  # step_per_block * total_block will the total number of MD step
# system setting
ev_kt=11604.568449040902
time_fs = 10.180507
target_temp = 300.0  # kelvin
print_traj = True
time_step_fs = 0.2 # fs
print_step = 200 # fs
logfile = open("md.log", 'w')
trajfile = open("traj.extxyz", 'w')
thermo_time = 20000  # fs
tau = 400


#convert to eV
target_temp = target_temp/ev_kt
thermo_step = thermo_time / time_step_fs
time_step = time_step_fs / time_fs
tau = tau * time_step




full_config = load_config("full_config.json")
if full_config.jnp_dtype=='float64':
    jax.config.update("jax_enable_x64", True)

if full_config.jnp_dtype=='float32':
    jax.config.update("jax_default_matmul_precision", "highest")

if "32" in full_config.jnp_dtype:
    int_dtype = np.int32
    float_dtype = np.float32
    jnp_dtype = jnp.float32
else:
    int_dtype = np.int64
    float_dtype = np.float64
    jnp_dtype = jnp.float64


fileobj=open("water.xyz",'r')
configuration = extxyz.read_extxyz(fileobj,index=slice(0,1))
atoms = next(configuration)
atoms = atoms.repeat((2, 2, 2))

key = random.PRNGKey(0)

positions = atoms.get_positions()
cell = np.array(atoms.get_cell())
disp_cell = jnp.zeros((3,3))
species = jnp.array(atoms.get_atomic_numbers().astype(int_dtype))
masses = jnp.array(atoms.get_masses().astype(float_dtype))
center_factor = jnp.ones(species.shape[0])
cell_gpu = jnp.array(cell)

# get the shift_fn
displacement_fn, shift_fn = space.periodic(cell.diagonal())

#====load model======================================================
if jax.default_backend() == 'gpu':
    ckpath = full_config.ckpath
else:
    ckpath = full_config.ckpath_cpu

options = oc.CheckpointManagerOptions()
ckpt = oc.CheckpointManager(ckpath, options=options)
step = ckpt.latest_step()
restored = ckpt.restore(step)
params = restored["params"]
params = stop_grad(params)
model_config = restored["config"]
model_config = convert_dtype(model_config, jnp_dtype=full_config.jnp_dtype)

config = ModelConfig(**model_config)

model = MPNN.MPNN(config)

neigh_fn = build_neigh.Neighlist(cutoff=full_config.cutoff, skin=skin, maxneigh=maxneigh, skin_neigh=skin_neigh, pbc=[1, 1, 1], getneigh=getneigh, jnp_dtype=full_config.jnp_dtype)

neighlist = neigh_fn.update(positions, cell)
positions_gpu = jnp.array(positions.astype(float_dtype))
neighlist = jnp.array(neighlist)

def energy_fn(positions, neighlist=neighlist):
    cut_neighlist, cut_shifts = neigh_fn.cut_neigh(positions, cell_gpu, neighlist)
    return model.apply(params, positions, cell_gpu, disp_cell, jax.lax.stop_gradient(cut_neighlist), jax.lax.stop_gradient(cut_shifts), center_factor, species)

init_fn, apply_fn = simulate.nvt_nose_hoover(energy_fn, shift_fn, time_step, target_temp, tau=tau)

state = init_fn(key, positions_gpu, mass=masses)

# for npt and cell opt another box in state are required
#state = init_fn(key, positions, temperature, cell=cell_gpu, neighlist=cut_neighlist, mass=masses, box=cell)


def simulation_step(i, carry):
    state, neighlist = carry

    state = apply_fn(state, neighlist=neighlist)

    return state, neighlist


def ens_mdstep(state, neighlist):
    state, neighlist = jax.lax.fori_loop(0, steps_per_block, simulation_step, (state, neighlist))
    return state

jit_md = jax.jit(ens_mdstep)


position_queue = queue.Queue(maxsize=1)
neighbor_list_queue = queue.Queue(maxsize=1)

def cpu_worker():
    while True:
        data = position_queue.get()
        if data is None:
            break

        positions_cpu, step, temp = jax.device_get(data)
        if np.mod(step, print_step) < 0.5 and step > thermo_step:
            logfile.write("time = {:10.3f}, temperature = {:7.2f} K \n".format(step * time_step_fs/1000.0, temp * ev_kt * 2.0 / (3.0 * positions_cpu.shape[0])))
            atoms.positions = positions_cpu
            extxyz.write_extxyz(trajfile, atoms)

        data = neigh_fn.update(positions_cpu, cell)

        neighbor_list_queue.put(jax.device_put(data))

worker_thread = threading.Thread(target=cpu_worker)
worker_thread.start()

print("--- Starting Advanced Pipeline Simulation ---")


state = jit_md(state, neighlist)

step = 0    
temp = target_temp * 1.5 * positions.shape[0]
start_time = time.time()
for i in range(total_blocks):
    position_queue.put((state.position, step, temp))

    state = jit_md(state, neighlist)
    
    neighlist = neighbor_list_queue.get()
    
    temp = quantity.kinetic_energy(velocity=state.velocity, mass=state.mass)
    step = step + steps_per_block
    
end_time = time.time()

position_queue.put(None)
worker_thread.join()

total_duration = start_time - end_time
print("\n--- Simulation Finished ---")
print(f"Total time for {total_blocks * steps_per_block} steps: {total_duration:.4f} seconds.")
logfile.close()
trajfile.close()
