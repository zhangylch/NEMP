import os
import jax
import jax.numpy as jnp
from ase import Atoms
from ase.stress import full_3x3_to_voigt_6_stress
from ase.calculators.calculator import Calculator
import numpy as np
import ase.calculators.nemp.MPNN as MPNN
from ase.calculators.nemp.convert_type import convert_dtype
from ase.calculators.nemp.data_config import ModelConfig
from ase.calculators.nemp.read_json import load_config
import orbax.checkpoint as oc
from ase.calculators.calculator import (Calculator, all_changes,
                                        PropertyNotImplementedError)
from flax import traverse_util

def stop_grad(variables, dtype):
    flat_vars = traverse_util.flatten_dict(variables)
    new_vars = {k: jax.lax.stop_gradient(v) for k, v in flat_vars.items()}
    new_vars = {k: v.astype(dtype) for k, v in new_vars.items()}
    return traverse_util.unflatten_dict(new_vars)


class NEMP(Calculator):
    
    implemented_properties = ['energy', 'forces', 'stress']
    
    def __init__(self, skin=1.0, pbc=[0, 0, 0], skin_neigh=20, getneigh=None, fconfig='full_config.json', initatoms=None, properties=['energy', 'forces'], **kwargs):

        Calculator.__init__(self, **kwargs)

        full_config = load_config(fconfig)

        if "32" in full_config.jnp_dtype:
            self.int_dtype = np.int32
            self.np_dtype = np.float32
            self.jnp_dtype = jnp.float32
            if full_config.jnp_dtype=="float32":
                jax.config.update("jax_default_matmul_precision", "highest")
        else:
            self.int_dtype = np.int64
            self.np_dtype = np.float64
            self.jnp_dtype = jnp.float64
            jax.config.update("jax_enable_x64", True)

        if jax.default_backend() == "gpu":
            ckpath = full_config.ckpath
        else:
            ckpath = full_config.ckpath_cpu

        #=======================================================
        self.cutoff = full_config.cutoff
        self.skin = skin
        self.getneigh = getneigh
        self.shifts = None
        self.neighlist = None
        self.species = None
        self.disp_cell = jnp.zeros((3,3))
        self.pbc = np.array(pbc, dtype=self.int_dtype)
        self.old_positions = initatoms.get_positions()
        self.properties = properties
        self.cut_neigh = self._cut_neigh()
        self.maxneigh = int(full_config.maxneigh * self.old_positions.shape[0])
        self.ghostneigh = skin_neigh + self.maxneigh

        # load params 
        options = oc.CheckpointManagerOptions()
        ckpt = oc.CheckpointManager(ckpath, options=options)
        step = ckpt.latest_step()
        restored = ckpt.restore(step)
        params = restored["params"]
        model_config = restored["config"]
        model_config = convert_dtype(model_config, jnp_dtype=full_config.jnp_dtype)
        config = ModelConfig(**model_config)
        model = MPNN.MPNN(config)
        self.params = stop_grad(params, self.jnp_dtype)

        #initialize the model
        cart = self.initialize_system(initatoms)
        self.center_factor = jnp.ones(self.species.shape[0])
        self.cell=jnp.array(np.array(initatoms.get_cell().astype(self.np_dtype)))
        self.inv_cell = jnp.linalg.inv(self.cell)
        positions = initatoms.get_positions()
        positions = jnp.array(positions.astype(self.np_dtype))
        positions, distvec,  skin_judge, neighlist, shifts = self.cut_neigh(positions, self.cell, self.neighlist, self.shifts, jnp.zeros((self.ghostneigh, 3)))
        self.skin_judge = skin_judge
        self.old_distvec = distvec
        key = jax.random.PRNGKey(0)
        params_rng = {"params": key}
        data = (positions, self.cell, self.disp_cell, neighlist, shifts, self.center_factor, self.species)
        tmp_params = model.init(params_rng, *data)
        

        if 'forces' in properties and 'stress' not in properties:
            def get_e_f(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species):
                ene, force = jax.value_and_grad(model.apply, argnums=1)(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species)
                return ene, -force
            self.pes = jax.jit(get_e_f)
        elif 'stress' in properties:
            def get_e_f_s(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species):
                ene, (force, stress) = jax.value_and_grad(model.apply, argnums=[1, 3])(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species)
                return ene, -force, stress
            self.pes = jax.jit(get_e_f_s)
        else:
            def get_energy(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species):
                return model.apply(params, coor, cell, disp_cell, neighlist, shiftimage, center_factor, species),
            self.pes = jax.jit(get_energy)

        
    def initialize_system(self, atoms, system_changes=["cell"]):
        positions = atoms.get_positions()
        if "cell" in system_changes:
            cell = atoms.get_cell()
            self.getneigh.init_neigh(self.cutoff+self.skin, self.cutoff+self.skin, cell.T, self.pbc)
            self.cell = jax.device_put(cell.astype(self.np_dtype))

        cart, neighlist, shifts, scutnum = self.getneigh.get_neigh(positions.T, np.int32(self.ghostneigh))

        self.neighlist= jax.device_put(neighlist.astype(self.int_dtype))
        self.shifts = jax.device_put(shifts.T.astype(self.np_dtype))
        self.species = jax.device_put(atoms.get_atomic_numbers().astype(self.int_dtype))
        return cart.T
        
    def _cut_neigh(self):
        def __cut_neigh(positions, cell, neighlist, shifts, old_distvec):    

            expand_cart = positions[neighlist]
            distvec = expand_cart[1] - expand_cart[0] + jnp.matmul(shifts, cell)
            distances = jnp.linalg.norm(distvec, axis=1)
            index_list = jnp.nonzero(jnp.logical_and(jnp.less(distances, self.cutoff), jnp.greater(distances, jnp.array(1e-3))), size=self.maxneigh, fill_value=self.ghostneigh-1)[0]
            skin_judge = jnp.greater(jnp.max(jnp.abs(distvec - old_distvec)), self.skin/2.0)
            return positions, distvec, skin_judge, neighlist[:, index_list], shifts[index_list]
        return jax.jit(__cut_neigh)


    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            
        Calculator.calculate(self, atoms, properties, system_changes)
        
        
        if "cell" in system_changes:
            self.inv_cell = jnp.linalg.inv(self.cell)

        if self.skin_judge:
            cart = self.initialize_system(atoms, system_changes)
            atoms.set_positions(cart)

        positions = atoms.get_positions()
        positions = jax.device_put(positions.astype(self.np_dtype))

        positions, distvec, skin_judge, neighlist, shifts = self.cut_neigh(positions, self.cell, self.neighlist, self.shifts, self.old_distvec)
        self.skin_judge = skin_judge
        if self.skin_judge:
            print("hello")
            self.old_distvec = distvec

        results = self.pes(self.params, positions, self.cell, self.disp_cell, neighlist, shifts, self.center_factor, self.species)
        # Convert results back to numpy arrays for ASE
        results = jax.device_get(jax.lax.stop_gradient(results))
        

        for name, iprop in zip(self.properties, results):
            if "stress" in name:
                virial = iprop
                stress = virial/atoms.get_volume()
                iprop = full_3x3_to_voigt_6_stress(stress)
            self.results[name] = iprop
