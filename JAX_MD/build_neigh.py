import os
import jax
import jax.numpy as jnp
import numpy as np

class Neighlist():
    
    def __init__(self, cutoff=5.0, skin=0.5, maxneigh=100, skin_neigh=20, pbc=[1, 1, 1], getneigh=None, jnp_dtype="float32"):

        if "32" in jnp_dtype:
            self.int_dtype = np.int32
            self.jnp_dtype = np.float32
        else:
            self.int_dtype = np.int64
            self.jnp_dtype = np.float64

        #=======================================================
        self.cutoff = cutoff
        self.r_cutoff = cutoff + skin
        self.getneigh = getneigh
        self.maxneigh = maxneigh
        self.ghostneigh = skin_neigh + maxneigh
        self.pbc = np.array(pbc).astype(np.int32)

        #initialize the model
        self.old_cell=np.zeros((3,3))
        
    def update(self, positions, cell):
        if np.sum(np.square(cell - self.old_cell)) > 1e-5:
            self.getneigh.init_neigh(self.r_cutoff, self.r_cutoff, cell.T, self.pbc)
            self.old_cell = cell
            

        positions = np.asarray(positions, dtype=np.float64)
        cart, neighlist, shifts, scutnum = self.getneigh.get_neigh(positions.T, np.int32(self.ghostneigh))
        
        return neighlist.astype(self.int_dtype)
            
    # cut the reduant neighbors due to the skin
    def cut_neigh(self, positions, cell, neighlist):
    
        cell_inv = jnp.linalg.inv(cell)
        fract_coor = jnp.dot(positions, cell_inv)
        expand_cart = fract_coor[neighlist]
        distvec = expand_cart[1] - expand_cart[0]
        shifts = -jnp.round(distvec)
        distvec = distvec + shifts
        distvec = jnp.dot(distvec, cell)
        distances = jnp.linalg.norm(distvec, axis=1)
        index_list = jnp.nonzero(jnp.logical_and(jnp.less(distances, self.cutoff), jnp.greater(distances, jnp.array(1e-3))), size=self.maxneigh, fill_value=self.ghostneigh-1)[0]
        return neighlist[:, index_list], shifts[index_list]

