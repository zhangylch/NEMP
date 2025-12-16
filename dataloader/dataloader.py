import jax 
import jax.numpy as jnp
import numpy as np
import dataloader.read_xyz as read_xyz
import fortran.getneigh as getneigh


class Dataloader():
    def __init__(self, maxneigh, batchsize, local_size=1, ncyc=5, initpot=0.0, cutoff=5.0, datafolder="./", ene_shift=True, force_table=True, stress_table=False, cross_val=True, jnp_dtype="float32", seed=0, eval_mode=False, Fshuffle=True, ntrain=10,  capacity=1.5):
            
        self.cutoff = cutoff
        self.capacity = capacity
        self.batchsize = batchsize
        self.ncyc = ncyc
        self.local_size = local_size
        self.force_table = force_table
        self.stress_table = stress_table
        self.cross_val = cross_val
        self.seed = seed


        if "32" in jnp_dtype:
            self.int_dtype = np.int32
            self.float_dtype = np.float32
        else:
            self.int_dtype = np.int64
            self.float_dtype = np.float64

        coordinates, cell, pbc, species, numatoms, pot, force_list, stress =  \
        read_xyz.read_xyz(datafolder, force_table=force_table, stress_table=stress_table)
        
        numatoms = np.array(numatoms)
        self.numpoint = numatoms.shape[0]
        pot = np.array(pot)
        f1 = open("energy.txt", 'w')
        for ipot in pot:
            f1.write("{} \n".format(ipot))
        f1.close()
        
        if ene_shift:
            if not eval_mode:
                initpot = np.sum(pot)/np.sum(numatoms)
            pot = pot - initpot * numatoms
        else:
            pot = pot - initpot

        self.initpot = initpot
        self.numatoms = np.array(numatoms)
        self.pbc = np.array(pbc)
        self.coordinates = coordinates
        self.maxnumatom = np.max(self.numatoms)
        self.maxneigh = maxneigh * self.maxnumatom
        cell = np.array(cell)
        expand_species = np.ones((self.numpoint, self.maxnumatom), dtype=self.int_dtype)
        center_factor = np.ones((self.numpoint, self.maxnumatom))

        if force_table:
            force = np.zeros((self.numpoint, self.maxnumatom, 3))

        # The purpose of these codes is to process conformational data consisting of different numbers of atoms into a regular tensor.
        for i in range(self.numpoint):
            expand_species[i, 0:self.numatoms[i]] = np.array(species[i], dtype=self.int_dtype)
            expand_species[i, self.numatoms[i]:] = expand_species[i, 0]
            if force_table:
                force[i, 0:self.numatoms[i]] = -force_list[i]
            center_factor[i, self.numatoms[i]:] = 0.0
        
        if force_table:
            self.std = jnp.sqrt(jnp.sum(jnp.square(force)) / (3*jnp.sum(self.numatoms)))
        else:
            self.std = 1.0
 
        #  statical over species
        reduce_spec = np.unique(expand_species)
        self.nspec = reduce_spec.shape[0]
        self.reduce_spec = jnp.array(reduce_spec.astype(self.int_dtype))
        x, y = jnp.meshgrid(self.reduce_spec, self.reduce_spec)
        self.com_spec = jnp.array(jnp.stack([y.ravel(), x.ravel()], axis=1).astype(self.float_dtype))
        print(self.com_spec)
       
 
        if Fshuffle:
            self.shuffle_list = np.random.RandomState(seed=self.seed).permutation(self.numpoint)
        else: 
            self.shuffle_list = np.arange(self.numpoint) 

        self.length = int(self.numpoint / self.batchsize)
        self.train_length = int(ntrain / self.batchsize)
        self.ntrain = ntrain
        self.nval = int((self.numpoint - self.ntrain) / self.batchsize) * self.batchsize

        self.species = expand_species
        self.center_factor = center_factor.astype(self.int_dtype)
        if force_table: self.force = force.astype(self.float_dtype)
        if stress_table: self.stress = np.array(stress).astype(self.float_dtype)
        self.cell = cell
        self.pot = pot.astype(self.float_dtype)
         
        print("initpot = {} \n".format(initpot))
        print("reduce_spec = {} \n".format(self.reduce_spec))
      
    def __iter__(self):
        self.ipoint = 0
        return self

    def __next__(self):
        uppoint = self.ipoint + self.batchsize
        if uppoint < self.numpoint + 0.5:
            index_batch = self.shuffle_list[self.ipoint:uppoint]
            coor = np.zeros((self.batchsize, self.maxnumatom, 3))
            neighlist = np.ones((self.batchsize, 2, self.maxneigh), dtype=np.int32)
            shiftimage = np.zeros((self.batchsize, 3, self.maxneigh))
            for inum in range(self.batchsize):
                i = index_batch[inum]
                icell = self.cell[i].T
                icart = self.coordinates[i]
                ipbc = self.pbc[i]
                getneigh.init_neigh(self.cutoff, self.cutoff, icell, ipbc, self.capacity)
                cart, tmp, tmp1, scutnum = getneigh.get_neigh(icart, np.int32(self.maxneigh))
                coor[inum, :self.numatoms[i]] = cart.T
                neighlist[inum] = tmp
                shiftimage[inum] = tmp1

            species = self.species[index_batch].reshape(self.local_size, self.ncyc, -1, self.maxnumatom)
            center_factor = self.center_factor[index_batch].reshape(self.local_size, self.ncyc, -1, self.maxnumatom)
            cell = self.cell[index_batch].reshape(self.local_size, self.ncyc, -1, 3, 3).astype(self.float_dtype)
            shiftimage = shiftimage.transpose((0, 2, 1)).reshape(self.local_size, self.ncyc, -1, self.maxneigh, 3).astype(self.float_dtype)
            coor = coor.reshape(self.local_size, self.ncyc, -1, self.maxnumatom, 3).astype(self.float_dtype)
            neighlist = neighlist.reshape(self.local_size, self.ncyc, -1, 2, self.maxneigh).astype(self.int_dtype)
            pot = self.pot[index_batch].reshape(self.local_size, self.ncyc, -1)
            abprop = (pot,)
            if self.force_table:
                force = self.force[index_batch].reshape(self.local_size, self.ncyc, -1, self.maxnumatom, 3)
                abprop = abprop + (force,)
            if self.stress_table:
                stress = self.stress[index_batch].reshape(self.local_size, self.ncyc, -1, 3, 3)
                abprop = abprop + (stress,)
             
            self.ipoint = uppoint
            return coor, cell, neighlist, shiftimage, center_factor, species, abprop
        else:
            if self.cross_val:
                self.seed = self.seed+1
                self.shuffle_list = np.random.RandomState(seed=self.seed).permutation(self.numpoint)
            else:
                self.seed = self.seed+1
                shuffle_list1 = np.random.RandomState(seed=self.seed).permutation(self.shuffle_list[:self.ntrain])
                self.shuffle_list[:self.ntrain] = shuffle_list1
                shuffle_list1 = np.random.RandomState(seed=self.seed).permutation(self.shuffle_list[self.ntrain:])
                self.shuffle_list[self.ntrain:] = shuffle_list1
            raise StopIteration

