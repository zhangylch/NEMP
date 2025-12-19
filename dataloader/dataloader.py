import jax 
import jax.numpy as jnp
import numpy as np
import dataloader.read_xyz as read_xyz
import fortran.getneigh as getneigh


class Dataloader():
    def __init__(self, maxneigh_per_node, batchsize, local_size=1, ncyc=5, initpot=0.0, cutoff=5.0, datafolder="./", ene_shift=True, force_table=True, stress_table=False, cross_val=True, jnp_dtype="float32", seed=0, eval_mode=False, Fshuffle=True, ntrain=10,  capacity=1.5, node_cap=1.0, edge_cap=1.0):
            
        self.cutoff = cutoff
        self.capacity = capacity
        self.batchsize = batchsize
        self.ncyc = ncyc
        self.local_size = local_size
        self.force_table = force_table
        self.stress_table = stress_table
        self.cross_val = cross_val
        self.seed = seed
        self.maxneigh_per_node = maxneigh_per_node


        if "32" in jnp_dtype:
            self.int_dtype = np.int32
            self.float_dtype = np.float32
        else:
            self.int_dtype = np.int64
            self.float_dtype = np.float64

        coordinates, cell, pbc, species, numatoms, pot, force_list, stress =  \
        read_xyz.read_xyz(datafolder, force_table=force_table, stress_table=stress_table)
        
        numatoms = np.array(numatoms)
        ave_node = np.mean(numatoms)
        self.batchnode = int(node_cap * ave_node * batchsize)
        self.numpoint = numatoms.shape[0]
        pot = np.array(pot)
        
        if ene_shift:
            if not eval_mode:
                initpot = np.sum(pot)/np.sum(numatoms)
            pot = pot - initpot * numatoms
        else:
            pot = pot - initpot

        self.initpot = initpot
        self.numatoms = np.array(numatoms).astype(self.int_dtype)
        self.pbc = np.array(pbc)
        self.coordinates = coordinates
        self.maxnumatom = np.max(self.numatoms)
        self.maxneigh = int(maxneigh_per_node * ave_node * batchsize / edge_cap)
        cell = np.array(cell)
        expand_species = np.ones((self.numpoint, self.maxnumatom), dtype=self.int_dtype)

        if force_table:
            force = np.zeros((self.numpoint, self.maxnumatom, 3))

        # The purpose of these codes is to process conformational data consisting of different numbers of atoms into a regular tensor.
        for i in range(self.numpoint):
            expand_species[i, 0:self.numatoms[i]] = np.array(species[i], dtype=self.int_dtype)
            expand_species[i, self.numatoms[i]:] = expand_species[i, 0]
            if force_table:
                force[i, 0:self.numatoms[i]] = -force_list[i]
        
        if force_table:
            self.std = np.sqrt(np.sum(np.square(force)) / (3*np.sum(self.numatoms)))
        else:
            self.std = 1.0
 
        #  statical over species
        reduce_spec = np.unique(expand_species)
        self.nspec = reduce_spec.shape[0]
        self.reduce_spec = jnp.array(reduce_spec.astype(self.int_dtype))
        x, y = jnp.meshgrid(self.reduce_spec, self.reduce_spec)
        self.com_spec = jnp.array(jnp.stack([y.ravel(), x.ravel()], axis=1).astype(self.float_dtype))
       
 
        if Fshuffle:
            self.shuffle_list = np.random.RandomState(seed=self.seed).permutation(self.numpoint)
        else: 
            self.shuffle_list = np.arange(self.numpoint) 

        self.size_per_step = self.batchsize * ncyc * local_size
        self.length = int(self.numpoint / self.size_per_step)
        self.train_length = int(ntrain / self.size_per_step)
        self.ntrain = ntrain
        self.nval = int((self.numpoint - self.ntrain) / self.size_per_step) * self.size_per_step

        self.species = expand_species
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
        uppoint = self.ipoint + self.size_per_step
        if uppoint < self.numpoint + 0.5:
            coor = np.zeros((self.local_size, self.ncyc, self.batchnode, 3))
            if self.force_table: force = np.zeros((self.local_size, self.ncyc, self.batchnode, 3))
            species = np.zeros((self.local_size, self.ncyc, self.batchnode))
            center_factor = np.ones((self.local_size, self.ncyc, self.batchnode))
            neighlist = np.ones((self.local_size, self.ncyc, 2, self.maxneigh), dtype=np.int32)
            celllist = np.ones((self.local_size, self.ncyc, self.batchnode), dtype=np.int32)
            shiftimage = np.zeros((self.local_size, self.ncyc, 3, self.maxneigh))
            cell = np.zeros((self.local_size, self.ncyc, self.batchsize, 3, 3))
            stress = np.zeros((self.local_size, self.ncyc, self.batchsize, 3, 3))
            pot = np.zeros((self.local_size, self.ncyc, self.batchsize))
            numatoms = np.ones((self.local_size, self.ncyc, self.batchsize))
            for igpu in range(self.local_size):
                for icyc in range(self.ncyc):
                    inode = 0
                    ineigh = 0
                    ibatch = 0
                    while True:
                        if  ibatch > self.batchsize-0.5: break
                        inum = self.shuffle_list[self.ipoint]
                        numatom = self.numatoms[inum]
                        if ineigh + self.maxneigh_per_node * numatom > self.maxneigh + 0.5 or inode + numatom > self.batchnode + 0.5: break
                        icell = self.cell[inum].T
                        icart = self.coordinates[inum]
                        ipbc = self.pbc[inum]
                        getneigh.init_neigh(self.cutoff, self.cutoff, icell, ipbc, self.capacity)
                        cart, tmp, tmp1, scutnum = getneigh.get_neigh(icart, np.int32(self.maxneigh_per_node * numatom))
                        coor[igpu, icyc, inode:inode+numatom] = cart.T
                        if self.force_table: force[igpu, icyc, inode:inode+numatom] = self.force[inum, :numatom]
                        if self.stress_table: stress[igpu, icyc, ibatch] = self.stress[inum]
                        species[igpu, icyc, inode:inode+numatom] = self.species[inum, :numatom]
                        cell[igpu, icyc, ibatch] = self.cell[inum]
                        celllist[igpu, icyc, inode:inode+numatom] = ibatch
                        neighlist[igpu, icyc, :, ineigh:ineigh+scutnum] = tmp[:, :scutnum] + inode
                        shiftimage[igpu, icyc, :, ineigh:ineigh+scutnum] = tmp1[:, :scutnum]
                        pot[igpu, icyc, ibatch] = self.pot[inum]
                        numatoms[igpu, icyc, ibatch] = numatom

                        self.ipoint +=1
                        inode += numatom
                        ibatch +=1
                        ineigh += scutnum
                    center_factor[igpu, icyc, inode:] = np.array(0.0, dtype = self.float_dtype)
                    neighlist[igpu, icyc, :, ineigh:] = inode-1
                    celllist[igpu, icyc, inode:] = ibatch-1

            abprop = (pot,)
            if self.force_table:
                abprop = abprop + (force,)
            if self.stress_table:
                abprop = abprop + (stress,)
             
            return coor.astype(self.float_dtype), cell.astype(self.float_dtype), neighlist.astype(self.int_dtype), celllist.astype(self.int_dtype), shiftimage.astype(self.float_dtype), center_factor.astype(self.float_dtype), species.astype(self.float_dtype), numatoms, abprop
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

